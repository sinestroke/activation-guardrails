"""Probe modules and masked aggregation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROBE_NAMES = ("mean", "softmax", "attention", "rmattn", "swim", "sctopk")


@dataclass(frozen=True)
class ProbeHyperparams:
    M: int = 16
    tau_swim: float = 1.0
    K: int = 8
    tau_s: float = 2.0
    lambda_segvar: float = 0.01
    gamma_ema: float = 0.1
    streaming_reduction: str = "max"
    attention_logit_scale: str = "none"
    rmattn_window: int = 10
    rmattn_hidden: int = 100
    rmattn_heads: int = 10
    rmattn_eval_aggregation: str = "rolling"


def probe_hyperparams_from_mapping(values: dict[str, Any] | None = None) -> ProbeHyperparams:
    known = {field.name for field in fields(ProbeHyperparams)}
    return ProbeHyperparams(**{key: value for key, value in (values or {}).items() if key in known})


def init_normal_(param: nn.Parameter, std: float = 0.02) -> None:
    with torch.no_grad():
        param.normal_(mean=0.0, std=std)


def fan_in_std(n_layers: int, hidden_size: int) -> float:
    return 1.0 / math.sqrt(max(1, int(n_layers) * int(hidden_size)))


def einsum_inputs(acts: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    if acts.dtype == weight.dtype or torch.is_autocast_enabled():
        return acts, weight
    if acts.device.type == "cuda":
        return acts, weight.to(dtype=acts.dtype)
    return acts.float(), weight


def masked_mean(logits: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    counts = mask.sum(dim=1).clamp_min(1)
    summed = logits.masked_fill(~mask, 0.0).sum(dim=1)
    mean = summed / counts
    empty = ~mask.any(dim=1)
    if empty.any():
        mean = torch.where(empty, logits.mean(dim=1), mean)
    return mean.float()


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    mask = mask.bool()
    masked_scores = scores.float().masked_fill(~mask, -torch.inf)
    empty = ~mask.any(dim=dim, keepdim=True)
    masked_scores = torch.where(empty, torch.zeros_like(masked_scores), masked_scores)
    weights = torch.softmax(masked_scores, dim=dim).masked_fill(~mask, 0.0)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return weights / denom


def normalized_attention_entropy(weights: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    mask = mask.bool()
    valid_counts = mask.sum(dim=dim).clamp_min(1).float()
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=dim)
    norm = valid_counts.clamp_min(2.0).log()
    return entropy / norm


def attention_weight_stats(weights: Tensor, mask: Tensor) -> dict[str, float]:
    mask = mask.bool()
    peak = weights.max(dim=1).values
    entropy = normalized_attention_entropy(weights, mask, dim=1)
    return {
        "attn_peak": float(peak.detach().mean().cpu()),
        "attn_entropy": float(entropy.detach().mean().cpu()),
    }


def window_attention_weight_stats(weights: Tensor, mask: Tensor, valid_windows: Tensor) -> dict[str, float]:
    valid = valid_windows.bool()
    if not valid.any():
        return {"attn_peak": float("nan"), "attn_entropy": float("nan")}
    selected = weights[valid].detach()
    selected_mask = mask[valid].bool()
    if selected.ndim == 3:
        selected_mask = selected_mask.unsqueeze(1).expand_as(selected)
        selected = selected.reshape(-1, selected.shape[-1])
        selected_mask = selected_mask.reshape(-1, selected_mask.shape[-1])
    return attention_weight_stats(selected, selected_mask)


def sliding_window_mean(logits: Tensor, end_mask: Tensor, width: int) -> tuple[Tensor, Tensor]:
    if width <= 0:
        raise ValueError("Window width must be positive")
    batch, seq_len = logits.shape
    means = logits.new_zeros((batch, seq_len), dtype=torch.float32)
    valid = torch.zeros((batch, seq_len), device=logits.device, dtype=torch.bool)
    if seq_len < width:
        return means, valid

    token_mask = end_mask.bool()
    masked_logits = logits.float().masked_fill(~token_mask, 0.0)
    cumsum = F.pad(masked_logits.cumsum(dim=1), (1, 0))
    sums = cumsum[:, width:] - cumsum[:, :-width]
    means[:, width - 1 :] = sums / float(width)
    mask_cumsum = F.pad(token_mask.long().cumsum(dim=1), (1, 0))
    window_counts = mask_cumsum[:, width:] - mask_cumsum[:, :-width]
    valid[:, width - 1 :] = window_counts == width
    return means, valid


def bce_with_logits_per_example(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels.float()
    return F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")


def window_fallback_fraction(window_mask: Tensor) -> float:
    fallback = ~window_mask.any(dim=1)
    return float(fallback.float().mean().detach().cpu())


class ProbeBase(nn.Module):
    probe_name: str = "base"

    def __init__(self, n_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)

    def feature_spec(self, site: str, include_embedding_layer: bool = False) -> dict[str, Any]:
        return {
            "site": site,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "include_embedding_layer": include_embedding_layer,
        }

    def token_logits(self, acts: Tensor) -> Tensor:
        raise NotImplementedError

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        raise NotImplementedError

    def score(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        return torch.sigmoid(self.score_logits(acts, probe_mask, hp))


class LinearTokenProbe(ProbeBase):
    def __init__(self, n_layers: int, hidden_size: int) -> None:
        super().__init__(n_layers, hidden_size)
        self.W = nn.Parameter(torch.empty(n_layers, hidden_size, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float32))
        init_normal_(self.W, fan_in_std(n_layers, hidden_size))

    def token_logits(self, acts: Tensor) -> Tensor:
        acts_for_einsum, weight = einsum_inputs(acts, self.W)
        return torch.einsum("bslh,lh->bs", acts_for_einsum, weight).float() + self.b


class MeanProbe(LinearTokenProbe):
    probe_name = "mean"

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        return probe_loss_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, labels, hp)

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        return probe_score_logits_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, hp)


class SoftmaxProbe(LinearTokenProbe):
    probe_name = "softmax"

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        return probe_loss_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, labels, hp)

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        return probe_score_logits_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, hp)


class AttentionProbe(LinearTokenProbe):
    probe_name = "attention"

    def __init__(self, n_layers: int, hidden_size: int) -> None:
        super().__init__(n_layers, hidden_size)
        self.W_m = nn.Parameter(torch.empty(n_layers, hidden_size, dtype=torch.float32))
        self.b_m = nn.Parameter(torch.zeros((), dtype=torch.float32))
        init_normal_(self.W_m, fan_in_std(n_layers, hidden_size))

    def attention_logits(self, acts: Tensor, scale_mode: str = "none") -> Tensor:
        acts_for_einsum, weight = einsum_inputs(acts, self.W_m)
        logits = torch.einsum("bslh,lh->bs", acts_for_einsum, weight).float() + self.b_m
        mode = str(scale_mode).lower().replace("-", "_")
        if mode in {"none", "off", "unscaled"}:
            return logits
        if mode in {"sqrt_feature_dim", "sqrt_fan_in", "scaled"}:
            return logits / math.sqrt(max(1, self.n_layers * self.hidden_size))
        raise ValueError(f"Unknown legacy attention_logit_scale={scale_mode!r}")

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        z = self.token_logits(acts)
        m = self.attention_logits(acts, hp.attention_logit_scale)
        return probe_loss_from_token_logits(
            self.probe_name,
            z,
            probe_mask,
            labels,
            hp,
            attention_logits=m,
        )

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        z = self.token_logits(acts)
        m = self.attention_logits(acts, hp.attention_logit_scale)
        return probe_score_logits_from_token_logits(
            self.probe_name,
            z,
            probe_mask,
            hp,
            attention_logits=m,
        )


class RMAttnProbe(ProbeBase):
    probe_name = "rmattn"

    def __init__(self, n_layers: int, hidden_size: int, hidden: int = 100, heads: int = 10) -> None:
        super().__init__(n_layers, hidden_size)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.W1 = nn.Parameter(torch.empty(n_layers, hidden_size, self.hidden, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.empty(self.hidden, self.hidden, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float32))
        self.W_q = nn.Parameter(torch.empty(self.heads, self.hidden, dtype=torch.float32))
        self.W_v = nn.Parameter(torch.empty(self.heads, self.hidden, dtype=torch.float32))
        self.b_out = nn.Parameter(torch.zeros((), dtype=torch.float32))
        init_normal_(self.W1, fan_in_std(n_layers, hidden_size))
        init_normal_(self.W2, 1.0 / math.sqrt(max(1, self.hidden)))
        init_normal_(self.W_q, 1.0 / math.sqrt(max(1, self.hidden)))
        init_normal_(self.W_v, 1.0 / math.sqrt(max(1, self.hidden)))

    def transformed_acts(self, acts: Tensor) -> Tensor:
        acts_for_einsum, weight = einsum_inputs(acts, self.W1)
        hidden = torch.einsum("bslh,lhr->bsr", acts_for_einsum, weight).float() + self.b1
        hidden = torch.relu(hidden)
        hidden = hidden @ self.W2 + self.b2
        return torch.relu(hidden)

    def head_logits(self, y: Tensor) -> tuple[Tensor, Tensor]:
        query = torch.einsum("bsd,hd->bsh", y.float(), self.W_q)
        value = torch.einsum("bsd,hd->bsh", y.float(), self.W_v)
        return query, value

    def token_logits(self, acts: Tensor) -> Tensor:
        _, values = self.head_logits(self.transformed_acts(acts))
        return values.sum(dim=-1) + self.b_out

    def fallback_logits(self, values: Tensor, probe_mask: Tensor) -> Tensor:
        return masked_mean(values.sum(dim=-1), probe_mask) + self.b_out

    def rolling_aggregation(
        self,
        values: Tensor,
        query: Tensor,
        probe_mask: Tensor,
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        batch, seq_len, _heads = values.shape
        fallback = self.fallback_logits(values, probe_mask)
        if seq_len < width:
            valid = torch.zeros((batch, 0), dtype=torch.bool, device=values.device)
            return fallback, valid, None, None

        value_windows = values.unfold(dimension=1, size=width, step=1)
        query_windows = query.unfold(dimension=1, size=width, step=1)
        mask_windows = probe_mask.unfold(dimension=1, size=width, step=1).bool()
        weights = masked_softmax(query_windows, mask_windows.unsqueeze(2), dim=-1)
        window_logits = (weights * value_windows.float()).sum(dim=-1)
        valid = probe_mask[:, width - 1 :].bool()
        masked = window_logits.masked_fill(~valid.unsqueeze(-1), -torch.inf)
        per_head = masked.max(dim=1).values
        agg = per_head.sum(dim=-1) + self.b_out
        no_window = ~valid.any(dim=1)
        agg = torch.where(no_window, fallback, agg)
        return agg, valid, weights, mask_windows

    def multimax_aggregation(self, values: Tensor, probe_mask: Tensor) -> Tensor:
        masked = values.masked_fill(~probe_mask.bool().unsqueeze(-1), -torch.inf)
        per_head = masked.max(dim=1).values
        fallback = values.mean(dim=1).sum(dim=-1) + self.b_out
        agg = per_head.sum(dim=-1) + self.b_out
        no_token = ~probe_mask.bool().any(dim=1)
        return torch.where(no_token, fallback, agg)

    def rolling_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        width = int(hp.rmattn_window)
        y = self.transformed_acts(acts)
        query, values = self.head_logits(y)
        agg, valid, weights, mask_windows = self.rolling_aggregation(values, query, probe_mask, width)
        return agg, valid, values.sum(dim=-1), weights, mask_windows

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        agg, valid, _, weights, mask_windows = self.rolling_logits(acts, probe_mask, hp)
        bce = bce_with_logits_per_example(agg, labels).mean()
        out = {
            "logits": agg.detach(),
            "scores": torch.sigmoid(agg.detach()),
            "loss_bce": float(bce.detach().cpu()),
            "fallback_fraction": window_fallback_fraction(valid) if valid.numel() else 1.0,
        }
        if weights is not None and mask_windows is not None and valid.any():
            out.update(window_attention_weight_stats(weights, mask_windows, valid))
        return bce, out

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        mode = hp.rmattn_eval_aggregation.lower().replace("-", "_")
        if mode in {"rolling", "roll"}:
            agg, _, _, _, _ = self.rolling_logits(acts, probe_mask, hp)
        elif mode in {"multimax", "multi_max", "mm"}:
            y = self.transformed_acts(acts)
            _, values = self.head_logits(y)
            agg = self.multimax_aggregation(values, probe_mask)
        else:
            raise ValueError(f"Unknown rmattn_eval_aggregation={hp.rmattn_eval_aggregation!r}")
        return agg


class SWiMProbe(LinearTokenProbe):
    probe_name = "swim"

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        return probe_loss_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, labels, hp)

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        return probe_score_logits_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, hp)

    def score(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        z = self.token_logits(acts)
        fallback = masked_mean(z, probe_mask)
        return ema_streaming_scores(
            z,
            probe_mask,
            hp.gamma_ema,
            hp.streaming_reduction,
            fallback_logits=fallback,
        )


class SCTopKProbe(LinearTokenProbe):
    probe_name = "sctopk"

    def forward(self, acts: Tensor, probe_mask: Tensor, labels: Tensor, hp: ProbeHyperparams) -> tuple[Tensor, dict[str, Any]]:
        return probe_loss_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, labels, hp)

    def score_logits(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        return probe_score_logits_from_token_logits(self.probe_name, self.token_logits(acts), probe_mask, hp)

    def score(self, acts: Tensor, probe_mask: Tensor, hp: ProbeHyperparams) -> Tensor:
        z = self.token_logits(acts)
        fallback = masked_mean(z, probe_mask)
        return ema_streaming_scores(
            z,
            probe_mask,
            hp.gamma_ema,
            hp.streaming_reduction,
            fallback_logits=fallback,
        )


def topk_window_logits(zbar: Tensor, valid: Tensor, k: int) -> Tensor:
    batch = zbar.shape[0]
    outputs: list[Tensor] = []
    for row, row_valid in zip(zbar, valid):
        count = int(row_valid.sum().item())
        if count == 0:
            outputs.append(row.new_zeros(()))
            continue
        row_k = min(int(k), count)
        values = row.masked_select(row_valid)
        outputs.append(torch.topk(values.float(), k=row_k).values.mean())
    return torch.stack(outputs, dim=0) if outputs else zbar.new_empty((batch,))


def segment_variance(zbar: Tensor, valid: Tensor, tau_s: float, eps: float = 1e-8) -> Tensor:
    p = torch.sigmoid(zbar.float() / tau_s).masked_fill(~valid.bool(), 0.0)
    denom = p.sum(dim=1).clamp_min(eps)
    mu = (p * zbar.float()).sum(dim=1) / denom
    var = (p * (zbar.float() - mu.unsqueeze(1)).pow(2)).sum(dim=1) / denom
    has_valid = valid.any(dim=1)
    return torch.where(has_valid, var, torch.zeros_like(var))


def ema_streaming_logits(
    logits: Tensor,
    probe_mask: Tensor,
    gamma: float,
    reduction: str = "max",
    fallback_logits: Tensor | None = None,
) -> Tensor:
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    outputs: list[Tensor] = []
    for row_index, (row, row_mask) in enumerate(zip(logits.float(), probe_mask.bool())):
        values = row.masked_select(row_mask)
        if values.numel() == 0:
            if fallback_logits is not None:
                outputs.append(fallback_logits[row_index].float())
                continue
            values = row
        ema = values[0]
        smoothed = [ema]
        for value in values[1:]:
            ema = gamma * value + (1.0 - gamma) * ema
            smoothed.append(ema)
        smooth = torch.stack(smoothed)
        if reduction == "max":
            outputs.append(smooth.max())
        elif reduction == "last":
            outputs.append(smooth[-1])
        elif reduction == "mean":
            outputs.append(smooth.mean())
        else:
            raise ValueError(f"Unknown streaming_reduction={reduction!r}")
    return torch.stack(outputs, dim=0)


def ema_streaming_scores(
    logits: Tensor,
    probe_mask: Tensor,
    gamma: float,
    reduction: str = "max",
    fallback_logits: Tensor | None = None,
) -> Tensor:
    return torch.sigmoid(
        ema_streaming_logits(
            logits,
            probe_mask,
            gamma,
            reduction,
            fallback_logits=fallback_logits,
        )
    )


def probe_loss_from_token_logits(
    probe_name: str,
    token_logits: Tensor,
    probe_mask: Tensor,
    labels: Tensor,
    hp: ProbeHyperparams,
    *,
    attention_logits: Tensor | None = None,
    collect_diagnostics: bool = True,
) -> tuple[Tensor, dict[str, Any]]:
    """Apply a probe's training aggregation to precomputed token logits.

    Some probes construct token logits one layer at a time so a dense
    feature tensor is never materialized.
    Keeping aggregation here makes that path identical to raw-activation probes.
    """

    name = probe_name.lower().replace("-", "").replace("_", "")
    z = token_logits.float()
    if name == "mean":
        agg = masked_mean(z, probe_mask)
        loss = bce_with_logits_per_example(agg, labels).mean()
        return loss, {"logits": agg.detach(), "scores": torch.sigmoid(agg.detach())}

    if name == "softmax":
        weights = masked_softmax(z, probe_mask)
        agg = (weights * z).sum(dim=1)
        loss = bce_with_logits_per_example(agg, labels).mean()
        out = {"logits": agg.detach(), "scores": torch.sigmoid(agg.detach())}
        if collect_diagnostics:
            out.update(attention_weight_stats(weights.detach(), probe_mask))
        return loss, out

    if name == "attention":
        if attention_logits is None:
            raise ValueError("attention probe requires attention_logits")
        weights = masked_softmax(attention_logits, probe_mask)
        agg = (weights * z).sum(dim=1)
        loss = bce_with_logits_per_example(agg, labels).mean()
        out = {"logits": agg.detach(), "scores": torch.sigmoid(agg.detach())}
        if collect_diagnostics:
            out.update(attention_weight_stats(weights.detach(), probe_mask))
        return loss, out

    if name == "swim":
        zbar, valid = sliding_window_mean(z, probe_mask, hp.M)
        fallback_logits = masked_mean(z, probe_mask)
        fallback_loss = bce_with_logits_per_example(fallback_logits, labels)
        window_labels = labels.float().unsqueeze(1).expand_as(zbar)
        per_window_bce = F.binary_cross_entropy_with_logits(zbar, window_labels, reduction="none")
        weights = masked_softmax(zbar / hp.tau_swim, valid)
        window_loss = (weights * per_window_bce).sum(dim=1)
        has_window = valid.any(dim=1)
        loss = torch.where(has_window, window_loss, fallback_loss).mean()
        weighted_logits = (weights * zbar).sum(dim=1)
        agg = torch.where(has_window, weighted_logits, fallback_logits)
        out = {
            "logits": agg.detach(),
            "scores": torch.sigmoid(agg.detach()),
        }
        if collect_diagnostics:
            out["fallback_fraction"] = window_fallback_fraction(valid)
        return loss, out

    if name == "sctopk":
        zbar, valid = sliding_window_mean(z, probe_mask, hp.M)
        seq_logits = topk_window_logits(zbar, valid, hp.K)
        fallback_logits = masked_mean(z, probe_mask)
        seq_logits = torch.where(valid.any(dim=1), seq_logits, fallback_logits)
        bce = bce_with_logits_per_example(seq_logits, labels)
        segvar = segment_variance(zbar, valid, hp.tau_s)
        gated_segvar = (1.0 - labels.float()) * segvar
        if collect_diagnostics and torch.any(labels.float() > 0.5):
            positive_max = gated_segvar[labels.float() > 0.5].detach().abs().max()
            if positive_max.item() != 0.0:
                raise AssertionError("SC-TopK SegVar must be exactly zero for positive examples")
        loss = (bce + hp.lambda_segvar * gated_segvar).mean()
        out = {
            "logits": seq_logits.detach(),
            "scores": torch.sigmoid(seq_logits.detach()),
        }
        if collect_diagnostics:
            out.update(
                {
                    "loss_bce": float(bce.detach().mean().cpu()),
                    "loss_segvar": float(gated_segvar.detach().mean().cpu()),
                    "fallback_fraction": window_fallback_fraction(valid),
                }
            )
        return loss, out

    raise ValueError(f"Token-logit aggregation is not implemented for probe {probe_name!r}")


def probe_score_logits_from_token_logits(
    probe_name: str,
    token_logits: Tensor,
    probe_mask: Tensor,
    hp: ProbeHyperparams,
    *,
    attention_logits: Tensor | None = None,
) -> Tensor:
    """Apply a probe's inference aggregation to precomputed token logits."""

    name = probe_name.lower().replace("-", "").replace("_", "")
    z = token_logits.float()
    if name == "mean":
        return masked_mean(z, probe_mask)
    if name == "softmax":
        weights = masked_softmax(z, probe_mask)
        return (weights * z).sum(dim=1)
    if name == "attention":
        if attention_logits is None:
            raise ValueError("attention probe requires attention_logits")
        weights = masked_softmax(attention_logits, probe_mask)
        return (weights * z).sum(dim=1)
    if name in {"swim", "sctopk"}:
        fallback = masked_mean(z, probe_mask)
        return ema_streaming_logits(
            z,
            probe_mask,
            hp.gamma_ema,
            hp.streaming_reduction,
            fallback_logits=fallback,
        )
    raise ValueError(f"Token-logit scoring is not implemented for probe {probe_name!r}")


def build_probe(name: str, n_layers: int, hidden_size: int, hp: ProbeHyperparams) -> ProbeBase:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized == "mean":
        return MeanProbe(n_layers, hidden_size)
    if normalized == "softmax":
        return SoftmaxProbe(n_layers, hidden_size)
    if normalized == "attention":
        return AttentionProbe(n_layers, hidden_size)
    if normalized == "rmattn":
        return RMAttnProbe(n_layers, hidden_size, hidden=hp.rmattn_hidden, heads=hp.rmattn_heads)
    if normalized == "swim":
        return SWiMProbe(n_layers, hidden_size)
    if normalized in {"sctopk", "sctop"}:
        return SCTopKProbe(n_layers, hidden_size)
    raise ValueError(f"Unknown probe name: {name}")


def build_all_probes(n_layers: int, hidden_size: int, hp: ProbeHyperparams) -> dict[str, ProbeBase]:
    return {name: build_probe(name, n_layers, hidden_size, hp) for name in PROBE_NAMES}
