"""Rank-based metrics for harmful-intent probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:
    roc_auc_score = None

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclass(frozen=True)
class ProbeMetrics:
    auroc: float
    tpr_at_target_fpr: float
    logspace_auroc: float
    tpr_fpr: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return {
            "auroc": self.auroc,
            tpr_metric_key(self.tpr_fpr): self.tpr_at_target_fpr,
            "logspace_auroc": self.logspace_auroc,
        }


def tpr_metric_key(tpr_fpr: float) -> str:
    if np.isclose(tpr_fpr, 0.01):
        return "tpr@1fpr"
    if np.isclose(tpr_fpr, 0.02):
        return "tpr@2fpr"
    raise ValueError(f"tpr_fpr must be 0.01 or 0.02, got {tpr_fpr}")


def _as_numpy(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got shape {array.shape}")
    return array


def auroc_score(y_true: Iterable[int] | np.ndarray, scores: Iterable[float] | np.ndarray) -> float:
    y = _as_numpy(y_true).astype(int)
    s = _as_numpy(scores)
    if np.unique(y).size < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(y, s))
    fpr, tpr, _ = fallback_roc_curve(y, s)
    return float(_trapz(tpr, fpr))


def tpr_at_fpr(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    max_fpr: float = 0.01,
) -> float:
    y = _as_numpy(y_true).astype(int)
    s = _as_numpy(scores)
    if np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = fallback_roc_curve(y, s)
    eligible = tpr[fpr <= max_fpr]
    if eligible.size:
        return float(np.max(eligible))

    first = int(np.searchsorted(fpr, max_fpr, side="left"))
    if first == 0 or first >= len(fpr):
        return 0.0
    x0, x1 = fpr[first - 1], fpr[first]
    y0, y1 = tpr[first - 1], tpr[first]
    if x1 == x0:
        return float(max(y0, y1))
    return float(y0 + (y1 - y0) * ((max_fpr - x0) / (x1 - x0)))


def logspace_auroc(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    min_fpr: float = 1e-3,
    max_fpr: float = 1e-1,
) -> float:
    """Integrate TPR over log10(FPR) in [min_fpr, max_fpr]."""

    if min_fpr <= 0 or max_fpr <= min_fpr:
        raise ValueError("Expected 0 < min_fpr < max_fpr")

    y = _as_numpy(y_true).astype(int)
    s = _as_numpy(scores)
    if np.unique(y).size < 2:
        return float("nan")

    fpr, tpr, _ = fallback_roc_curve(y, s)
    order = np.argsort(fpr, kind="mergesort")
    fpr = fpr[order]
    tpr = tpr[order]

    unique_fpr, inverse = np.unique(fpr, return_inverse=True)
    tpr_at_fpr = np.zeros_like(unique_fpr)
    for idx, value in zip(inverse, tpr):
        tpr_at_fpr[idx] = max(tpr_at_fpr[idx], value)
    if unique_fpr[0] > 0:
        unique_fpr = np.concatenate(([0.0], unique_fpr))
        tpr_at_fpr = np.concatenate(([0.0], tpr_at_fpr))

    internal = unique_fpr[(unique_fpr > min_fpr) & (unique_fpr < max_fpr)]
    window_fpr = np.unique(np.concatenate(([min_fpr], internal, [max_fpr])))
    if window_fpr.size < 2:
        return 0.0

    indices = np.searchsorted(unique_fpr, window_fpr[:-1], side="right") - 1
    indices = np.clip(indices, 0, len(tpr_at_fpr) - 1)
    heights = tpr_at_fpr[indices]
    log_fpr = np.log10(window_fpr)
    area = float(np.sum(heights * np.diff(log_fpr)))
    return float(area / (np.log10(max_fpr) - np.log10(min_fpr)))


def fallback_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    y = y_true[order]
    s = scores[order]
    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    positives = tps[-1]
    negatives = fps[-1]
    if positives == 0 or negatives == 0:
        raise ValueError("ROC curve is undefined with a single class")
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, s[threshold_idxs]]
    return fps / negatives, tps / positives, thresholds


def compute_rank_metrics(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    *,
    tpr_fpr: float = 0.01,
) -> ProbeMetrics:
    metric_fpr = float(tpr_fpr)
    tpr_metric_key(metric_fpr)
    return ProbeMetrics(
        auroc=auroc_score(y_true, scores),
        tpr_at_target_fpr=tpr_at_fpr(y_true, scores, max_fpr=metric_fpr),
        logspace_auroc=logspace_auroc(y_true, scores, min_fpr=1e-3, max_fpr=1e-1),
        tpr_fpr=metric_fpr,
    )


def threshold_for_fpr(negative_scores: Iterable[float] | np.ndarray, target_fpr: float = 0.01) -> float:
    scores = _as_numpy(negative_scores)
    if scores.size == 0:
        return float("nan")
    # Score >= threshold is classified positive. This is the empirical
    # (1-target_fpr) quantile of negative scores.
    return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))


def fpr_at_threshold(y_true: Iterable[int] | np.ndarray, scores: Iterable[float] | np.ndarray, threshold: float) -> float:
    y = _as_numpy(y_true).astype(int)
    s = _as_numpy(scores)
    negatives = y == 0
    if not np.any(negatives):
        return float("nan")
    return float(np.mean(s[negatives] >= threshold))
