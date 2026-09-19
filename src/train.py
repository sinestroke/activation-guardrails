"""Train six activation probes across activation sites."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.data import (
    DataSettings,
    LengthBucketedBatchSampler,
    available_splits,
    collate_examples,
    data_paths,
    inspect_data_files,
    load_dataset_split,
)
from src.extraction import extract_activation_sites, extract_activations, model_input_device, residual_sanity_check
from src.metrics import compute_rank_metrics
from src.models import load_adapter_and_model, package_version
from src.probes import PROBE_NAMES, ProbeHyperparams, build_all_probes
from src.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    config_sha256,
    file_record,
    file_records,
    git_metadata,
    model_metadata,
    runtime_metadata,
    write_json,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "<your-org>/<your-finetuned-gemma3-12b-it>",
    "model_revision": None,
    "tokenizer_name": None,
    "tokenizer_revision": None,
    "require_tokenizer_model_vocab_match": True,
    "hf_token_env": "HF_TOKEN",
    "local_files_only": False,
    "model_loader": "nnsight",
    "nnsight_wrap_hf_for_vlm": True,
    "device_map": "auto",
    "dtype": "bfloat16",
    "trust_remote_code": False,
    "attn_implementation": None,
    "activation_types": ["residual", "mlp", "attention"],
    "normalize_activations": True,
    "activation_norm_eps": 1e-6,
    "include_embedding_layer": False,
    "probe_region": "all",
    "max_seq_len": 8192,
    "M": 16,
    "tau_swim": 1.0,
    "K": 8,
    "tau_s": 2.0,
    "lambda_segvar": 0.01,
    "gamma_ema": 0.1,
    "streaming_reduction": "max",
    "rmattn_window": 10,
    "rmattn_hidden": 100,
    "rmattn_heads": 10,
    "rmattn_eval_aggregation": "rolling",
    "lr": 5.0e-5,
    "weight_decay": 1.0e-3,
    "final_lr": 5.0e-6,
    "lr_scheduler": "cosine",
    "lr_scheduler_t_max": 20,
    "micro_batch_size": 32,
    "grad_accum_steps": 4,
    "length_bucket_batches": False,
    "epochs": 20,
    "early_stopping_metric": "auroc",
    "tpr_fpr": 0.01,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "n_seeds": 5,
    "fused_seed_training": True,
    "eval_splits": ["internal_eval", "benign_nearmiss_eval"],
    "grad_clip": 1.0,
    "fused_extraction": True,
    "seed": 0,
    "log_every": 50,
    "use_wandb": False,
    "wandb_project": "harm-probes",
    "wandb_entity": None,
    "wandb_group": "probe-sweep",
    "wandb_mode": "online",
    "wandb_name_prefix": None,
    "log_attn_weight_stats": True,
    "output_dir": "results/probe_training",
    "checkpoint_dir": "checkpoints/probes",
    "module_tree_depth": 3,
    "artifact_layout": {"enabled": False},
    "allow_existing_artifacts": False,
    "data": {},
}


@dataclass
class EarlyStopState:
    probe: str
    best_metric: float = -float("inf")
    epochs_without_gain: int = 0
    stalled: bool = False
    best_path: str | None = None
    has_checkpoint: bool = False

    def update(self, metric: float, min_delta: float, patience: int) -> bool:
        improved = self.has_checkpoint is False
        if not np.isnan(metric) and metric > self.best_metric + min_delta:
            improved = True
        if improved:
            self.best_metric = float(metric) if not np.isnan(metric) else self.best_metric
            self.epochs_without_gain = 0
            self.stalled = False
            self.has_checkpoint = True
            return True
        self.epochs_without_gain += 1
        if self.epochs_without_gain >= patience:
            self.stalled = True
        return False


@dataclass
class SiteTrainingBundle:
    site: str
    site_seed: int
    run_seed: int
    config: dict[str, Any]
    metrics_path: Path
    probes: dict[str, Any]
    optimizers: dict[str, Any]
    schedulers: dict[str, Any]
    states: dict[str, EarlyStopState]
    wandb_run: Any = None
    global_step: int = 0

    def active(self) -> bool:
        return not all(state.stalled for state in self.states.values())


@dataclass
class SeedTrainingRun:
    run_seed: int
    config: dict[str, Any]
    metrics_path: Path


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_config = loaded.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        return deep_update(load_config(base_path), loaded)
    return deep_update(DEFAULT_CONFIG, loaded)


def resolve_artifact_layout(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    layout = cfg.get("artifact_layout") or {}
    if not isinstance(layout, dict):
        raise ValueError("artifact_layout must be a mapping")
    if not bool(layout.get("enabled", False)):
        return cfg

    required = ("experiment_name", "backbone_family", "backbone_variant")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"artifact_layout requires explicit provenance fields: {missing}")
    run_name = str(layout.get("run_name") or cfg.get("run_name") or "default")
    components = [str(cfg["experiment_name"]), str(cfg["backbone_family"]), str(cfg["backbone_variant"]), run_name]
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(Path(layout.get("output_root", "results/probe_training")).joinpath(*components))
    cfg["checkpoint_dir"] = str(Path(layout.get("checkpoint_root", "checkpoints/probes")).joinpath(*components))
    return cfg


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    for key in (
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "micro_batch_size",
        "grad_accum_steps",
        "length_bucket_batches",
        "epochs",
        "lr",
        "weight_decay",
        "final_lr",
        "lr_scheduler_t_max",
        "n_seeds",
        "fused_seed_training",
        "seed",
        "output_dir",
        "checkpoint_dir",
        "model_loader",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
        "wandb_name_prefix",
        "fused_extraction",
        "allow_existing_artifacts",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.local_files_only:
        cfg["local_files_only"] = True
    if args.activation_types:
        cfg["activation_types"] = args.activation_types
    if args.smoke:
        if not cfg.get("fused_extraction"):
            cfg["activation_types"] = ["residual"]
        cfg["micro_batch_size"] = min(int(cfg["micro_batch_size"]), 8)
        cfg["grad_accum_steps"] = 1
        cfg["epochs"] = 1
        cfg["smoke"] = True
        cfg["layer_limit"] = 2
        cfg["max_train_examples"] = 8
        cfg["max_dev_examples"] = 8
        cfg["max_steps_per_epoch"] = 1
        cfg["n_seeds"] = 1
        cfg["checkpoint_dir"] = str(Path(cfg["checkpoint_dir"]) / "smoke")
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / "smoke")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--model-loader", choices=["nnsight", "hf"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--activation-types", nargs="+")
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--length-bucket-batches", dest="length_bucket_batches", action="store_true")
    parser.add_argument("--no-length-bucket-batches", dest="length_bucket_batches", action="store_false")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--final-lr", type=float)
    parser.add_argument("--lr-scheduler-t-max", type=int)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--fused-seeds", dest="fused_seed_training", action="store_true")
    parser.add_argument("--no-fused-seeds", dest="fused_seed_training", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name-prefix")
    parser.add_argument("--fused-extraction", dest="fused_extraction", action="store_true")
    parser.add_argument("--no-fused-extraction", dest="fused_extraction", action="store_false")
    parser.add_argument("--allow-existing-artifacts", dest="allow_existing_artifacts", action="store_true")
    parser.set_defaults(
        use_wandb=None,
        length_bucket_batches=None,
        fused_extraction=None,
        fused_seed_training=None,
        allow_existing_artifacts=None,
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hp_from_config(config: dict[str, Any]) -> ProbeHyperparams:
    return ProbeHyperparams(
        M=int(config["M"]),
        tau_swim=float(config["tau_swim"]),
        K=int(config["K"]),
        tau_s=float(config["tau_s"]),
        lambda_segvar=float(config["lambda_segvar"]),
        gamma_ema=float(config["gamma_ema"]),
        streaming_reduction=str(config["streaming_reduction"]),
        rmattn_window=int(config["rmattn_window"]),
        rmattn_hidden=int(config["rmattn_hidden"]),
        rmattn_heads=int(config.get("rmattn_heads", 10)),
        rmattn_eval_aggregation=str(config.get("rmattn_eval_aggregation", "rolling")),
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def make_loader(
    dataset,
    batch_size: int,
    pad_token_id: int,
    seed: int,
    shuffle: bool,
    length_bucket_batches: bool = False,
) -> DataLoader:
    if length_bucket_batches:
        sampler = LengthBucketedBatchSampler(
            [example.length for example in dataset.examples],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda examples: collate_examples(examples, pad_token_id=pad_token_id),
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=lambda examples: collate_examples(examples, pad_token_id=pad_token_id),
    )


def save_checkpoint(
    path: Path,
    probe_name: str,
    probe,
    site: str,
    hp: ProbeHyperparams,
    config: dict[str, Any],
    feature_spec: dict[str, Any],
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_activations = bool(config.get("normalize_activations", True))
    activation_preprocessing = {
        "method": "rms_hidden_dim" if normalize_activations else "none",
        "normalize_activations": normalize_activations,
        "activation_norm_eps": float(config.get("activation_norm_eps") or 1e-6) if normalize_activations else None,
    }
    checkpoint_provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "invocation_id": config.get("invocation_id"),
        "run_id": config.get("run_id"),
        "experiment_name": config.get("experiment_name"),
        "backbone_name": config.get("backbone_name"),
        "backbone_family": config.get("backbone_family"),
        "backbone_variant": config.get("backbone_variant"),
        "model_name": config.get("model_name"),
        "model_revision": config.get("model_revision"),
        "resolved_model": config.get("resolved_model"),
        "tokenizer_name": config.get("tokenizer_name") or config.get("model_name"),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "resolved_tokenizer": config.get("resolved_tokenizer"),
        "run_name": config.get("run_name"),
        "seed": config.get("seed"),
        "site": site,
        "probe_name": probe_name,
        "resolved_config_sha256": config_sha256(config),
        "training_manifest": config.get("training_manifest"),
        "activation_preprocessing": activation_preprocessing,
    }
    torch.save(
        {
            "checkpoint_format_version": 2,
            "probe_name": probe_name,
            "state_dict": probe.state_dict(),
            "site": site,
            "probe_hyperparams": asdict(hp),
            "feature_spec": feature_spec,
            "activation_preprocessing": activation_preprocessing,
            "best_metric": metric,
            "config": config,
            "provenance": checkpoint_provenance,
        },
        path,
    )


def effective_layer_count(site: str, dims, config: dict[str, Any]) -> int:
    layer_limit = config.get("layer_limit")
    count = int(layer_limit) if layer_limit is not None else int(dims.n_layers)
    if site == "residual" and bool(config["include_embedding_layer"]) and layer_limit is None:
        count += 1
    return count


def build_optimizers_and_schedulers(
    probes: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    optimizers = {
        name: torch.optim.AdamW(
            probe.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        for name, probe in probes.items()
    }
    scheduler_name = str(config.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        schedulers = {
            name: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config.get("lr_scheduler_t_max", config["epochs"])),
                eta_min=float(config.get("final_lr", 0.0)),
            )
            for name, optimizer in optimizers.items()
        }
    elif scheduler_name in {"none", "off", "false"}:
        schedulers = {}
    else:
        raise ValueError(f"Unknown lr_scheduler={scheduler_name!r}")
    return optimizers, schedulers


def init_wandb_run(site: str, config: dict[str, Any], reinit: str = "finish_previous") -> Any:
    if not config.get("use_wandb"):
        return None
    import wandb

    run_name = site
    if config.get("wandb_name_prefix"):
        run_name = f"{config['wandb_name_prefix']}-{site}"
    if int(config.get("n_seeds", 1)) > 1:
        run_name = f"{run_name}-seed{config['seed']}"
    wandb_kwargs = {
        "project": config["wandb_project"],
        "name": run_name,
        "group": config.get("wandb_group"),
        "mode": config["wandb_mode"],
        "config": config,
        "reinit": reinit,
    }
    if config.get("wandb_entity"):
        wandb_kwargs["entity"] = config["wandb_entity"]
    run = wandb.init(**wandb_kwargs)
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("dev/*", step_metric="global_step")
    return run


@torch.no_grad()
def evaluate_probes(
    probes: dict[str, Any],
    adapter,
    model,
    loader: DataLoader,
    site: str,
    hp: ProbeHyperparams,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    labels: list[float] = []
    scores: dict[str, list[float]] = {name: [] for name in probes}
    layer_limit = config.get("layer_limit")
    for batch in tqdm(loader, desc=f"eval {site}", leave=False):
        acts, probe_mask, y = extract_activations(
            adapter,
            model,
            batch,
            site=site,
            include_embedding_layer=bool(config["include_embedding_layer"]),
            layer_limit=layer_limit,
            use_nnsight=bool(config.get("use_nnsight", True)),
            normalize_activations=bool(config.get("normalize_activations", True)),
            activation_norm_eps=float(config.get("activation_norm_eps", 1e-6)),
        )
        labels.extend(y.detach().cpu().tolist())
        for name, probe in probes.items():
            score = probe.score(acts, probe_mask, hp)
            scores[name].extend(score.detach().cpu().tolist())
        del acts
    return {
        name: compute_rank_metrics(
            labels,
            values,
            tpr_fpr=float(config.get("tpr_fpr", 0.01)),
        ).as_dict()
        for name, values in scores.items()
    }


def train_site(
    site: str,
    adapter,
    model,
    tokenizer,
    dims,
    train_dataset,
    dev_dataset,
    config: dict[str, Any],
    metrics_path: Path,
    site_seed: int,
) -> None:
    hp = hp_from_config(config)
    layer_limit = config.get("layer_limit")
    effective_layers = effective_layer_count(site, dims, config)
    device = model_input_device(model)
    probes = build_all_probes(effective_layers, dims.hidden_size, hp)
    probes = {name: probe.to(device) for name, probe in probes.items()}
    optimizers, schedulers = build_optimizers_and_schedulers(probes, config)
    states = {name: EarlyStopState(name) for name in probes}

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    train_loader = make_loader(
        train_dataset,
        batch_size=int(config["micro_batch_size"]),
        pad_token_id=pad_token_id,
        seed=site_seed,
        shuffle=True,
        length_bucket_batches=bool(config.get("length_bucket_batches", False)),
    )
    dev_loader = make_loader(
        dev_dataset,
        batch_size=int(config["micro_batch_size"]),
        pad_token_id=pad_token_id,
        seed=site_seed,
        shuffle=False,
        length_bucket_batches=bool(config.get("length_bucket_batches", False)),
    )

    wandb_run = init_wandb_run(site, config)

    global_step = 0
    for epoch in range(int(config["epochs"])):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        start_time = time.time()
        step_losses: dict[str, float] = {}
        step_extra: dict[str, Any] = {}
        accum_counter = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"train {site} epoch {epoch}")):
            if config.get("max_steps_per_epoch") is not None and step >= int(config["max_steps_per_epoch"]):
                break
            acts, probe_mask, labels = extract_activations(
                adapter,
                model,
                batch,
                site=site,
                include_embedding_layer=bool(config["include_embedding_layer"]),
                layer_limit=layer_limit,
                use_nnsight=bool(config.get("use_nnsight", True)),
                normalize_activations=bool(config.get("normalize_activations", True)),
                activation_norm_eps=float(config.get("activation_norm_eps", 1e-6)),
            )
            if config.get("smoke"):
                print(f"smoke_activation_shape site={site} shape={tuple(acts.shape)} dtype={acts.dtype}")
            accum_counter += 1
            for name, probe in probes.items():
                if states[name].stalled:
                    continue
                with autocast_context(device):
                    loss, out = probe(acts, probe_mask, labels, hp)
                (loss / int(config["grad_accum_steps"])).backward()
                step_losses[name] = float(loss.detach().cpu())
                step_extra[name] = out
            del acts

            should_step = accum_counter % int(config["grad_accum_steps"]) == 0
            if should_step:
                grad_norms: dict[str, float] = {}
                for name, probe in probes.items():
                    if states[name].stalled:
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), float(config["grad_clip"]))
                    grad_norms[name] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    optimizers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(config["log_every"]) == 0 or config.get("smoke"):
                    payload = {
                        "event": "train_step",
                        "site": site,
                        "seed": int(config["seed"]),
                        "epoch": epoch,
                        "step": global_step,
                        "loss": step_losses,
                        "grad_norm": grad_norms,
                        "learning_rate": {
                            name: float(optimizer.param_groups[0]["lr"])
                            for name, optimizer in optimizers.items()
                        },
                    }
                    for name, out in step_extra.items():
                        if name == "sctopk":
                            payload["sctopk_loss_bce"] = out.get("loss_bce")
                            payload["sctopk_loss_segvar"] = out.get("loss_segvar")
                        if out.get("loss_bce") is not None:
                            payload[f"{name}_loss_bce"] = out.get("loss_bce")
                        if config.get("log_attn_weight_stats") and "attn_peak" in out:
                            payload[f"{name}_attn_peak"] = out["attn_peak"]
                            payload[f"{name}_attn_entropy"] = out["attn_entropy"]
                        if "fallback_fraction" in out:
                            payload[f"{name}_fallback_fraction"] = out["fallback_fraction"]
                    if torch.cuda.is_available():
                        payload["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    payload["sys/sec_per_step"] = (time.time() - start_time) / max(1, step + 1)
                    append_jsonl(metrics_path, payload)
                    if wandb_run is not None:
                        log_to_wandb(wandb_run, payload, global_step)

        if accum_counter % int(config["grad_accum_steps"]) != 0:
            for name, probe in probes.items():
                if states[name].stalled:
                    continue
                torch.nn.utils.clip_grad_norm_(probe.parameters(), float(config["grad_clip"]))
                optimizers[name].step()
                optimizers[name].zero_grad(set_to_none=True)

        for scheduler in schedulers.values():
            scheduler.step()

        dev_metrics = evaluate_probes(probes, adapter, model, dev_loader, site, hp, config)
        epoch_payload = {
            "event": "dev_epoch",
            "site": site,
            "seed": int(config["seed"]),
            "epoch": epoch,
            "step": global_step,
            "metrics": dev_metrics,
        }
        append_jsonl(metrics_path, epoch_payload)
        if wandb_run is not None:
            log_to_wandb(wandb_run, epoch_payload, global_step)

        for name, metrics in dev_metrics.items():
            metric = float(metrics.get(config["early_stopping_metric"], float("nan")))
            improved = states[name].update(
                metric,
                min_delta=float(config["early_stopping_min_delta"]),
                patience=int(config["early_stopping_patience"]),
            )
            if improved:
                checkpoint_path = Path(config["checkpoint_dir"]) / site / f"{name}.pt"
                states[name].best_path = str(checkpoint_path)
                save_checkpoint(
                    checkpoint_path,
                    name,
                    probes[name],
                    site,
                    hp,
                    config,
                    probes[name].feature_spec(site, bool(config["include_embedding_layer"])),
                    metric,
                )
        status = {name: {"best": state.best_metric, "stalled": state.stalled} for name, state in states.items()}
        print(f"early_stop_status site={site} epoch={epoch} {json.dumps(status, sort_keys=True)}")
        if all(state.stalled for state in states.values()):
            print(f"All probes stalled for site={site}; ending site run.")
            break

    for name, state in states.items():
        if state.best_path and Path(state.best_path).exists():
            checkpoint = torch.load(state.best_path, map_location=device)
            probes[name].load_state_dict(checkpoint["state_dict"])
    if wandb_run is not None:
        wandb_run.finish()


def build_site_training_bundle(
    site: str,
    site_seed: int,
    run_seed: int,
    dims,
    hp: ProbeHyperparams,
    config: dict[str, Any],
    metrics_path: Path,
    device: torch.device,
) -> SiteTrainingBundle:
    seed_everything(site_seed)
    probes = build_all_probes(effective_layer_count(site, dims, config), dims.hidden_size, hp)
    probes = {name: probe.to(device) for name, probe in probes.items()}
    optimizers, schedulers = build_optimizers_and_schedulers(probes, config)
    return SiteTrainingBundle(
        site=site,
        site_seed=site_seed,
        run_seed=run_seed,
        config=config,
        metrics_path=metrics_path,
        probes=probes,
        optimizers=optimizers,
        schedulers=schedulers,
        states={name: EarlyStopState(name) for name in probes},
        wandb_run=init_wandb_run(site, config, reinit="create_new"),
    )


@torch.no_grad()
def evaluate_fused_site_bundles(
    bundles: list[SiteTrainingBundle],
    adapter,
    model,
    loader: DataLoader,
    hp: ProbeHyperparams,
    config: dict[str, Any],
) -> dict[tuple[int, str], dict[str, dict[str, float]]]:
    labels: list[float] = []
    scores: dict[tuple[int, str], dict[str, list[float]]] = {
        (bundle.run_seed, bundle.site): {name: [] for name in bundle.probes}
        for bundle in bundles
    }
    sites = list(dict.fromkeys(bundle.site for bundle in bundles))
    desc = f"eval fused[{','.join(sites)}]"
    for batch in tqdm(loader, desc=desc, leave=False):
        activations, probe_mask, y = extract_activation_sites(
            adapter,
            model,
            batch,
            sites=sites,
            include_embedding_layer=bool(config["include_embedding_layer"]),
            layer_limit=config.get("layer_limit"),
            normalize_activations=bool(config.get("normalize_activations", True)),
            activation_norm_eps=float(config.get("activation_norm_eps", 1e-6)),
        )
        labels.extend(y.detach().cpu().tolist())
        for bundle in bundles:
            acts = activations[bundle.site]
            for name, probe in bundle.probes.items():
                values = probe.score(acts, probe_mask, hp)
                scores[(bundle.run_seed, bundle.site)][name].extend(values.detach().cpu().tolist())
        del activations
    return {
        key: {
            name: compute_rank_metrics(
                labels,
                values,
                tpr_fpr=float(config.get("tpr_fpr", 0.01)),
            ).as_dict()
            for name, values in site_scores.items()
        }
        for key, site_scores in scores.items()
    }


def train_bundles_fused(
    bundles: list[SiteTrainingBundle],
    adapter,
    model,
    tokenizer,
    train_dataset,
    dev_dataset,
    config: dict[str, Any],
    loader_seed: int,
) -> None:
    if not bundles:
        raise ValueError("Fused training requires at least one site/seed bundle")
    hp = hp_from_config(config)
    device = model_input_device(model)
    sites = list(dict.fromkeys(bundle.site for bundle in bundles))
    run_seeds = list(dict.fromkeys(bundle.run_seed for bundle in bundles))
    seed_everything(loader_seed)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    train_loader = make_loader(
        train_dataset,
        batch_size=int(config["micro_batch_size"]),
        pad_token_id=pad_token_id,
        seed=loader_seed,
        shuffle=True,
        length_bucket_batches=bool(config.get("length_bucket_batches", False)),
    )
    dev_loader = make_loader(
        dev_dataset,
        batch_size=int(config["micro_batch_size"]),
        pad_token_id=pad_token_id,
        seed=loader_seed,
        shuffle=False,
        length_bucket_batches=bool(config.get("length_bucket_batches", False)),
    )
    print(
        f"fused_extraction=true fused_seed_training={len(run_seeds) > 1} "
        f"sites={sites} seeds={run_seeds} train_batches={len(train_loader)} "
        f"dev_batches={len(dev_loader)} backbone_forwards_per_batch=1",
        flush=True,
    )

    try:
        for epoch in range(int(config["epochs"])):
            active_bundles = [bundle for bundle in bundles if bundle.active()]
            if not active_bundles:
                print("All probes stalled across all sites and seeds; ending fused run.")
                break
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            epoch_start = time.time()
            accum_counter = 0
            step_losses: dict[tuple[int, str], dict[str, float]] = {
                (bundle.run_seed, bundle.site): {} for bundle in active_bundles
            }
            step_extra: dict[tuple[int, str], dict[str, Any]] = {
                (bundle.run_seed, bundle.site): {} for bundle in active_bundles
            }
            active_sites = list(dict.fromkeys(bundle.site for bundle in active_bundles))
            desc = f"train fused sites=[{','.join(active_sites)}] seeds={run_seeds} epoch {epoch}"

            for step, batch in enumerate(tqdm(train_loader, desc=desc)):
                if config.get("max_steps_per_epoch") is not None and step >= int(config["max_steps_per_epoch"]):
                    break
                activations, probe_mask, labels = extract_activation_sites(
                    adapter,
                    model,
                    batch,
                    sites=active_sites,
                    include_embedding_layer=bool(config["include_embedding_layer"]),
                    layer_limit=config.get("layer_limit"),
                    normalize_activations=bool(config.get("normalize_activations", True)),
                    activation_norm_eps=float(config.get("activation_norm_eps", 1e-6)),
                )
                accum_counter += 1
                for bundle in active_bundles:
                    key = (bundle.run_seed, bundle.site)
                    acts = activations[bundle.site]
                    if config.get("smoke"):
                        print(
                            f"smoke_activation_shape seed={bundle.run_seed} site={bundle.site} "
                            f"shape={tuple(acts.shape)} dtype={acts.dtype}"
                        )
                    for name, probe in bundle.probes.items():
                        if bundle.states[name].stalled:
                            continue
                        with autocast_context(device):
                            loss, out = probe(acts, probe_mask, labels, hp)
                        (loss / int(config["grad_accum_steps"])).backward()
                        step_losses[key][name] = float(loss.detach().cpu())
                        step_extra[key][name] = out
                del activations

                if accum_counter % int(config["grad_accum_steps"]) != 0:
                    continue
                for bundle in active_bundles:
                    key = (bundle.run_seed, bundle.site)
                    grad_norms: dict[str, float] = {}
                    for name, probe in bundle.probes.items():
                        if bundle.states[name].stalled:
                            continue
                        grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), float(config["grad_clip"]))
                        grad_norms[name] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                        bundle.optimizers[name].step()
                        bundle.optimizers[name].zero_grad(set_to_none=True)
                    bundle.global_step += 1
                    if bundle.global_step % int(config["log_every"]) != 0 and not config.get("smoke"):
                        continue
                    payload = {
                        "event": "train_step",
                        "site": bundle.site,
                        "seed": bundle.run_seed,
                        "epoch": epoch,
                        "step": bundle.global_step,
                        "loss": step_losses[key],
                        "grad_norm": grad_norms,
                        "learning_rate": {
                            name: float(optimizer.param_groups[0]["lr"])
                            for name, optimizer in bundle.optimizers.items()
                        },
                    }
                    for name, out in step_extra[key].items():
                        if name == "sctopk":
                            payload["sctopk_loss_bce"] = out.get("loss_bce")
                            payload["sctopk_loss_segvar"] = out.get("loss_segvar")
                        if out.get("loss_bce") is not None:
                            payload[f"{name}_loss_bce"] = out.get("loss_bce")
                        if config.get("log_attn_weight_stats") and "attn_peak" in out:
                            payload[f"{name}_attn_peak"] = out["attn_peak"]
                            payload[f"{name}_attn_entropy"] = out["attn_entropy"]
                        if "fallback_fraction" in out:
                            payload[f"{name}_fallback_fraction"] = out["fallback_fraction"]
                    if torch.cuda.is_available():
                        payload["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    payload["sys/sec_per_step"] = (time.time() - epoch_start) / max(1, step + 1)
                    append_jsonl(bundle.metrics_path, payload)
                    if bundle.wandb_run is not None:
                        log_to_wandb(bundle.wandb_run, payload, bundle.global_step)

            if accum_counter % int(config["grad_accum_steps"]) != 0:
                for bundle in active_bundles:
                    for name, probe in bundle.probes.items():
                        if bundle.states[name].stalled:
                            continue
                        torch.nn.utils.clip_grad_norm_(probe.parameters(), float(config["grad_clip"]))
                        bundle.optimizers[name].step()
                        bundle.optimizers[name].zero_grad(set_to_none=True)

            for bundle in active_bundles:
                for scheduler in bundle.schedulers.values():
                    scheduler.step()

            dev_metrics_by_bundle = evaluate_fused_site_bundles(
                active_bundles, adapter, model, dev_loader, hp, config
            )
            for bundle in active_bundles:
                dev_metrics = dev_metrics_by_bundle[(bundle.run_seed, bundle.site)]
                epoch_payload = {
                    "event": "dev_epoch",
                    "site": bundle.site,
                    "seed": bundle.run_seed,
                    "epoch": epoch,
                    "step": bundle.global_step,
                    "metrics": dev_metrics,
                }
                append_jsonl(bundle.metrics_path, epoch_payload)
                if bundle.wandb_run is not None:
                    log_to_wandb(bundle.wandb_run, epoch_payload, bundle.global_step)

                bundle_config = bundle.config
                for name, metrics in dev_metrics.items():
                    metric = float(metrics.get(bundle_config["early_stopping_metric"], float("nan")))
                    improved = bundle.states[name].update(
                        metric,
                        min_delta=float(bundle_config["early_stopping_min_delta"]),
                        patience=int(bundle_config["early_stopping_patience"]),
                    )
                    if improved:
                        checkpoint_path = Path(bundle_config["checkpoint_dir"]) / bundle.site / f"{name}.pt"
                        bundle.states[name].best_path = str(checkpoint_path)
                        save_checkpoint(
                            checkpoint_path,
                            name,
                            bundle.probes[name],
                            bundle.site,
                            hp,
                            bundle_config,
                            bundle.probes[name].feature_spec(
                                bundle.site, bool(bundle_config["include_embedding_layer"])
                            ),
                            metric,
                        )
                status = {
                    name: {"best": state.best_metric, "stalled": state.stalled}
                    for name, state in bundle.states.items()
                }
                print(
                    f"early_stop_status seed={bundle.run_seed} site={bundle.site} "
                    f"epoch={epoch} {json.dumps(status, sort_keys=True)}"
                )

        for bundle in bundles:
            for name, state in bundle.states.items():
                if state.best_path and Path(state.best_path).exists():
                    checkpoint = torch.load(state.best_path, map_location=device)
                    bundle.probes[name].load_state_dict(checkpoint["state_dict"])
    finally:
        for bundle in bundles:
            if bundle.wandb_run is not None:
                bundle.wandb_run.finish()


def train_sites_fused(
    sites: list[str],
    adapter,
    model,
    tokenizer,
    dims,
    train_dataset,
    dev_dataset,
    config: dict[str, Any],
    metrics_path: Path,
    run_seed: int,
) -> None:
    sites = list(dict.fromkeys(str(site) for site in sites))
    if not sites:
        raise ValueError("fused_extraction requires at least one activation type")
    hp = hp_from_config(config)
    device = model_input_device(model)
    configured_site_seeds = config.get("site_seeds", {})
    bundles = [
        build_site_training_bundle(
            site,
            int(configured_site_seeds.get(site, run_seed + offset)),
            run_seed,
            dims,
            hp,
            config,
            metrics_path,
            device,
        )
        for offset, site in enumerate(sites)
    ]
    train_bundles_fused(
        bundles,
        adapter,
        model,
        tokenizer,
        train_dataset,
        dev_dataset,
        config,
        loader_seed=run_seed,
    )


def train_seed_runs_fused(
    seed_runs: list[SeedTrainingRun],
    adapter,
    model,
    tokenizer,
    dims,
    train_dataset,
    dev_dataset,
    config: dict[str, Any],
    loader_seed: int,
) -> None:
    sites = list(dict.fromkeys(str(site) for site in config["activation_types"]))
    if not sites:
        raise ValueError("fused seed training requires at least one activation type")
    hp = hp_from_config(config)
    device = model_input_device(model)
    bundles: list[SiteTrainingBundle] = []
    for seed_run in seed_runs:
        configured_site_seeds = seed_run.config.get("site_seeds", {})
        for offset, site in enumerate(sites):
            bundles.append(
                build_site_training_bundle(
                    site,
                    int(configured_site_seeds.get(site, seed_run.run_seed + offset)),
                    seed_run.run_seed,
                    dims,
                    hp,
                    seed_run.config,
                    seed_run.metrics_path,
                    device,
                )
            )
    train_bundles_fused(
        bundles,
        adapter,
        model,
        tokenizer,
        train_dataset,
        dev_dataset,
        config,
        loader_seed=loader_seed,
    )


def flatten_for_wandb(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    site = payload.get("site", "site")
    if payload["event"] == "train_step":
        for name, value in payload.get("loss", {}).items():
            flat[f"train/{name}/loss"] = value
        for name, value in payload.get("grad_norm", {}).items():
            flat[f"train/{name}/grad_norm"] = value
        for name, value in payload.get("learning_rate", {}).items():
            flat[f"train/{name}/lr"] = value
        for key, value in payload.items():
            if key.startswith("sys/"):
                flat[key] = value
            elif key.endswith("_attn_peak") or key.endswith("_attn_entropy") or key.endswith("_fallback_fraction"):
                flat[f"train/{key}"] = value
            elif key.endswith("_loss_bce"):
                flat[f"train/{key}"] = value
        if payload.get("sctopk_loss_bce") is not None:
            flat["train/sctopk/loss_bce"] = payload["sctopk_loss_bce"]
        if payload.get("sctopk_loss_segvar") is not None:
            flat["train/sctopk/loss_segvar"] = payload["sctopk_loss_segvar"]
    elif payload["event"] == "dev_epoch":
        for probe, metrics in payload["metrics"].items():
            for metric, value in metrics.items():
                flat[f"dev/{probe}/{metric}"] = value
    flat["site"] = site
    if "seed" in payload:
        flat["seed"] = payload["seed"]
    return flat


def log_to_wandb(wandb_run: Any, payload: dict[str, Any], global_step: int) -> None:
    flat = flatten_for_wandb(payload)
    flat["global_step"] = int(global_step)
    wandb_run.log(flat, commit=True)


def yaml_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_resolved_config(config: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(config)
    resolved["versions"] = {
        "torch": str(torch.__version__),
        "transformers": package_version("transformers"),
        "nnsight": package_version("nnsight"),
        "numpy": np.__version__,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(yaml_safe_value(resolved), sort_keys=False),
        encoding="utf-8",
    )


def split_data_records(settings: DataSettings, splits: tuple[str, ...] = ("train", "dev")) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split in splits:
        if split in settings.files:
            records[split] = {"files": file_records(data_paths(settings.files[split]))}
        elif split in settings.exchange_files:
            records[split] = {
                "exchange_files": file_records(data_paths(settings.exchange_files[split])),
                "annotation_files": file_records(data_paths(settings.annotation_files[split])),
            }
    return records


def artifact_root_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def validate_artifact_roots(config: dict[str, Any]) -> None:
    if bool(config.get("allow_existing_artifacts", False)):
        return
    occupied = [
        str(path)
        for path in (Path(config["output_dir"]), Path(config["checkpoint_dir"]))
        if artifact_root_has_files(path)
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to mix a new training invocation with existing artifacts in "
            f"{occupied}. Choose a new artifact_layout.run_name or pass --allow-existing-artifacts."
        )


def training_manifest(
    config: dict[str, Any],
    source_config_path: Path,
    settings: DataSettings,
    dims: Any,
    decoder_path: str,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "probe_training",
        "invocation_id": config["invocation_id"],
        "run_id": config.get("run_id"),
        "experiment": {
            "name": config.get("experiment_name"),
            "run_name": config.get("run_name"),
        },
        "backbone": {
            "name": config.get("backbone_name"),
            "family": config.get("backbone_family"),
            "variant": config.get("backbone_variant"),
            **dict(config.get("resolved_model", {})),
        },
        "tokenizer": {
            "configured_name": config.get("tokenizer_name") or config.get("model_name"),
            "configured_revision": config.get("tokenizer_revision"),
            **dict(config.get("resolved_tokenizer", {})),
        },
        "probe_features": {
            "activation_types": list(config.get("activation_types", [])),
            "fused_extraction": bool(config.get("fused_extraction", False)),
            "fused_seed_training": bool(config.get("fused_seed_training", False)),
            "seed_training_mode": config.get("seed_training_mode"),
            "shared_data_order_seed": config.get("shared_data_order_seed"),
            "normalize_activations": bool(config.get("normalize_activations", True)),
            "activation_norm_eps": config.get("activation_norm_eps"),
            "include_embedding_layer": bool(config.get("include_embedding_layer", False)),
            "probe_region": config.get("probe_region"),
            "max_seq_len": config.get("max_seq_len"),
            "decoder_path": decoder_path,
            "n_layers": int(dims.n_layers),
            "hidden_size": int(dims.hidden_size),
        },
        "seed": seed,
        "source_config": file_record(source_config_path),
        "resolved_config_sha256": config_sha256(config),
        "data": split_data_records(settings),
        "artifacts": {
            "output_dir": str(Path(config["output_dir"]).resolve()),
            "checkpoint_dir": str(Path(config["checkpoint_dir"]).resolve()),
        },
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "runtime": runtime_metadata(),
        "versions": {
            "torch": str(torch.__version__),
            "transformers": package_version("transformers"),
            "nnsight": package_version("nnsight"),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    config = resolve_artifact_layout(load_config(args.config))
    config = apply_cli_overrides(config, args)
    config["source_config_path"] = str(args.config.resolve())
    config["invocation_id"] = str(uuid.uuid4())
    base_seed = int(config["seed"])
    n_seeds = int(config.get("n_seeds", 1))
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seeds_are_fused = bool(config.get("fused_seed_training", False)) and n_seeds > 1
    config["seed_training_mode"] = "fused_shared_order" if seeds_are_fused else "independent_runs"
    config["shared_data_order_seed"] = base_seed if seeds_are_fused else None
    seed_everything(base_seed)
    base_output_dir = Path(config["output_dir"])
    base_checkpoint_dir = Path(config["checkpoint_dir"])
    validate_artifact_roots(config)

    settings = DataSettings.from_config(config)
    configured_eval_splits = list(config.get("eval_splits", []))
    known_splits = set(available_splits(settings))
    inspect_splits = [split for split in ["train", "dev", *configured_eval_splits] if split in known_splits]
    inspect_data_files(settings, splits=inspect_splits)
    adapter, model, tokenizer, dims, decoder_path = load_adapter_and_model(config)
    config["resolved_model"] = model_metadata(
        model,
        requested_model_name=str(config["model_name"]),
        requested_revision=config.get("model_revision"),
    )
    config["run_id"] = ":".join(
        str(value)
        for value in (
            config.get("experiment_name", "probe-training"),
            config.get("backbone_family", config.get("backbone_name", "backbone")),
            config.get("backbone_variant", "unspecified"),
            config.get("run_name", "default"),
            config["invocation_id"],
        )
    )
    root_manifest_path = base_output_dir / "training_manifest.json"
    config["training_manifest"] = str(root_manifest_path.resolve())
    write_resolved_config(config, base_output_dir)
    write_json(root_manifest_path, training_manifest(config, args.config, settings, dims, decoder_path))

    train_dataset = load_dataset_split(
        "train",
        settings,
        tokenizer,
        limit=config.get("max_train_examples"),
    )
    dev_dataset = load_dataset_split(
        "dev",
        settings,
        tokenizer,
        limit=config.get("max_dev_examples"),
    )

    if config.get("smoke"):
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
        smoke_loader = make_loader(
            train_dataset,
            batch_size=min(8, len(train_dataset)),
            pad_token_id=pad_token_id,
            seed=0,
            shuffle=False,
            length_bucket_batches=bool(config.get("length_bucket_batches", False)),
        )
        first_batch = next(iter(smoke_loader))
        residual_sanity_check(adapter, model, first_batch)

    seed_runs: list[SeedTrainingRun] = []
    for seed_index in range(n_seeds):
        run_seed = base_seed + seed_index
        run_config = copy.deepcopy(config)
        run_config["seed"] = run_seed
        run_config["seed_index"] = seed_index
        run_config["site_seeds"] = {
            str(site): run_seed + offset
            for offset, site in enumerate(run_config["activation_types"])
        }
        if n_seeds > 1:
            run_config["output_dir"] = str(base_output_dir / f"seed_{run_seed}")
            run_config["checkpoint_dir"] = str(base_checkpoint_dir / f"seed_{run_seed}")

        output_dir = Path(run_config["output_dir"])
        metrics_path = output_dir / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")
        seed_manifest_path = output_dir / "training_manifest.json"
        run_config["training_manifest"] = str(seed_manifest_path.resolve())
        run_config["run_id"] = f"{config['run_id']}:seed-{run_seed}"
        write_resolved_config(run_config, output_dir)
        write_json(
            seed_manifest_path,
            training_manifest(run_config, args.config, settings, dims, decoder_path, seed=run_seed),
        )
        append_jsonl(
            metrics_path,
            {
                "event": "run_start",
                "run_id": run_config["run_id"],
                "invocation_id": run_config["invocation_id"],
                "seed": run_seed,
                "fused_seed_training": seeds_are_fused,
                "shared_data_order_seed": base_seed if seeds_are_fused else run_seed,
                "training_manifest": str(seed_manifest_path.resolve()),
                "resolved_config_sha256": config_sha256(run_config),
            },
        )
        print(f"seed_run seed={run_seed} seed_index={seed_index} output_dir={output_dir}")
        seed_runs.append(SeedTrainingRun(run_seed, run_config, metrics_path))

    if seeds_are_fused:
        if not config.get("fused_extraction"):
            raise ValueError("fused_seed_training requires fused_extraction=true")
        train_seed_runs_fused(
            seed_runs,
            adapter,
            model,
            tokenizer,
            dims,
            train_dataset,
            dev_dataset,
            config,
            loader_seed=base_seed,
        )
        return

    for seed_run in seed_runs:
        run_config = seed_run.config
        run_seed = seed_run.run_seed
        seed_everything(run_seed)
        if run_config.get("fused_extraction"):
            train_sites_fused(
                list(run_config["activation_types"]),
                adapter,
                model,
                tokenizer,
                dims,
                train_dataset,
                dev_dataset,
                run_config,
                seed_run.metrics_path,
                run_seed,
            )
        else:
            for site in run_config["activation_types"]:
                site_seed = int(run_config["site_seeds"][site])
                train_site(
                    site,
                    adapter,
                    model,
                    tokenizer,
                    dims,
                    train_dataset,
                    dev_dataset,
                    run_config,
                    seed_run.metrics_path,
                    site_seed,
                )


if __name__ == "__main__":
    main()
