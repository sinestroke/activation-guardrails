"""Evaluate probes and LoRA classifier baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import html
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.classifier_data import (
    collate_completion_features,
    completion_features,
    load_classifier_examples,
    read_system_prompt,
)
from src.classifier_train import (
    classification_scores_from_logprobs,
    load_config as load_classifier_config,
    load_tokenizer as load_classifier_tokenizer,
    model_input_device as classifier_model_input_device,
    move_batch_to_device as move_classifier_batch_to_device,
    select_model_class,
    sequence_log_likelihood,
    update_completion_logprobs,
)
from src.data import DataSettings, available_splits, data_paths, inspect_data_files, load_dataset_split
from src.extraction import extract_activation_sites, extract_activations, model_input_device, rms_normalize_activations
from src.metrics import compute_rank_metrics, tpr_metric_key
from src.models import (
    load_adapter_and_model,
    local_files_only_enabled,
    require_hf_token,
    torch_dtype_from_name,
)
from src.probes import PROBE_NAMES, ProbeHyperparams, build_probe, probe_hyperparams_from_mapping
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
from src.train import load_config as load_probe_config
from src.train import make_loader


PROBE_ORDER = ["mean", "softmax", "attention", "rmattn", "swim", "sctopk"]
SITE_ORDER = ["residual", "mlp", "attention"]
METRIC_ORDER = ["auroc", "logspace_auroc", "tpr@1fpr"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "logspace_auroc": "Log-space AUROC",
    "tpr@1fpr": "TPR@1% FPR",
    "tpr@2fpr": "TPR@2% FPR",
}
ACTIVE_TPR_FPR = 0.01


def configure_metric_protocol(config: Mapping[str, Any]) -> None:
    global ACTIVE_TPR_FPR
    ACTIVE_TPR_FPR = float(config.get("tpr_fpr", 0.01))
    selected_tpr_metric = tpr_metric_key(ACTIVE_TPR_FPR)
    METRIC_ORDER[:] = ["auroc", "logspace_auroc", selected_tpr_metric]


@dataclass(frozen=True)
class ActivationPreprocessing:
    normalize_activations: bool
    activation_norm_eps: float | None

    @property
    def method(self) -> str:
        return "rms_hidden_dim" if self.normalize_activations else "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "normalize_activations": self.normalize_activations,
            "activation_norm_eps": self.activation_norm_eps,
        }


def activation_preprocessing_from_config(config: Mapping[str, Any]) -> ActivationPreprocessing:
    normalize = bool(config.get("normalize_activations", True))
    eps = float(config.get("activation_norm_eps") or 1e-6) if normalize else None
    return ActivationPreprocessing(normalize, eps)


def checkpoint_activation_preprocessing(
    checkpoint: Mapping[str, Any],
    path: Path,
    fallback: ActivationPreprocessing,
) -> ActivationPreprocessing:
    embedded = checkpoint.get("activation_preprocessing")
    if isinstance(embedded, Mapping):
        normalize = bool(
            embedded.get(
                "normalize_activations",
                str(embedded.get("method", "rms_hidden_dim")).lower() != "none",
            )
        )
        eps = float(embedded.get("activation_norm_eps") or 1e-6) if normalize else None
        return ActivationPreprocessing(normalize, eps)

    training_config = checkpoint.get("config")
    if isinstance(training_config, Mapping) and "normalize_activations" in training_config:
        return activation_preprocessing_from_config(training_config)

    print(
        f"warning: checkpoint {path} has no embedded activation preprocessing; "
        f"falling back to evaluation config {fallback.as_dict()}",
        flush=True,
    )
    return fallback


SITE_LABELS = {
    "residual": "Residual stream",
    "mlp": "MLP activations",
    "attention": "Attention activations",
}
METHOD_LABELS = {
    "mean": "Mean",
    "softmax": "Softmax",
    "attention": "Attention",
    "rmattn": "RMAttn",
    "swim": "SWiM",
    "sctopk": "SCTopK",
    "llm_lora": "LLM LoRA",
}
METHOD_COLORS = {
    "mean": "#4E79A7",
    "softmax": "#F28E2B",
    "attention": "#E15759",
    "rmattn": "#76B7B2",
    "swim": "#59A14F",
    "sctopk": "#EDC948",
    "llm_lora": "#B07AA1",
}
SITE_COLORS = {
    "residual": "#4E79A7",
    "mlp": "#59A14F",
    "attention": "#E15759",
}
DEFAULT_PLOT_VIEWS = ["per_split_overview", "activation_comparison", "best_activation_overall"]
PLOT_VIEW_ALIASES = {
    "overview": "per_split_overview",
    "per_split": "per_split_overview",
    "per-split-overview": "per_split_overview",
    "activation-comparison": "activation_comparison",
    "best-activation-overall": "best_activation_overall",
    "best_activation": "best_activation_overall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "--probe-config",
        type=Path,
        dest="probe_config",
        required=True,
    )
    parser.add_argument("--classifier-config", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--evaluation-name")
    parser.add_argument("--target-backbone-name")
    parser.add_argument("--probe-checkpoint-backbone-name")
    parser.add_argument("--classifier-model-name")
    parser.add_argument("--classifier-model-revision")
    parser.add_argument("--model-loader", choices=["nnsight", "hf"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--activation-types", nargs="+")
    parser.add_argument("--checkpoint-dir", "--probe-checkpoint-dir", dest="probe_checkpoint_dir")
    parser.add_argument("--classifier-checkpoint-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--classifier-batch-size", type=int)
    parser.add_argument("--probe-seeds", nargs="+", type=int)
    parser.add_argument("--classifier-seeds", nargs="+", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--eval-splits", nargs="+")
    parser.add_argument("--eval-profile")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--prediction-filename")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument(
        "--merge-existing-results",
        action="store_true",
        help=(
            "Preserve rows for skipped scorer families from eval_results_per_dataset_raw.csv "
            "and replace rows for scorer families evaluated in this invocation."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-inspect", action="store_true")
    return parser.parse_args()


def apply_probe_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.evaluation_name is not None:
        cfg["evaluation_name"] = args.evaluation_name
    if args.target_backbone_name is not None:
        cfg["target_backbone_name"] = args.target_backbone_name
    if args.probe_checkpoint_backbone_name is not None:
        cfg["probe_checkpoint_backbone_name"] = args.probe_checkpoint_backbone_name
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.model_revision is not None:
        cfg["model_revision"] = args.model_revision
    if args.model_loader is not None:
        cfg["model_loader"] = args.model_loader
    if args.local_files_only:
        cfg["local_files_only"] = True
    if args.activation_types:
        cfg["activation_types"] = args.activation_types
    if args.probe_checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.probe_checkpoint_dir
    if args.batch_size is not None:
        cfg["micro_batch_size"] = args.batch_size
    if args.max_eval_examples is not None:
        cfg["max_eval_examples"] = args.max_eval_examples
    if args.probe_seeds is not None:
        cfg["evaluation_seeds"] = list(args.probe_seeds)
    return cfg


def apply_classifier_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.classifier_model_name is not None:
        cfg["model_name"] = args.classifier_model_name
        cfg["classifier_model_name"] = args.classifier_model_name
    elif args.model_name is not None:
        cfg["model_name"] = args.model_name
        cfg["classifier_model_name"] = args.model_name
    if args.classifier_model_revision is not None:
        cfg["model_revision"] = args.classifier_model_revision
    if args.local_files_only:
        cfg["local_files_only"] = True
    if args.classifier_checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.classifier_checkpoint_dir
    if args.classifier_batch_size is not None:
        cfg["eval_micro_batch_size"] = args.classifier_batch_size
    elif args.batch_size is not None:
        cfg["eval_micro_batch_size"] = args.batch_size
    if args.max_eval_examples is not None:
        cfg["max_eval_examples"] = args.max_eval_examples
    if args.classifier_seeds is not None:
        cfg["evaluation_seeds"] = list(args.classifier_seeds)
    return cfg


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def evaluation_config(config: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    value = config.get("evaluation", {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("evaluation config must be a mapping")
    cfg = copy.deepcopy(value)
    profiles = cfg.pop("profiles", None)
    default_profile = cfg.pop("default_profile", None)
    cfg.pop("profile", None)
    selected_profile = profile or default_profile
    if selected_profile:
        if not isinstance(profiles, dict) or selected_profile not in profiles:
            available = sorted(profiles) if isinstance(profiles, dict) else []
            raise ValueError(f"Unknown evaluation profile {selected_profile!r}; available profiles: {available}")
        profile_cfg = profiles[selected_profile] or {}
        if not isinstance(profile_cfg, dict):
            raise ValueError(f"evaluation.profiles.{selected_profile} must be a mapping")
        cfg = deep_update(cfg, profile_cfg)
        cfg["profile"] = selected_profile
    if "splits" not in cfg and "eval_splits" not in cfg and config.get("eval_splits") is not None:
        cfg["eval_splits"] = config["eval_splits"]
    return cfg


def eval_dataset_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    value = eval_config.get("dataset", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("evaluation.dataset must be a mapping")
    return value


def metric_aggregate_specs(eval_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = eval_dataset_config(eval_config).get("metric_aggregates", {})
    if raw in (None, False):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("evaluation.dataset.metric_aggregates must be a mapping")
    specs: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"evaluation.dataset.metric_aggregates.{name} must be a mapping")
        splits = [str(split) for split in value.get("splits", [])]
        if not splits:
            raise ValueError(f"evaluation.dataset.metric_aggregates.{name}.splits cannot be empty")
        reduction = str(value.get("reduction", "unweighted_mean"))
        if reduction not in {"unweighted_mean", "example_weighted_mean"}:
            raise ValueError(f"Unknown metric aggregate reduction {reduction!r} for {name!r}")
        specs[str(name)] = {
            "splits": splits,
            "reduction": reduction,
            "require_all_splits": bool(value.get("require_all_splits", True)),
        }
    return specs


def scorer_config(eval_config: dict[str, Any], name: str) -> dict[str, Any]:
    scorers = eval_config.get("scorers", {})
    if isinstance(scorers, list):
        return {"enabled": name in scorers}
    if scorers is None:
        return {}
    if not isinstance(scorers, dict):
        raise ValueError("evaluation.scorers must be a mapping or list")
    value = scorers.get(name, {})
    if isinstance(value, bool):
        return {"enabled": value}
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"evaluation.scorers.{name} must be a mapping or boolean")
    return value


def scorer_enabled(eval_config: dict[str, Any], name: str, default: bool = True) -> bool:
    cfg = scorer_config(eval_config, name)
    return bool(cfg.get("enabled", default))


def apply_evaluation_dataset_overrides(config: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    dataset_cfg = eval_dataset_config(eval_config)
    data_cfg = cfg.setdefault("data", {})
    schema = dataset_cfg.get("schema", dataset_cfg.get("format"))
    if schema is not None:
        data_cfg["format"] = str(schema)
    for key in (
        "files",
        "exchange_files",
        "annotation_files",
        "id_field",
        "annotation_id_field",
        "input_field",
        "label_field",
        "label_mapping",
        "drop_label_values",
        "chat_template",
        "prefer_stored_token_ids",
        "probe_region",
        "max_seq_len",
        "print_rows",
    ):
        if key in dataset_cfg:
            data_cfg[key] = copy.deepcopy(dataset_cfg[key])
    return cfg


def apply_probe_scorer_config(config: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    probes = scorer_config(eval_config, "probes")
    activation_sites = probes.get("activation_sites", probes.get("sites"))
    if activation_sites:
        cfg["activation_types"] = list(activation_sites)
    checkpoint_dir = probes.get("checkpoint_dir")
    if checkpoint_dir is not None:
        cfg["checkpoint_dir"] = checkpoint_dir
    batch_size = probes.get("batch_size")
    if batch_size is not None:
        cfg["micro_batch_size"] = int(batch_size)
    max_examples = probes.get("max_eval_examples")
    if max_examples is not None:
        cfg["max_eval_examples"] = int(max_examples)
    if probes.get("seeds") is not None:
        cfg["evaluation_seeds"] = [int(seed) for seed in probes["seeds"]]
    return cfg


def apply_classifier_scorer_config(config: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    classifier = scorer_config(eval_config, "classifier")
    checkpoint_dir = classifier.get("checkpoint_dir")
    if checkpoint_dir is not None:
        cfg["checkpoint_dir"] = checkpoint_dir
    batch_size = classifier.get("batch_size")
    if batch_size is not None:
        cfg["eval_micro_batch_size"] = int(batch_size)
    max_examples = classifier.get("max_eval_examples")
    if max_examples is not None:
        cfg["max_eval_examples"] = int(max_examples)
    if classifier.get("seeds") is not None:
        cfg["evaluation_seeds"] = [int(seed) for seed in classifier["seeds"]]
    return cfg


def use_shared_evaluation_data(config: dict[str, Any], source_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if "data" in source_config:
        cfg["data"] = copy.deepcopy(source_config["data"])
    return cfg


def split_source_paths(settings: DataSettings, split: str) -> list[Path]:
    if split in settings.files:
        return data_paths(settings.files[split])
    if split in settings.exchange_files:
        return data_paths(settings.exchange_files[split])
    return [Path(split)]


def split_source_path(settings: DataSettings, split: str) -> Path:
    return split_source_paths(settings, split)[0]


def matches_any_pattern(split: str, path: Path, patterns: Iterable[str], file_only: bool = False) -> bool:
    if not patterns:
        return False
    targets = [path.name, str(path)] if file_only else [split, path.name, str(path)]
    return any(fnmatch.fnmatch(target, pattern) for pattern in patterns for target in targets)


def select_eval_splits(
    settings: DataSettings,
    eval_config: dict[str, Any],
    requested: list[str] | None = None,
) -> list[str]:
    dataset_cfg = eval_dataset_config(eval_config)
    available = set(available_splits(settings))
    if requested:
        splits = list(requested)
    elif dataset_cfg.get("splits") is not None:
        splits = list(dataset_cfg["splits"])
    elif dataset_cfg.get("eval_splits") is not None:
        splits = list(dataset_cfg["eval_splits"])
    elif dataset_cfg.get("include_patterns"):
        include_patterns = list(dataset_cfg.get("include_patterns", []))
        exclude_patterns = list(dataset_cfg.get("exclude_patterns", []))
        splits = []
        for split in sorted(available):
            paths = split_source_paths(settings, split)
            if not any(matches_any_pattern(split, path, include_patterns) for path in paths):
                continue
            if any(matches_any_pattern(split, path, exclude_patterns) for path in paths):
                continue
            splits.append(split)
    else:
        splits = list(eval_config.get("splits", [])) or list(eval_config.get("eval_splits", []))
    missing = sorted(set(splits) - available)
    if missing:
        raise ValueError(f"eval_splits are not configured in data files: {missing}")

    require_patterns = list(dataset_cfg.get("require_file_patterns", []))
    disallow_patterns = list(dataset_cfg.get("disallow_file_patterns", []))
    rejected = []
    for split in splits:
        for path in split_source_paths(settings, split):
            if require_patterns and not matches_any_pattern(split, path, require_patterns, file_only=True):
                rejected.append((split, str(path), f"does not match required file patterns {require_patterns}"))
            if disallow_patterns and matches_any_pattern(split, path, disallow_patterns, file_only=True):
                rejected.append((split, str(path), f"matches disallowed file patterns {disallow_patterns}"))
    if rejected:
        raise ValueError(f"Invalid evaluation splits: {rejected}")
    if not splits:
        raise ValueError("No evaluation splits selected; set evaluation.dataset.splits or include_patterns")
    return splits


def normalize_plot_view(value: str) -> str:
    normalized = str(value).strip().replace("-", "_")
    return PLOT_VIEW_ALIASES.get(normalized, normalized)


def plot_views(eval_config: dict[str, Any]) -> list[str]:
    plots = eval_config.get("plots", {})
    if plots is False:
        return []
    if isinstance(plots, list):
        raw_views = plots
    elif isinstance(plots, dict):
        raw_views = plots.get("views", DEFAULT_PLOT_VIEWS)
    elif plots in (None, True):
        raw_views = DEFAULT_PLOT_VIEWS
    else:
        raise ValueError("evaluation.plots must be a mapping, list, boolean, or null")

    views: list[str] = []
    for value in raw_views:
        view = normalize_plot_view(value)
        if view not in DEFAULT_PLOT_VIEWS:
            raise ValueError(f"Unknown evaluation plot view {value!r}")
        if view not in views:
            views.append(view)
    return views


def plots_enabled(eval_config: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.no_plots:
        return False
    plots = eval_config.get("plots", {})
    if plots is False:
        return False
    if isinstance(plots, dict):
        return bool(plots.get("enabled", True))
    return True


def selected_plot_splits(eval_config: dict[str, Any], splits: Iterable[str]) -> set[str]:
    split_list = list(splits)
    plots = eval_config.get("plots", {})
    if not isinstance(plots, dict):
        return set(split_list)
    if plots.get("splits") is not None:
        return {str(split) for split in plots["splits"]}
    include_patterns = list(plots.get("include_split_patterns", []))
    exclude_patterns = list(plots.get("exclude_split_patterns", []))
    selected: set[str] = set()
    for split in split_list:
        if include_patterns and not any(fnmatch.fnmatch(split, pattern) for pattern in include_patterns):
            continue
        if exclude_patterns and any(fnmatch.fnmatch(split, pattern) for pattern in exclude_patterns):
            continue
        selected.add(split)
    return selected


def eval_output_dir(eval_config: dict[str, Any], args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    configured = eval_config.get("output_dir")
    if configured:
        return Path(configured)
    experiment = str(config.get("experiment_name", "eval"))
    return Path("results/eval") / experiment


def prediction_export_settings(
    eval_config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    value = eval_config.get("predictions", {})
    if value is False or value is None:
        config: dict[str, Any] = {}
    elif value is True:
        config = {"enabled": True}
    elif isinstance(value, dict):
        config = value
    else:
        raise ValueError("evaluation.predictions must be a mapping, boolean, or null")
    enabled = bool(config.get("enabled", False) or args.export_predictions)
    filename = str(args.prediction_filename or config.get("filename", "predictions.csv"))
    if Path(filename).name != filename:
        raise ValueError("prediction filename must be a basename inside the evaluation output directory")
    return enabled, filename


def balanced_test_splits(settings: DataSettings, requested: list[str] | None = None) -> list[str]:
    """Backward-compatible split helper; prefer evaluation.dataset in configs."""
    eval_config = {
        "dataset": {
            "include_patterns": ["test_*_balanced", "test_*_balance"],
            "require_file_patterns": ["*_balanced.jsonl", "*_balance.jsonl"],
            "disallow_file_patterns": ["*_raw.jsonl"],
        }
    }
    return select_eval_splits(settings, eval_config, requested=requested)


def seed_from_path(path: Path) -> int:
    if path.name.startswith("seed_"):
        return int(path.name.removeprefix("seed_"))
    for part in path.parts:
        if part.startswith("seed_"):
            return int(part.removeprefix("seed_"))
    return 0


def discover_seed_roots(base_dir: Path, seeds: Iterable[int] | None = None) -> list[tuple[int, Path]]:
    if not base_dir.exists():
        print(f"warning: checkpoint directory does not exist: {base_dir}")
        return []
    selected = {int(seed) for seed in seeds} if seeds is not None else None
    seed_dirs = sorted(path for path in base_dir.glob("seed_*") if path.is_dir())
    if seed_dirs:
        discovered = [(seed_from_path(path), path) for path in seed_dirs]
        return [(seed, path) for seed, path in discovered if selected is None or seed in selected]
    return [(0, base_dir)] if selected is None or 0 in selected else []


def validate_seed_roots(
    base_dir: Path,
    seed_roots: list[tuple[int, Path]],
    requested_seeds: Iterable[int] | None,
    scorer_name: str,
) -> None:
    requested = {int(seed) for seed in requested_seeds or ()}
    discovered = {seed for seed, _path in seed_roots}
    missing = sorted(requested - discovered)
    if seed_roots and not missing:
        return

    nested_runs = sorted(
        {
            seed_dir.parent
            for seed_dir in base_dir.glob("*/seed_*")
            if seed_dir.is_dir()
        }
    )
    hint = ""
    if nested_runs:
        candidates = ", ".join(str(path) for path in nested_runs)
        hint = f" checkpoint_dir may be one level too high; candidate artifact runs: {candidates}."
    detail = f" missing requested seeds={missing}." if missing else ""
    raise FileNotFoundError(
        f"No complete {scorer_name} seed selection under {base_dir}; "
        f"discovered seeds={sorted(discovered)}.{detail}{hint}"
    )


def probe_exchange_logits(probe: Any, acts: torch.Tensor, probe_mask: torch.Tensor, hp: ProbeHyperparams) -> torch.Tensor:
    scorer = getattr(probe, "score_logits", None)
    if callable(scorer):
        return scorer(acts, probe_mask, hp).float()
    scores = probe.score(acts, probe_mask, hp).float()
    return torch.logit(scores.clamp(1e-7, 1.0 - 1e-7))


def append_probe_prediction_rows(
    rows: list[dict[str, Any]],
    batch: Mapping[str, Any],
    labels: torch.Tensor,
    logits: torch.Tensor,
    site: str,
    seed: int,
    method: str,
    target_backbone: str | None,
    training_backbone: str | None,
    preprocessing: ActivationPreprocessing,
) -> None:
    row_keys = batch.get("row_keys")
    exchange_ids = batch.get("exchange_ids")
    splits = batch.get("splits")
    if row_keys is None or exchange_ids is None or splits is None:
        raise ValueError("Prediction export requires row_keys, exchange_ids, and splits in probe batches")
    probabilities = torch.sigmoid(logits.float()).detach().cpu().tolist()
    for row_key, exchange_id, split, label, logit, probability in zip(
        row_keys,
        exchange_ids,
        splits,
        labels.detach().cpu().tolist(),
        logits.detach().cpu().tolist(),
        probabilities,
    ):
        rows.append(
            {
                "family": "probe",
                "method": method,
                "site": site,
                "seed": int(seed),
                "split": str(split),
                "row_key": str(row_key),
                "exchange_id": str(exchange_id),
                "label": int(label),
                "logit": float(logit),
                "probability": float(probability),
                "target_backbone": target_backbone,
                "training_backbone": training_backbone,
                "activation_preprocessing": preprocessing.method,
            }
        )


def validate_probe_checkpoint_provenance(
    checkpoint: dict[str, Any],
    path: Path,
    site: str,
    probe_name: str,
    expected_backbone: str | None,
    require_provenance: bool,
) -> None:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        if require_provenance:
            raise ValueError(f"Checkpoint has no embedded provenance: {path}")
        print(f"warning: checkpoint has no embedded provenance: {path}")
        return
    checks = {
        "site": site,
        "probe_name": probe_name,
    }
    if expected_backbone:
        checks["backbone_name"] = expected_backbone
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in checks.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint provenance mismatch for {path}: {mismatches}")


def load_site_probes(
    site: str,
    checkpoint_root: Path,
    device: torch.device,
    expected_backbone: str | None = None,
    require_provenance: bool = False,
    provenance_records: list[dict[str, Any]] | None = None,
    fallback_preprocessing: ActivationPreprocessing | None = None,
) -> tuple[dict[str, Any], ProbeHyperparams, ActivationPreprocessing]:
    probes: dict[str, Any] = {}
    hp: ProbeHyperparams | None = None
    preprocessing: ActivationPreprocessing | None = None
    fallback = fallback_preprocessing or ActivationPreprocessing(True, 1e-6)
    for name in PROBE_NAMES:
        path = checkpoint_root / site / f"{name}.pt"
        if not path.exists():
            print(f"warning: missing checkpoint {path}")
            continue
        checkpoint = torch.load(path, map_location=device)
        validate_probe_checkpoint_provenance(
            checkpoint,
            path,
            site,
            name,
            expected_backbone=expected_backbone,
            require_provenance=require_provenance,
        )
        checkpoint_preprocessing = checkpoint_activation_preprocessing(checkpoint, path, fallback)
        if preprocessing is not None and checkpoint_preprocessing != preprocessing:
            raise ValueError(
                f"Inconsistent activation preprocessing within {checkpoint_root / site}: "
                f"{preprocessing.as_dict()} vs {checkpoint_preprocessing.as_dict()} from {path}"
            )
        preprocessing = checkpoint_preprocessing
        if provenance_records is not None:
            provenance_records.append(
                {
                    **file_record(path, include_sha256=False),
                    "best_metric": checkpoint.get("best_metric"),
                    "provenance": checkpoint.get("provenance"),
                    "activation_preprocessing": checkpoint_preprocessing.as_dict(),
                }
            )
        probe_hp = probe_hyperparams_from_mapping(checkpoint.get("probe_hyperparams"))
        hp = hp or probe_hp
        feature = checkpoint["feature_spec"]
        probe = build_probe(name, int(feature["n_layers"]), int(feature["hidden_size"]), probe_hp)
        probe.load_state_dict(checkpoint["state_dict"])
        probe.to(device)
        probe.eval()
        probes[name] = probe
    if hp is None:
        hp = ProbeHyperparams()
    return probes, hp, preprocessing or fallback


def load_site_probe_sets(
    site: str,
    seed_roots: list[tuple[int, Path]],
    device: torch.device,
    expected_backbone: str | None = None,
    require_provenance: bool = False,
    provenance_records: list[dict[str, Any]] | None = None,
    fallback_preprocessing: ActivationPreprocessing | None = None,
) -> tuple[list[tuple[int, dict[str, Any], ProbeHyperparams]], ActivationPreprocessing]:
    probe_sets: list[tuple[int, dict[str, Any], ProbeHyperparams]] = []
    preprocessing: ActivationPreprocessing | None = None
    fallback = fallback_preprocessing or ActivationPreprocessing(True, 1e-6)
    for seed, checkpoint_root in seed_roots:
        probes, hp, seed_preprocessing = load_site_probes(
            site,
            checkpoint_root,
            device,
            expected_backbone=expected_backbone,
            require_provenance=require_provenance,
            provenance_records=provenance_records,
            fallback_preprocessing=fallback,
        )
        if probes:
            if preprocessing is not None and seed_preprocessing != preprocessing:
                raise ValueError(
                    f"Cannot ensemble checkpoints with different activation preprocessing for site={site}: "
                    f"{preprocessing.as_dict()} vs seed={seed} {seed_preprocessing.as_dict()}"
                )
            preprocessing = seed_preprocessing
            probe_sets.append((seed, probes, hp))
    return probe_sets, preprocessing or fallback


def validate_probe_sets(
    checkpoint_base: Path,
    site: str,
    seed_roots: list[tuple[int, Path]],
    probe_sets: list[tuple[int, dict[str, Any], ProbeHyperparams]],
) -> None:
    expected_seeds = {seed for seed, _path in seed_roots}
    loaded = {seed: set(probes) for seed, probes, _hp in probe_sets}
    missing_seeds = sorted(expected_seeds - set(loaded))
    missing_probes = {
        seed: sorted(set(PROBE_NAMES) - names)
        for seed, names in loaded.items()
        if set(PROBE_NAMES) - names
    }
    if not missing_seeds and not missing_probes:
        return
    raise FileNotFoundError(
        f"Incomplete probe checkpoints under {checkpoint_base} for site={site}: "
        f"missing_seeds={missing_seeds}, missing_probes={missing_probes}"
    )


def apply_activation_preprocessing(acts: torch.Tensor, preprocessing: ActivationPreprocessing) -> torch.Tensor:
    if not preprocessing.normalize_activations:
        return acts
    return rms_normalize_activations(
        acts,
        eps=float(preprocessing.activation_norm_eps or 1e-6),
    )


@torch.no_grad()
def score_probe_sets_split(
    probe_sets: list[tuple[int, dict[str, Any], ProbeHyperparams]],
    adapter,
    model,
    loader,
    site: str,
    config: dict[str, Any],
    preprocessing: ActivationPreprocessing,
    desc: str | None = None,
    prediction_rows: list[dict[str, Any]] | None = None,
    target_backbone: str | None = None,
    training_backbone: str | None = None,
) -> tuple[np.ndarray, dict[tuple[int, str], np.ndarray]]:
    labels: list[float] = []
    scores: dict[tuple[int, str], list[float]] = {
        (seed, name): []
        for seed, probes, _hp in probe_sets
        for name in probes
    }
    for batch in tqdm(loader, desc=desc, leave=False):
        acts, probe_mask, y = extract_activations(
            adapter,
            model,
            batch,
            site=site,
            include_embedding_layer=bool(config["include_embedding_layer"]),
            layer_limit=config.get("layer_limit"),
            use_nnsight=bool(config.get("use_nnsight", True)),
            normalize_activations=False,
        )
        acts = apply_activation_preprocessing(acts, preprocessing)
        labels.extend(y.detach().cpu().tolist())
        for seed, probes, hp in probe_sets:
            for name, probe in probes.items():
                if prediction_rows is None:
                    values = probe.score(acts, probe_mask, hp)
                else:
                    logits = probe_exchange_logits(probe, acts, probe_mask, hp)
                    values = torch.sigmoid(logits)
                    append_probe_prediction_rows(
                        prediction_rows,
                        batch,
                        y,
                        logits,
                        site,
                        seed,
                        name,
                        target_backbone,
                        training_backbone,
                        preprocessing,
                    )
                scores[(seed, name)].extend(values.detach().cpu().tolist())
        del acts
    return np.asarray(labels, dtype=np.int64), {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}


@torch.no_grad()
def score_probe_sites_split(
    probe_sets_by_site: dict[str, list[tuple[int, dict[str, Any], ProbeHyperparams]]],
    preprocessing_by_site: dict[str, ActivationPreprocessing],
    adapter,
    model,
    loader,
    config: dict[str, Any],
    desc: str | None = None,
    prediction_rows: list[dict[str, Any]] | None = None,
    target_backbone: str | None = None,
    training_backbone: str | None = None,
) -> tuple[np.ndarray, dict[tuple[str, int, str], np.ndarray]]:
    labels: list[float] = []
    scores: dict[tuple[str, int, str], list[float]] = {
        (site, seed, name): []
        for site, probe_sets in probe_sets_by_site.items()
        for seed, probes, _hp in probe_sets
        for name in probes
    }
    sites = list(probe_sets_by_site)
    for batch in tqdm(loader, desc=desc, leave=False):
        activations, probe_mask, y = extract_activation_sites(
            adapter,
            model,
            batch,
            sites=sites,
            include_embedding_layer=bool(config["include_embedding_layer"]),
            layer_limit=config.get("layer_limit"),
            normalize_activations=False,
        )
        labels.extend(y.detach().cpu().tolist())
        for site, probe_sets in probe_sets_by_site.items():
            acts = activations.pop(site)
            acts = apply_activation_preprocessing(acts, preprocessing_by_site[site])
            for seed, probes, hp in probe_sets:
                for name, probe in probes.items():
                    if prediction_rows is None:
                        values = probe.score(acts, probe_mask, hp)
                    else:
                        logits = probe_exchange_logits(probe, acts, probe_mask, hp)
                        values = torch.sigmoid(logits)
                        append_probe_prediction_rows(
                            prediction_rows,
                            batch,
                            y,
                            logits,
                            site,
                            seed,
                            name,
                            target_backbone,
                            training_backbone,
                            preprocessing_by_site[site],
                        )
                    scores[(site, seed, name)].extend(values.detach().cpu().tolist())
            del acts
    return np.asarray(labels, dtype=np.int64), {
        key: np.asarray(values, dtype=np.float64)
        for key, values in scores.items()
    }


def metric_dict(y_true: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    if np.unique(y).size < 2:
        return {metric: float("nan") for metric in METRIC_ORDER}
    return compute_rank_metrics(y, s, tpr_fpr=ACTIVE_TPR_FPR).as_dict()


def evaluate_probes(
    config: dict[str, Any],
    eval_splits: list[str],
    inspect: bool = True,
    prediction_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    checkpoint_base = Path(config["checkpoint_dir"])
    seed_roots = discover_seed_roots(checkpoint_base, config.get("evaluation_seeds"))
    validate_seed_roots(
        checkpoint_base,
        seed_roots,
        config.get("evaluation_seeds"),
        scorer_name="probe",
    )
    settings = DataSettings.from_config(config)
    if inspect:
        inspect_data_files(settings, splits=eval_splits)
    adapter, model, tokenizer, _dims, _path = load_adapter_and_model(config)
    resolved_model = model_metadata(
        model,
        requested_model_name=str(config["model_name"]),
        requested_revision=config.get("model_revision"),
    )
    if config.get("resolved_target_model_adapter") is not None:
        resolved_model["target_model_adapter"] = copy.deepcopy(config["resolved_target_model_adapter"])
    config["evaluation_resolved_model"] = resolved_model
    device = model_input_device(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    loaders = {}
    for index, split in enumerate(eval_splits):
        dataset = load_dataset_split(split, settings, tokenizer, limit=config.get("max_eval_examples"))
        loaders[split] = make_loader(
            dataset,
            int(config["micro_batch_size"]),
            pad_token_id,
            seed=index,
            shuffle=False,
            length_bucket_batches=bool(config.get("length_bucket_batches", False)),
        )

    rows: list[dict[str, Any]] = []
    expected_backbone = config.get("probe_checkpoint_backbone_name", config.get("backbone_name"))
    provenance_cfg = config.get("evaluation_provenance", {})
    require_provenance = bool(provenance_cfg.get("require_checkpoint_provenance", False))
    loaded_checkpoint_provenance: list[dict[str, Any]] = []
    fallback_preprocessing = activation_preprocessing_from_config(config)
    if config.get("fused_extraction"):
        probe_sets_by_site: dict[str, list[tuple[int, dict[str, Any], ProbeHyperparams]]] = {}
        preprocessing_by_site: dict[str, ActivationPreprocessing] = {}
        for site in config["activation_types"]:
            probe_sets, preprocessing = load_site_probe_sets(
                site,
                seed_roots,
                device,
                expected_backbone=str(expected_backbone) if expected_backbone else None,
                require_provenance=require_provenance,
                provenance_records=loaded_checkpoint_provenance,
                fallback_preprocessing=fallback_preprocessing,
            )
            validate_probe_sets(checkpoint_base, site, seed_roots, probe_sets)
            probe_sets_by_site[site] = probe_sets
            preprocessing_by_site[site] = preprocessing
        if probe_sets_by_site:
            site_labels = ",".join(probe_sets_by_site)
            preprocessing_labels = {
                site: preprocessing.as_dict()
                for site, preprocessing in preprocessing_by_site.items()
            }
            for split, loader in loaders.items():
                desc = f"eval probes fused sites=[{site_labels}] split={split}"
                print(
                    f"{desc} batches={len(loader)} backbone_forwards_per_batch=1 "
                    f"preprocessing={json.dumps(preprocessing_labels, sort_keys=True)}",
                    flush=True,
                )
                y_true, split_scores = score_probe_sites_split(
                    probe_sets_by_site,
                    preprocessing_by_site,
                    adapter,
                    model,
                    loader,
                    config,
                    desc=desc,
                    prediction_rows=prediction_rows,
                    target_backbone=config.get("target_backbone_name", config.get("backbone_name")),
                    training_backbone=str(expected_backbone) if expected_backbone else None,
                )
                for (site, seed, name), scores in split_scores.items():
                    metrics = metric_dict(y_true, scores)
                    rows.append(
                        {
                            "family": "probe",
                            "method": name,
                            "site": site,
                            "seed": seed,
                            "split": split,
                            "scope": "dataset",
                            "n_examples": len(y_true),
                            "target_backbone": config.get("target_backbone_name", config.get("backbone_name")),
                            "training_backbone": expected_backbone,
                            "activation_preprocessing": preprocessing_by_site[site].method,
                            **metrics,
                        }
                    )
                print(
                    f"{desc} done examples={len(y_true)} scored_probe_checkpoints={len(split_scores)}",
                    flush=True,
                )
        config["loaded_probe_checkpoints"] = loaded_checkpoint_provenance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows

    for site in config["activation_types"]:
        probe_sets, preprocessing = load_site_probe_sets(
            site,
            seed_roots,
            device,
            expected_backbone=str(expected_backbone) if expected_backbone else None,
            require_provenance=require_provenance,
            provenance_records=loaded_checkpoint_provenance,
            fallback_preprocessing=fallback_preprocessing,
        )
        validate_probe_sets(checkpoint_base, site, seed_roots, probe_sets)
        seed_labels = ",".join(str(seed) for seed, _probes, _hp in probe_sets)
        for split, loader in loaders.items():
            desc = f"eval probes site={site} split={split}"
            print(
                f"{desc} batches={len(loader)} seeds=[{seed_labels}] "
                f"preprocessing={json.dumps(preprocessing.as_dict(), sort_keys=True)}",
                flush=True,
            )
            y_true, split_scores = score_probe_sets_split(
                probe_sets,
                adapter,
                model,
                loader,
                site,
                config,
                preprocessing,
                desc=desc,
                prediction_rows=prediction_rows,
                target_backbone=config.get("target_backbone_name", config.get("backbone_name")),
                training_backbone=str(expected_backbone) if expected_backbone else None,
            )
            for (seed, name), scores in split_scores.items():
                metrics = metric_dict(y_true, scores)
                rows.append(
                    {
                        "family": "probe",
                        "method": name,
                        "site": site,
                        "seed": seed,
                        "split": split,
                        "scope": "dataset",
                        "n_examples": len(y_true),
                        "target_backbone": config.get("target_backbone_name", config.get("backbone_name")),
                        "training_backbone": expected_backbone,
                        "activation_preprocessing": preprocessing.method,
                        **metrics,
                    }
                )
            print(f"{desc} done examples={len(y_true)} scored_probe_checkpoints={len(split_scores)}", flush=True)
        del probe_sets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    config["loaded_probe_checkpoints"] = loaded_checkpoint_provenance
    return rows


def load_classifier_adapter(config: dict[str, Any], checkpoint_dir: Path) -> tuple[torch.nn.Module, Any]:
    from peft import PeftModel
    from transformers import AutoConfig

    hf_token_env = str(config.get("hf_token_env", "HF_TOKEN"))
    local_files_only = local_files_only_enabled(config)
    token = os.environ.get(hf_token_env)
    if not token and not local_files_only:
        token = require_hf_token(hf_token_env)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    model_config = AutoConfig.from_pretrained(
        config["model_name"],
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        revision=config.get("model_revision"),
    )
    model_cls = select_model_class(model_config)
    kwargs: dict[str, Any] = {
        "token": token,
        "torch_dtype": torch_dtype_from_name(config.get("dtype", "bfloat16")),
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = config["attn_implementation"]
    if config.get("model_revision") is not None:
        kwargs["revision"] = config["model_revision"]
    model = model_cls.from_pretrained(config["model_name"], **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer = load_classifier_tokenizer(config)
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=False)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def classifier_split_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: dict[str, Any],
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    classifier_config = config.get("classifier", {})
    positive_target = str(classifier_config.get("positive_target", "YES"))
    negative_target = str(classifier_config.get("negative_target", "NO"))
    chat_template_kwargs = classifier_config.get("chat_template_kwargs", {})
    max_seq_len = int(config.get("max_seq_len", 8192))
    features = []
    for example in examples:
        features.append(
            completion_features(
                example,
                tokenizer,
                positive_target,
                "positive",
                max_seq_len,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
        features.append(
            completion_features(
                example,
                tokenizer,
                negative_target,
                "negative",
                max_seq_len,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    loader = DataLoader(
        features,
        batch_size=int(config.get("eval_micro_batch_size") or config.get("micro_batch_size", 1)),
        shuffle=False,
        collate_fn=lambda rows: collate_completion_features(rows, pad_token_id=pad_token_id),
    )
    device = classifier_model_input_device(model)
    logprobs: dict[str, dict[str, float]] = {}
    for batch in tqdm(loader, desc=desc or "eval classifier", leave=False):
        batch = move_classifier_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        values = sequence_log_likelihood(outputs.logits, batch["input_ids"], batch["target_mask"])
        update_completion_logprobs(logprobs, batch, values)

    labels, scores, missing = classification_scores_from_logprobs(examples, logprobs)
    if missing:
        print(f"warning: missing paired completion scores for {missing} evaluation examples")
    scored_examples = [
        example
        for example in examples
        if "positive" in logprobs.get(example.row_key, {})
        and "negative" in logprobs.get(example.row_key, {})
    ]
    return labels, scores, scored_examples


def evaluate_classifier(
    config: dict[str, Any],
    eval_splits: list[str],
    inspect: bool = True,
    prediction_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    checkpoint_base = Path(config["checkpoint_dir"])
    seed_roots = discover_seed_roots(checkpoint_base, config.get("evaluation_seeds"))
    validate_seed_roots(
        checkpoint_base,
        seed_roots,
        config.get("evaluation_seeds"),
        scorer_name="classifier",
    )
    missing_adapters = [
        str(checkpoint_root / "adapter_config.json")
        for _seed, checkpoint_root in seed_roots
        if not (checkpoint_root / "adapter_config.json").exists()
    ]
    if missing_adapters:
        raise FileNotFoundError(f"Missing classifier adapter configs: {missing_adapters}")

    settings = DataSettings.from_config(config)
    if inspect:
        inspect_data_files(settings, splits=eval_splits)
    classifier_config = config.get("classifier", {})
    classifier_model_name = config.get("classifier_model_name", config.get("model_name"))
    target_backbone_name = config.get("evaluation_target_backbone_name")
    system_prompt = read_system_prompt(classifier_config["system_prompt_path"])
    # Tokenizer is loaded once here to build examples consistently; adapter eval
    # reloads it with each model so saved tokenizers can still be supported later.
    _tokenizer = load_classifier_tokenizer(config)
    examples_by_split = {
        split: load_classifier_examples(
            split,
            settings,
            system_prompt,
            positive_target=str(classifier_config.get("positive_target", "YES")),
            negative_target=str(classifier_config.get("negative_target", "NO")),
            limit=config.get("max_eval_examples"),
        )
        for split in eval_splits
    }

    rows: list[dict[str, Any]] = []
    for seed, checkpoint_root in seed_roots:
        model, tokenizer = load_classifier_adapter(config, checkpoint_root)
        if "evaluation_resolved_model" not in config:
            config["evaluation_resolved_model"] = model_metadata(
                model,
                requested_model_name=str(config["model_name"]),
                requested_revision=config.get("model_revision"),
            )
        try:
            for split, examples in examples_by_split.items():
                desc = f"eval classifier seed={seed} split={split}"
                print(f"{desc} examples={len(examples)}", flush=True)
                y_true, scores, scored_examples = classifier_split_scores(
                    model,
                    tokenizer,
                    examples,
                    config,
                    desc=desc,
                )
                if prediction_rows is not None:
                    for example, label, logit in zip(scored_examples, y_true, scores):
                        prediction_rows.append(
                            {
                                "family": "classifier",
                                "method": "llm_lora",
                                "site": "classifier",
                                "seed": int(seed),
                                "split": split,
                                "row_key": str(example.row_key),
                                "exchange_id": str(example.example_id),
                                "label": int(label),
                                "logit": float(logit),
                                "probability": float(1.0 / (1.0 + math.exp(-float(np.clip(logit, -700, 700))))),
                                "target_backbone": target_backbone_name,
                                "training_backbone": classifier_model_name,
                                "classifier_model": classifier_model_name,
                                "activation_preprocessing": None,
                            }
                        )
                rows.append(
                    {
                        "family": "classifier",
                        "method": "llm_lora",
                        "site": "classifier",
                        "seed": seed,
                        "split": split,
                        "scope": "dataset",
                        "n_examples": len(y_true),
                        "target_backbone": target_backbone_name,
                        "training_backbone": classifier_model_name,
                        "classifier_model": classifier_model_name,
                        **metric_dict(y_true, scores),
                    }
                )
                print(f"{desc} done scored_examples={len(y_true)}", flush=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def aggregate_metric_rows(rows: list[dict[str, Any]], eval_config: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    dataset_rows = [row for row in rows if row.get("scope", "dataset") == "dataset"]
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in dataset_rows:
        key = (str(row["family"]), str(row["method"]), str(row["site"]), int(row["seed"]))
        grouped[key][str(row["split"])] = row

    for aggregate_name, spec in metric_aggregate_specs(eval_config).items():
        member_splits = list(spec["splits"])
        for (family, method, site, seed), by_split in sorted(grouped.items()):
            available = [split for split in member_splits if split in by_split]
            missing = [split for split in member_splits if split not in by_split]
            if missing and spec["require_all_splits"]:
                raise ValueError(
                    f"Metric aggregate {aggregate_name!r} is missing splits {missing} "
                    f"for {family}/{method}/{site}/seed_{seed}"
                )
            if not available:
                continue
            reference = by_split[available[0]]
            aggregate: dict[str, Any] = {
                "family": family,
                "method": method,
                "site": site,
                "seed": seed,
                "split": aggregate_name,
                "scope": "aggregate",
                "aggregate_reduction": spec["reduction"],
                "aggregate_members": json.dumps(available),
                "n_datasets": len(available),
                "n_examples": sum(int(by_split[split].get("n_examples", 0)) for split in available),
                "target_backbone": reference.get("target_backbone"),
                "training_backbone": reference.get("training_backbone"),
                "activation_preprocessing": reference.get("activation_preprocessing"),
            }
            for metric in METRIC_ORDER:
                values = np.asarray([float(by_split[split].get(metric, float("nan"))) for split in available])
                finite = np.isfinite(values)
                if not finite.any():
                    aggregate[metric] = float("nan")
                    continue
                if spec["reduction"] == "example_weighted_mean":
                    weights = np.asarray([float(by_split[split].get("n_examples", 0)) for split in available])
                    valid_weights = weights[finite]
                    aggregate[metric] = (
                        float(np.average(values[finite], weights=valid_weights))
                        if valid_weights.sum() > 0
                        else float(values[finite].mean())
                    )
                else:
                    aggregate[metric] = float(values[finite].mean())
            aggregate_rows.append(aggregate)
    return aggregate_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in METRIC_ORDER:
            value = float(row.get(metric, float("nan")))
            if math.isfinite(value):
                grouped[
                    (
                        row["family"],
                        row["method"],
                        row["site"],
                        row["split"],
                        str(row.get("scope", "dataset")),
                        metric,
                    )
                ].append(value)
    summary = []
    for (family, method, site, split, scope, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "method": method,
                "site": site,
                "split": split,
                "scope": scope,
                "metric": metric,
                "n": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    integer_fields = {"seed", "n_examples", "n_datasets", "n", "label"}
    float_fields = set(METRIC_ORDER) | {"tpr@1fpr", "tpr@2fpr", "mean", "std", "logit", "probability"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif key in integer_fields:
                row[key] = int(value)
            elif key in float_fields:
                row[key] = float(value)
    return rows


def merge_existing_rows(
    existing_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    replaced_families: set[str],
) -> list[dict[str, Any]]:
    preserved = [
        row
        for row in existing_rows
        if str(row.get("family")) not in replaced_families
    ]
    return preserved + current_rows


def checkpoint_inventory(
    root: str | Path,
    patterns: tuple[str, ...],
    seeds: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return [file_record(base, include_sha256=False)]
    selected_roots = [path for _seed, path in discover_seed_roots(base, seeds)]
    paths = sorted(
        {
            path
            for selected_root in selected_roots
            for pattern in patterns
            for path in selected_root.rglob(pattern)
            if path.is_file()
        }
    )
    return file_records(paths, include_sha256=False)


def evaluation_manifest_payload(
    args: argparse.Namespace,
    eval_config: dict[str, Any],
    probe_config: dict[str, Any],
    classifier_config: dict[str, Any],
    eval_splits: list[str],
    dataset_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    prediction_filename: str | None = None,
    prediction_row_count: int = 0,
    merged_source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = DataSettings.from_config(probe_config)
    datasets = {
        split: file_records(split_source_paths(settings, split))
        for split in eval_splits
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "evaluation",
        "evaluation_name": probe_config.get("evaluation_name", probe_config.get("experiment_name")),
        "profile": eval_config.get("profile"),
        "target_backbone": {
            "name": probe_config.get("target_backbone_name", probe_config.get("backbone_name")),
            "family": probe_config.get("backbone_family"),
            "variant": probe_config.get("backbone_variant"),
            **dict(probe_config.get("evaluation_resolved_model", {})),
        },
        "probe_training_backbone": probe_config.get(
            "probe_checkpoint_backbone_name",
            probe_config.get("backbone_name"),
        ),
        "classifier_backbone": {
            "name": classifier_config.get("classifier_model_name", classifier_config.get("model_name")),
            "family": classifier_config.get("classifier_model_family"),
            "size": classifier_config.get("classifier_model_size"),
            "role": classifier_config.get("classifier_role", "guard_classifier"),
            **dict(classifier_config.get("evaluation_resolved_model", {})),
        },
        "source_configs": {
            "probe": file_record(args.probe_config),
            "classifier": file_record(args.classifier_config),
            "effective_probe_config_sha256": config_sha256(probe_config),
            "effective_classifier_config_sha256": config_sha256(classifier_config),
        },
        "datasets": datasets,
        "metric_protocol": {
            "metrics": list(METRIC_ORDER),
            "logspace_auroc_fpr_range": [1e-3, 1e-1],
            "tpr_fpr": ACTIVE_TPR_FPR,
            "aggregates": metric_aggregate_specs(eval_config),
        },
        "checkpoints": {
            "probes": (
                []
                if args.skip_probes or not scorer_enabled(eval_config, "probes")
                else probe_config.get("loaded_probe_checkpoints")
                or checkpoint_inventory(probe_config["checkpoint_dir"], ("*.pt",))
            ),
            "classifier": (
                []
                if args.skip_classifier or not scorer_enabled(eval_config, "classifier")
                else checkpoint_inventory(
                    classifier_config["checkpoint_dir"],
                    ("adapter_config.json", "*.safetensors", "training_summary.json"),
                    classifier_config.get("evaluation_seeds"),
                )
            ),
        },
        "result_counts": {
            "dataset_rows": len(dataset_rows),
            "aggregate_rows": len(aggregate_rows),
            "prediction_rows": int(prediction_row_count),
        },
        "execution": {
            "skip_probes": bool(args.skip_probes),
            "skip_classifier": bool(args.skip_classifier),
            "merge_existing_results": bool(args.merge_existing_results),
            "merged_sources": list(merged_source_records or ()),
        },
        "prediction_export": {
            "enabled": prediction_filename is not None,
            "filename": prediction_filename,
            "probe_score": "pre-sigmoid exchange logit",
            "classifier_score": "log P(positive completion) - log P(negative completion)",
        },
        "selected_seeds": {
            "probes": probe_config.get("evaluation_seeds"),
            "classifier": classifier_config.get("evaluation_seeds"),
        },
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "runtime": runtime_metadata(),
    }


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "normal",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222"{transform}>{html.escape(text)}</text>'
    )


def zoom_ylim(values: list[float], errors: list[float]) -> tuple[float, float]:
    finite = [(v, e) for v, e in zip(values, errors) if math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    lo = max(0.0, min(v - e for v, e in finite))
    hi = min(1.0, max(v + e for v, e in finite))
    span = hi - lo
    pad = max(0.01, span * 0.18)
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    if hi - lo < 0.04:
        mid = (hi + lo) / 2.0
        lo = max(0.0, mid - 0.02)
        hi = min(1.0, mid + 0.02)
    return lo, hi


def svg_bar_chart(
    labels: list[str],
    values: list[float],
    errors: list[float],
    colors: list[str],
    title: str,
    ylabel: str,
    legend: list[tuple[str, str]],
    width: int = 880,
    height: int = 560,
) -> str:
    left, right, top, bottom = 78, 34, 58, 132
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymin, ymax = zoom_ylim(values, errors)

    def y_pos(value: float) -> float:
        return top + (ymax - value) / max(1e-12, ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 26, title, size=18, weight="bold"),
        svg_text(20, top + plot_h / 2, ylabel, size=13, rotate=-90),
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(svg_text(left - 8, y + 4, f"{value:.3f}", size=11, anchor="end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#333" stroke-width="1.2"/>')

    slot = plot_w / max(1, len(labels))
    bar_w = min(56, slot * 0.68)
    for index, (label, value, error, color) in enumerate(zip(labels, values, errors, colors)):
        if not math.isfinite(value):
            continue
        cx = left + slot * (index + 0.5)
        y_value = y_pos(value)
        y_base = y_pos(ymin)
        rect_y = min(y_value, y_base)
        rect_h = abs(y_base - y_value)
        parts.append(
            f'<rect x="{cx-bar_w/2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" '
            f'fill="{color}" stroke="#222" stroke-width="0.7"/>'
        )
        parts.append(svg_text(cx, rect_y - 6, f"{value:.3f}", size=10))
        if math.isfinite(error) and error > 0:
            y_low = y_pos(max(ymin, value - error))
            y_high = y_pos(min(ymax, value + error))
            parts.append(f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" stroke="#111" stroke-width="1.3"/>')
            parts.append(f'<line x1="{cx-7:.1f}" y1="{y_low:.1f}" x2="{cx+7:.1f}" y2="{y_low:.1f}" stroke="#111" stroke-width="1.3"/>')
            parts.append(f'<line x1="{cx-7:.1f}" y1="{y_high:.1f}" x2="{cx+7:.1f}" y2="{y_high:.1f}" stroke="#111" stroke-width="1.3"/>')
        parts.append(svg_text(cx - 4, top + plot_h + 26, label, size=11, anchor="end", rotate=-38))

    legend_x = left
    legend_y = height - 32
    for label, color in legend:
        parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y-10:.1f}" width="11" height="11" fill="{color}" stroke="#222" stroke-width="0.5"/>')
        parts.append(svg_text(legend_x + 16, legend_y, label, size=10, anchor="start"))
        legend_x += 18 + len(label) * 6.2
    parts.append(f'<text x="{width-right}" y="{height-10}" font-size="10" font-family="Arial, sans-serif" text-anchor="end" fill="#666">mean +/- std across seeds; y-axis zoomed</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def summary_lookup(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {(row["method"], row["site"], row["split"], row["metric"]): row for row in summary}


def plot_eval_summary(summary: list[dict[str, Any]], output_dir: Path, views: Iterable[str]) -> None:
    selected_views = set(views)
    table = summary_lookup(summary)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    splits = sorted({row["split"] for row in summary})
    methods = PROBE_ORDER + ["llm_lora"]
    method_labels = [METHOD_LABELS[method] for method in methods]
    method_colors = [METHOD_COLORS[method] for method in methods]
    method_legend = [(METHOD_LABELS[method], METHOD_COLORS[method]) for method in methods]

    for split in splits:
        if "per_split_overview" in selected_views:
            for site in SITE_ORDER:
                for metric in METRIC_ORDER:
                    values, errors = [], []
                    for method in methods:
                        lookup_site = "classifier" if method == "llm_lora" else site
                        item = table.get((method, lookup_site, split, metric))
                        values.append(float(item["mean"]) if item else float("nan"))
                        errors.append(float(item["std"]) if item else 0.0)
                    write_svg(
                        plot_dir / f"overview_{split}_{site}_{metric}.svg",
                        svg_bar_chart(
                            method_labels,
                            values,
                            errors,
                            method_colors,
                            f"{split} - {SITE_LABELS[site]} - {METRIC_LABELS[metric]}",
                            METRIC_LABELS[metric],
                            method_legend,
                        ),
                    )
        if "activation_comparison" in selected_views:
            for probe in PROBE_ORDER:
                for metric in ("logspace_auroc", tpr_metric_key(ACTIVE_TPR_FPR)):
                    values, errors = [], []
                    for site in SITE_ORDER:
                        item = table.get((probe, site, split, metric))
                        values.append(float(item["mean"]) if item else float("nan"))
                        errors.append(float(item["std"]) if item else 0.0)
                    labels = [SITE_LABELS[site] for site in SITE_ORDER]
                    colors = [SITE_COLORS[site] for site in SITE_ORDER]
                    legend = [(SITE_LABELS[site], SITE_COLORS[site]) for site in SITE_ORDER]
                    write_svg(
                        plot_dir / f"activation_comparison_{split}_{probe}_{metric}.svg",
                        svg_bar_chart(
                            labels,
                            values,
                            errors,
                            colors,
                            f"{split} - {METHOD_LABELS[probe]} - {METRIC_LABELS[metric]}",
                            METRIC_LABELS[metric],
                            legend,
                            width=620,
                            height=500,
                        ),
                    )


def site_short_label(site: str) -> str:
    return {
        "residual": "Residual",
        "mlp": "MLP",
        "attention": "Attention",
        "classifier": "Classifier",
    }.get(site, site)


def seed_averages(rows: list[dict[str, Any]], method: str, site: str, metric: str) -> list[float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method or row.get("site") != site:
            continue
        value = float(row.get(metric, float("nan")))
        if math.isfinite(value):
            by_seed[int(row["seed"])].append(value)
    return [mean(values) for _seed, values in sorted(by_seed.items()) if values]


def best_activation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        for probe in PROBE_ORDER:
            candidates = []
            for site in SITE_ORDER:
                values = seed_averages(rows, probe, site, metric)
                if not values:
                    continue
                candidates.append(
                    {
                        "family": "probe",
                        "method": probe,
                        "site": site,
                        "metric": metric,
                        "n": len(values),
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                    }
                )
            if candidates:
                best_rows.append(max(candidates, key=lambda row: float(row["mean"])))

        classifier_values = seed_averages(rows, "llm_lora", "classifier", metric)
        if classifier_values:
            best_rows.append(
                {
                    "family": "classifier",
                    "method": "llm_lora",
                    "site": "classifier",
                    "metric": metric,
                    "n": len(classifier_values),
                    "mean": mean(classifier_values),
                    "std": stdev(classifier_values) if len(classifier_values) > 1 else 0.0,
                }
            )
    return best_rows


def plot_best_activation_overall(rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    best_rows = best_activation_summary(rows)
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        by_metric[str(row["metric"])].append(row)

    for metric in METRIC_ORDER:
        metric_rows = by_metric.get(metric, [])
        ordered = []
        for probe in PROBE_ORDER:
            ordered.extend(row for row in metric_rows if row["method"] == probe)
        ordered.extend(row for row in metric_rows if row["method"] == "llm_lora")
        if not ordered:
            continue

        labels = []
        values = []
        errors = []
        colors = []
        legend = []
        for row in ordered:
            method = str(row["method"])
            site = str(row["site"])
            label = METHOD_LABELS[method] if site == "classifier" else f"{METHOD_LABELS[method]}-{site_short_label(site)}"
            labels.append(label)
            values.append(float(row["mean"]))
            errors.append(float(row["std"]))
            color = METHOD_COLORS[method]
            colors.append(color)
            legend.append((label, color))

        write_svg(
            plot_dir / f"best_activation_overall_{metric}.svg",
            svg_bar_chart(
                labels,
                values,
                errors,
                colors,
                f"Best activation per probe - {METRIC_LABELS[metric]}",
                METRIC_LABELS[metric],
                legend,
                width=960,
                height=580,
            ),
        )
    return best_rows


def main() -> None:
    args = parse_args()
    loaded_probe_config = load_probe_config(args.probe_config)
    configure_metric_protocol(loaded_probe_config)
    eval_config = evaluation_config(loaded_probe_config, profile=args.eval_profile)
    probe_config = apply_evaluation_dataset_overrides(loaded_probe_config, eval_config)
    probe_config = apply_probe_scorer_config(probe_config, eval_config)
    probe_config = apply_probe_overrides(probe_config, args)
    probe_config["evaluation_provenance"] = copy.deepcopy(eval_config.get("provenance", {}))

    loaded_classifier_config = load_classifier_config(args.classifier_config)
    classifier_config = apply_evaluation_dataset_overrides(loaded_classifier_config, eval_config)
    classifier_config = use_shared_evaluation_data(classifier_config, probe_config)
    classifier_config = apply_classifier_scorer_config(classifier_config, eval_config)
    classifier_config = apply_classifier_overrides(classifier_config, args)
    classifier_config["evaluation_target_backbone_name"] = probe_config.get(
        "target_backbone_name",
        probe_config.get("backbone_name"),
    )

    split_settings = DataSettings.from_config(probe_config)
    eval_splits = select_eval_splits(split_settings, eval_config, requested=args.eval_splits)
    print(f"eval_splits={json.dumps(eval_splits)}")

    output_dir = eval_output_dir(eval_config, args, probe_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_predictions, prediction_filename = prediction_export_settings(eval_config, args)
    dataset_raw_path = output_dir / "eval_results_per_dataset_raw.csv"
    prediction_path = output_dir / prediction_filename

    current_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] | None = [] if export_predictions else None
    replaced_families: set[str] = set()
    merged_source_records: list[dict[str, Any]] = []
    run_probes = scorer_enabled(eval_config, "probes") and not args.skip_probes
    run_classifier = scorer_enabled(eval_config, "classifier") and not args.skip_classifier

    if run_probes:
        probe_rows = evaluate_probes(
            probe_config,
            eval_splits,
            inspect=not args.no_inspect,
            prediction_rows=prediction_rows,
        )
        if not probe_rows:
            raise RuntimeError("Probe evaluation was enabled but produced zero metric rows")
        current_rows.extend(probe_rows)
        replaced_families.add("probe")
    if run_classifier:
        classifier_rows = evaluate_classifier(
            classifier_config,
            eval_splits,
            inspect=not args.no_inspect,
            prediction_rows=prediction_rows,
        )
        if not classifier_rows:
            raise RuntimeError("Classifier evaluation was enabled but produced zero metric rows")
        current_rows.extend(classifier_rows)
        replaced_families.add("classifier")

    if args.merge_existing_results:
        if not dataset_raw_path.exists():
            raise FileNotFoundError(
                f"--merge-existing-results requires existing dataset metrics at {dataset_raw_path}"
            )
        merged_source_records.append(file_record(dataset_raw_path))
        existing_rows = read_csv(dataset_raw_path)
        preserved_families = {
            str(row.get("family"))
            for row in existing_rows
            if str(row.get("family")) not in replaced_families
        }
        dataset_rows = merge_existing_rows(existing_rows, current_rows, replaced_families)
        if prediction_rows is not None:
            if prediction_path.exists():
                merged_source_records.append(file_record(prediction_path))
                prediction_rows = merge_existing_rows(
                    read_csv(prediction_path),
                    prediction_rows,
                    replaced_families,
                )
            elif preserved_families:
                raise FileNotFoundError(
                    f"Cannot preserve prediction families {sorted(preserved_families)} because "
                    f"{prediction_path} does not exist"
                )
        print(
            f"merged existing results preserved_families={sorted(preserved_families)} "
            f"replaced_families={sorted(replaced_families)}",
            flush=True,
        )
    else:
        if not current_rows:
            raise RuntimeError("No scorers were run and no existing results were requested")
        dataset_rows = current_rows

    aggregate_rows = aggregate_metric_rows(dataset_rows, eval_config)
    rows = dataset_rows + aggregate_rows
    raw_path = output_dir / "eval_results_raw.csv"
    summary_path = output_dir / "eval_results_summary.csv"
    dataset_summary_path = output_dir / "eval_results_per_dataset_summary.csv"
    aggregate_raw_path = output_dir / "eval_results_aggregate_raw.csv"
    aggregate_summary_path = output_dir / "eval_results_aggregate_summary.csv"
    best_path = output_dir / "eval_best_activation_summary.csv"
    manifest_path = output_dir / "evaluation_manifest.json"
    summary = summarize(rows)
    write_csv(raw_path, rows)
    write_csv(summary_path, summary)
    write_csv(dataset_raw_path, dataset_rows)
    write_csv(dataset_summary_path, summarize(dataset_rows))
    write_csv(aggregate_raw_path, aggregate_rows)
    write_csv(aggregate_summary_path, summarize(aggregate_rows))
    if prediction_rows is not None:
        write_csv(prediction_path, prediction_rows)
    plot_split_names = selected_plot_splits(eval_config, eval_splits)
    best_source_rows = [row for row in rows if row.get("split") in plot_split_names]
    best_rows = best_activation_summary(best_source_rows)
    write_csv(best_path, best_rows)
    write_json(
        manifest_path,
        evaluation_manifest_payload(
            args,
            eval_config,
            probe_config,
            classifier_config,
            eval_splits,
            dataset_rows,
            aggregate_rows,
            prediction_filename=prediction_filename if prediction_rows is not None else None,
            prediction_row_count=len(prediction_rows or ()),
            merged_source_records=merged_source_records,
        ),
    )
    if plots_enabled(eval_config, args):
        views = plot_views(eval_config)
        plot_summary_rows = [row for row in summary if row.get("split") in plot_split_names]
        plot_raw_rows = [row for row in rows if row.get("split") in plot_split_names]
        plot_eval_summary(plot_summary_rows, output_dir, views)
        if "best_activation_overall" in set(views):
            plot_best_activation_overall(plot_raw_rows, output_dir)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {dataset_summary_path}")
    print(f"wrote {aggregate_summary_path}")
    print(f"wrote {best_path}")
    print(f"wrote {manifest_path}")
    if prediction_rows is not None:
        print(f"wrote {prediction_path} prediction_rows={len(prediction_rows)}")
    print(f"raw_rows={len(rows)} summary_rows={len(summary)} best_activation_rows={len(best_rows)}")


if __name__ == "__main__":
    main()
