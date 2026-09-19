"""LoRA finetuning for generative YES/NO classifier baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.classifier_data import (
    TokenizedClassifierDataset,
    collate_classifier_features,
    collate_completion_features,
    completion_features,
    load_classifier_examples,
    read_system_prompt,
)
from src.data import DataSettings, available_splits, inspect_data_files
from src.metrics import compute_rank_metrics
from src.models import local_files_only_enabled, package_version, require_hf_token, torch_dtype_from_name


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "classifier-baseline",
    "backbone_name": "gemma-3-12b-it",
    "model_name": "<your-org>/<your-model>",
    "model_revision": None,
    "hf_token_env": "HF_TOKEN",
    "local_files_only": False,
    "device_map": "auto",
    "dtype": "bfloat16",
    "trust_remote_code": False,
    "attn_implementation": None,
    "max_seq_len": 8192,
    "micro_batch_size": 2,
    "eval_micro_batch_size": 2,
    "grad_accum_steps": 32,
    "epochs": 1,
    "lr": 2.0e-4,
    "final_lr": 0.0,
    "weight_decay": 0.0,
    "lr_scheduler": "cosine",
    "grad_clip": 1.0,
    "gradient_checkpointing": True,
    "seed": 0,
    "n_seeds": 5,
    "log_every": 10,
    "use_wandb": False,
    "wandb_project": "harm-classifiers",
    "wandb_entity": None,
    "wandb_group": "classifier-sweep",
    "wandb_mode": "online",
    "wandb_name_prefix": None,
    "output_dir": "results/classifier_training",
    "checkpoint_dir": "checkpoints/classifiers",
    "classifier": {
        "system_prompt_path": "constitutions/judges/generic_high_stakes/v1.md",
        "positive_target": "YES",
        "negative_target": "NO",
        "chat_template_kwargs": {},
    },
    "lora": {
        "r": 32,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": "all_linear",
        "bias": "none",
        "task_type": None,
    },
    "data": {},
}


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


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    for key in (
        "model_name",
        "model_revision",
        "micro_batch_size",
        "eval_micro_batch_size",
        "grad_accum_steps",
        "epochs",
        "lr",
        "final_lr",
        "weight_decay",
        "n_seeds",
        "seed",
        "output_dir",
        "checkpoint_dir",
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
        "wandb_name_prefix",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.model_name is not None:
        cfg["classifier_model_name"] = args.model_name
    if args.local_files_only:
        cfg["local_files_only"] = True
    if args.gradient_checkpointing is not None:
        cfg["gradient_checkpointing"] = args.gradient_checkpointing
    if args.max_train_examples is not None:
        cfg["max_train_examples"] = args.max_train_examples
    if args.max_dev_examples is not None:
        cfg["max_dev_examples"] = args.max_dev_examples
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--eval-micro-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--final-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-dev-examples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name-prefix")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(use_wandb=None, gradient_checkpointing=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        "peft": package_version("peft"),
        "numpy": np.__version__,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(yaml_safe_value(resolved), sort_keys=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_input_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key, value in moved.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
    return moved


def select_model_class(model_config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    architectures = [str(item) for item in getattr(model_config, "architectures", []) or []]
    if any("Gemma3ForConditionalGeneration" in arch for arch in architectures):
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            return AutoModelForCausalLM
    return AutoModelForCausalLM


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    hf_token_env = str(config.get("hf_token_env", "HF_TOKEN"))
    local_files_only = local_files_only_enabled(config)
    token = os.environ.get(hf_token_env)
    if not token and not local_files_only:
        token = require_hf_token(hf_token_env)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        token=token,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=local_files_only,
        revision=config.get("model_revision"),
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(config: dict[str, Any]) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig

    hf_token_env = str(config.get("hf_token_env", "HF_TOKEN"))
    local_files_only = local_files_only_enabled(config)
    token = os.environ.get(hf_token_env)
    if not token and not local_files_only:
        token = require_hf_token(hf_token_env)
    dtype = torch_dtype_from_name(config.get("dtype", "bfloat16"))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    model_config = AutoConfig.from_pretrained(
        config["model_name"],
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        revision=config.get("model_revision"),
    )
    model_cls = select_model_class(model_config)
    model_kwargs: dict[str, Any] = {
        "token": token,
        "torch_dtype": dtype,
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    if config.get("model_revision") is not None:
        model_kwargs["revision"] = config["model_revision"]
    model = model_cls.from_pretrained(config["model_name"], **model_kwargs)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    lora = config.get("lora", {})
    target_modules = lora.get("target_modules", "all_linear")
    if target_modules == "all_linear":
        target_modules = "all-linear"
    lora_kwargs: dict[str, Any] = {
        "r": int(lora.get("r", 32)),
        "lora_alpha": int(lora.get("alpha", 16)),
        "lora_dropout": float(lora.get("dropout", 0.1)),
        "target_modules": target_modules,
        "bias": str(lora.get("bias", "none")),
    }
    task_type = lora.get("task_type")
    if task_type:
        from peft import TaskType

        lora_kwargs["task_type"] = getattr(TaskType, str(task_type))
    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        raise ValueError("Need at least two tokens to compute causal LM loss")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def sequence_log_likelihood(logits: torch.Tensor, input_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return torch.zeros((logits.shape[0],), dtype=torch.float32, device=logits.device)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = target_mask[:, 1:].float()
    token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_log_probs * shift_mask).sum(dim=-1)


def update_completion_logprobs(
    logprobs: dict[str, dict[str, float]],
    batch: dict[str, Any],
    values: torch.Tensor,
) -> None:
    for row_key, target_name, value in zip(batch["row_keys"], batch["target_names"], values.detach().cpu()):
        logprobs.setdefault(str(row_key), {})[str(target_name)] = float(value)


def classification_scores_from_logprobs(
    examples: list[Any],
    logprobs: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    labels: list[int] = []
    scores: list[float] = []
    missing = 0
    for example in examples:
        pair = logprobs.get(example.row_key, {})
        if "positive" not in pair or "negative" not in pair:
            missing += 1
            continue
        labels.append(int(example.label))
        scores.append(pair["positive"] - pair["negative"])
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), missing


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    tokenizer: Any,
    dev_examples: list[Any],
    config: dict[str, Any],
) -> dict[str, float]:
    classifier_config = config.get("classifier", {})
    positive_target = str(classifier_config.get("positive_target", "YES"))
    negative_target = str(classifier_config.get("negative_target", "NO"))
    chat_template_kwargs = classifier_config.get("chat_template_kwargs", {})
    max_seq_len = int(config.get("max_seq_len", 8192))
    features = []
    for example in dev_examples:
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
        batch_size=int(config.get("eval_micro_batch_size") or config["micro_batch_size"]),
        shuffle=False,
        collate_fn=lambda rows: collate_completion_features(rows, pad_token_id=pad_token_id),
    )
    device = model_input_device(model)
    logprobs: dict[str, dict[str, float]] = {}
    model.eval()
    for batch in tqdm(loader, desc="eval classifier", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        values = sequence_log_likelihood(outputs.logits, batch["input_ids"], batch["target_mask"])
        update_completion_logprobs(logprobs, batch, values)

    labels, scores, missing = classification_scores_from_logprobs(dev_examples, logprobs)
    metrics = compute_rank_metrics(
        labels,
        scores,
        tpr_fpr=float(config.get("tpr_fpr", 0.01)),
    ).as_dict()
    metrics["score_mean"] = float(np.mean(scores)) if scores.size else float("nan")
    metrics["score_std"] = float(np.std(scores)) if scores.size else float("nan")
    metrics["missing_scores"] = float(missing)
    return metrics


def flatten_for_wandb(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if payload["event"] == "train_step":
        flat["train/loss"] = payload["loss"]
        flat["train/lr"] = payload["lr"]
        flat["train/epoch"] = payload["epoch"]
    elif payload["event"] == "dev_epoch":
        for key, value in payload["metrics"].items():
            flat[f"dev/{key}"] = value
        flat["epoch"] = payload["epoch"]
    if "seed" in payload:
        flat["seed"] = payload["seed"]
    return flat


def log_to_wandb(wandb_run: Any, payload: dict[str, Any], global_step: int) -> None:
    flat = flatten_for_wandb(payload)
    flat["global_step"] = int(global_step)
    wandb_run.log(flat, commit=True)


def make_train_loader(dataset: TokenizedClassifierDataset, tokenizer: Any, config: dict[str, Any], seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    return DataLoader(
        dataset,
        batch_size=int(config["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate_classifier_features(rows, pad_token_id=pad_token_id),
    )


def train_one_seed(
    run_config: dict[str, Any],
    tokenizer: Any,
    train_examples: list[Any],
    dev_examples: list[Any],
    output_dir: Path,
    checkpoint_dir: Path,
) -> None:
    seed = int(run_config["seed"])
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_config, output_dir)
    metrics_path = output_dir / "metrics.jsonl"

    train_dataset = TokenizedClassifierDataset(
        train_examples,
        tokenizer,
        max_seq_len=int(run_config["max_seq_len"]),
        chat_template_kwargs=run_config.get("classifier", {}).get("chat_template_kwargs", {}),
    )
    print(f"classifier_tokenized split=train total={len(train_dataset)} truncations={train_dataset.truncations}")
    train_loader = make_train_loader(train_dataset, tokenizer, run_config, seed=seed)
    model = load_lora_model(run_config)
    device = model_input_device(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(run_config["lr"]),
        weight_decay=float(run_config.get("weight_decay", 0.0)),
    )
    steps_per_epoch = math.ceil(len(train_loader) / int(run_config["grad_accum_steps"]))
    total_steps = max(1, steps_per_epoch * int(run_config["epochs"]))
    scheduler_name = str(run_config.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(run_config.get("final_lr", 0.0)),
        )
    elif scheduler_name in {"none", "off", "false"}:
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler={scheduler_name!r}")

    wandb_run = None
    if run_config.get("use_wandb"):
        import wandb

        run_name = run_config.get("wandb_name_prefix") or run_config.get("experiment_name", "classifier")
        if int(run_config.get("n_seeds", 1)) > 1:
            run_name = f"{run_name}-seed{seed}"
        wandb_kwargs = {
            "project": run_config["wandb_project"],
            "name": run_name,
            "group": run_config.get("wandb_group"),
            "mode": run_config["wandb_mode"],
            "config": run_config,
            "reinit": "finish_previous",
        }
        if run_config.get("wandb_entity"):
            wandb_kwargs["entity"] = run_config["wandb_entity"]
        wandb_run = wandb.init(**wandb_kwargs)
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("train/*", step_metric="global_step")
        wandb_run.define_metric("dev/*", step_metric="global_step")

    global_step = 0
    best_metric = -float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(run_config["epochs"])):
        model.train()
        start_time = time.time()
        accum_counter = 0
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"train classifier epoch {epoch}")):
            batch = move_batch_to_device(batch, device)
            with autocast_context(device):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss = causal_lm_loss(outputs.logits, batch["labels"])
            (loss / int(run_config["grad_accum_steps"])).backward()
            running_loss += float(loss.detach().cpu())
            accum_counter += 1
            should_step = accum_counter % int(run_config["grad_accum_steps"]) == 0 or step == len(train_loader) - 1
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(run_config["grad_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if global_step % int(run_config["log_every"]) == 0:
                    payload = {
                        "event": "train_step",
                        "seed": seed,
                        "epoch": epoch,
                        "step": global_step,
                        "loss": running_loss / max(1, accum_counter),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "sec_per_step": (time.time() - start_time) / max(1, step + 1),
                    }
                    append_jsonl(metrics_path, payload)
                    if wandb_run is not None:
                        log_to_wandb(wandb_run, payload, global_step)

        dev_metrics = evaluate_classifier(model, tokenizer, dev_examples, run_config)
        epoch_payload = {
            "event": "dev_epoch",
            "seed": seed,
            "epoch": epoch,
            "step": global_step,
            "metrics": dev_metrics,
        }
        append_jsonl(metrics_path, epoch_payload)
        if wandb_run is not None:
            log_to_wandb(wandb_run, epoch_payload, global_step)
        metric = float(dev_metrics.get("logspace_auroc", float("nan")))
        if not math.isnan(metric) and metric >= best_metric:
            best_metric = metric
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            (checkpoint_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "best_metric": best_metric,
                        "best_metric_name": "logspace_auroc",
                        "epoch": epoch,
                        "step": global_step,
                        "metrics": dev_metrics,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"classifier_dev_status epoch={epoch} {json.dumps(dev_metrics, sort_keys=True)}")

    if wandb_run is not None:
        wandb_run.finish()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    n_seeds = int(config.get("n_seeds", 1))
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seed_everything(int(config["seed"]))

    settings = DataSettings.from_config(config)
    known_splits = set(available_splits(settings))
    inspect_splits = [split for split in ("train", "dev") if split in known_splits]
    inspect_data_files(settings, splits=inspect_splits)

    classifier_config = config.get("classifier", {})
    system_prompt = read_system_prompt(classifier_config["system_prompt_path"])
    tokenizer = load_tokenizer(config)
    train_examples = load_classifier_examples(
        "train",
        settings,
        system_prompt,
        positive_target=str(classifier_config.get("positive_target", "YES")),
        negative_target=str(classifier_config.get("negative_target", "NO")),
        limit=config.get("max_train_examples"),
    )
    dev_examples = load_classifier_examples(
        "dev",
        settings,
        system_prompt,
        positive_target=str(classifier_config.get("positive_target", "YES")),
        negative_target=str(classifier_config.get("negative_target", "NO")),
        limit=config.get("max_dev_examples"),
    )

    base_seed = int(config["seed"])
    base_output_dir = Path(config["output_dir"])
    base_checkpoint_dir = Path(config["checkpoint_dir"])
    if n_seeds > 1:
        write_resolved_config(config, base_output_dir)

    for seed_index in range(n_seeds):
        run_seed = base_seed + seed_index
        run_config = copy.deepcopy(config)
        run_config["seed"] = run_seed
        run_config["seed_index"] = seed_index
        if n_seeds > 1:
            run_config["output_dir"] = str(base_output_dir / f"seed_{run_seed}")
            run_config["checkpoint_dir"] = str(base_checkpoint_dir / f"seed_{run_seed}")
        output_dir = Path(run_config["output_dir"])
        checkpoint_dir = Path(run_config["checkpoint_dir"])
        print(f"classifier_seed_run seed={run_seed} seed_index={seed_index} output_dir={output_dir}")
        train_one_seed(run_config, tokenizer, train_examples, dev_examples, output_dir, checkpoint_dir)


if __name__ == "__main__":
    main()
