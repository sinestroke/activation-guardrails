#!/usr/bin/env python3
"""Validate release configs, inheritance, and downloaded dataset paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier_train import load_config as load_classifier_config
from src.data import DataSettings, data_paths
from src.eval import evaluation_config, select_eval_splits
from src.train import load_config as load_probe_config


PROBE_CONFIGS = (
    REPO_ROOT / "configs/probes/high-stakes-gemma3-12b-vanilla.yaml",
    REPO_ROOT / "configs/probes/chemical-harm-gemma3-12b-vanilla.yaml",
)
CLASSIFIER_CONFIGS = (
    REPO_ROOT / "configs/classifiers/high-stakes-llama3.2-1b-vanilla.yaml",
    REPO_ROOT / "configs/classifiers/chemical-harm-llama3.2-1b-instruct.yaml",
)
EVALUATION_CONFIGS = (
    REPO_ROOT / "configs/evaluation/high-stakes.yaml",
    REPO_ROOT / "configs/evaluation/chemical-harm.yaml",
)


def missing_data_files(config: dict) -> list[Path]:
    settings = DataSettings.from_config(config)
    return sorted(
        path
        for value in settings.files.values()
        for path in data_paths(value)
        if not (REPO_ROOT / path).is_file()
    )


def main() -> None:
    errors: list[str] = []
    loaded: list[tuple[Path, dict]] = []
    for path in PROBE_CONFIGS + EVALUATION_CONFIGS:
        loaded.append((path, load_probe_config(path)))
    for path in CLASSIFIER_CONFIGS:
        loaded.append((path, load_classifier_config(path)))

    for path, config in loaded:
        missing = missing_data_files(config)
        if missing:
            errors.append(f"{path.relative_to(REPO_ROOT)}: {len(missing)} missing data paths")

    for path in EVALUATION_CONFIGS:
        config = load_probe_config(path)
        eval_config = evaluation_config(config)
        settings = DataSettings.from_config(config)
        splits = select_eval_splits(settings, eval_config)
        print(f"{path.relative_to(REPO_ROOT)} splits={json.dumps(splits)}")

    if errors:
        raise SystemExit("\n".join(errors) + "\nRun: python scripts/download_datasets.py --dataset all")
    print(f"validated {len(loaded)} release configs")


if __name__ == "__main__":
    main()
