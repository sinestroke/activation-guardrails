#!/usr/bin/env python3
"""Download the pinned Hugging Face datasets used by the release configs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "high-stakes", "chemical-harm"),
        default="all",
        help="Dataset to download; defaults to both.",
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def validate_files(local_dir: Path, required_files: list[str]) -> None:
    missing = [str(local_dir / relative) for relative in required_files if not (local_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError("Dataset download is missing required files:\n" + "\n".join(missing))


def download(name: str, spec: dict, token_env: str, force_download: bool) -> None:
    token = os.environ.get(token_env)
    if spec.get("requires_token") and not token:
        raise RuntimeError(
            f"{spec['repo_id']} requires Hugging Face authentication. "
            f"Accept the dataset access terms and set {token_env}."
        )

    local_dir = REPO_ROOT / spec["local_dir"]
    print(f"downloading {name}: {spec['repo_id']}@{spec['revision']} -> {local_dir}")
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=spec["revision"],
        local_dir=local_dir,
        token=token,
        force_download=force_download,
    )
    validate_files(local_dir, list(spec["required_files"]))
    print(f"validated {len(spec['required_files'])} required files for {name}")


def main() -> None:
    args = parse_args()
    datasets = load_manifest()["datasets"]
    selected = list(datasets) if args.dataset == "all" else [args.dataset]
    for name in selected:
        download(name, datasets[name], args.token_env, args.force_download)


if __name__ == "__main__":
    main()

