"""Small, dependency-light helpers for reproducible artifact provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROVENANCE_SCHEMA_VERSION = 1


def safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(safe_value(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def file_record(path: str | Path, *, include_sha256: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if not resolved.exists():
        return record
    stat = resolved.stat()
    record.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if include_sha256 and resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
    return record


def file_records(paths: Iterable[str | Path], *, include_sha256: bool = True) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for path in paths:
        key = str(Path(path).expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        records.append(file_record(path, include_sha256=include_sha256))
    return records


def git_metadata(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "repository_root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status.splitlines() if status else [],
    }


def runtime_metadata() -> dict[str, Any]:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_PROCID",
        "SLURM_LOCALID",
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def model_metadata(model: Any, requested_model_name: str, requested_revision: str | None = None) -> dict[str, Any]:
    candidates = [model]
    for attr in ("_model", "model", "module"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate is not model:
            candidates.append(candidate)
    model_config = next((getattr(candidate, "config", None) for candidate in candidates if getattr(candidate, "config", None) is not None), None)
    return {
        "requested_model_name": str(requested_model_name),
        "requested_revision": requested_revision,
        "resolved_name_or_path": getattr(model_config, "_name_or_path", None),
        "resolved_commit_hash": getattr(model_config, "_commit_hash", None),
        "model_type": getattr(model_config, "model_type", None),
        "architectures": list(getattr(model_config, "architectures", None) or []),
        "model_class": type(candidates[-1]).__name__,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(safe_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

