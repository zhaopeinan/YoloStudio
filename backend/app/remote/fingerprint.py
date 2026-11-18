from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _file_signature(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": path.name,
        "size": stat.st_size,
        "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
    }


def inventory_directory(root: Path) -> tuple[list[dict], int, int]:
    """Return (file signatures, file_count, total_bytes) under root."""
    files: list[dict] = []
    total_bytes = 0
    if not root.exists():
        return files, 0, 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        total_bytes += stat.st_size
        files.append(
            {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
            }
        )
    return files, len(files), total_bytes


def compute_sync_fingerprint(
    *,
    project_id: int,
    version_id: int,
    model_ref: str,
    version_root: Path,
    model_path: Path,
) -> tuple[str, int, int]:
    """
    Stable fingerprint for version + model payload.

    Returns (fingerprint_hex, files_total, bytes_total).
    """
    version_files, version_count, version_bytes = inventory_directory(version_root)
    model_sig = _file_signature(model_path)
    model_bytes = 0 if model_sig.get("missing") else int(model_sig["size"])
    payload = {
        "project_id": project_id,
        "version_id": version_id,
        "model_ref": model_ref,
        "version_files": version_files,
        "model": model_sig,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, version_count + (0 if model_sig.get("missing") else 1), version_bytes + model_bytes


def write_manifest(
    destination: Path,
    *,
    project_id: int,
    version_id: int,
    model_ref: str,
    fingerprint: str,
    model_remote_path: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "project_id": project_id,
                "version_id": version_id,
                "model_ref": model_ref,
                "fingerprint": fingerprint,
                "model_remote_path": model_remote_path,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
