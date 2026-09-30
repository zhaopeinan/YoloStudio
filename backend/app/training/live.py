"""Read-only live training monitor helpers (CSV + log tail + nvidia-smi).

These never touch the Ultralytics training thread; they only read files / spawn
short-timeout nvidia-smi queries.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.models import TrainingRun

# Preferred series shown by the live monitor UI (others still available in `latest`).
PREFERRED_SERIES = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
)

_LIVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_GPU_CACHE: tuple[float, dict[str, Any]] | None = None
_CACHE_TTL_SEC = 0.4


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _to_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_log_tail(path: Path | str | None, *, max_bytes: int = 8192, max_lines: int = 40) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return []
    try:
        size = file_path.stat().st_size
        with file_path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # drop partial first line
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def parse_results_csv(csv_path: Path) -> tuple[list[int], dict[str, list[float | None]], dict[str, float]]:
    """Return (epochs, series, latest_nonzero_or_last)."""
    if not csv_path.exists() or not csv_path.is_file():
        return [], {}, {}

    try:
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return [], {}, {}
            fieldnames = [name.strip() for name in reader.fieldnames]
            reader.fieldnames = fieldnames

            epochs: list[int] = []
            series: dict[str, list[float | None]] = {
                name: [] for name in fieldnames if name and name != "epoch"
            }

            for row in reader:
                epoch_raw = (row.get("epoch") or "").strip()
                epoch_val = _to_float(epoch_raw)
                if epoch_val is None:
                    continue
                epochs.append(int(epoch_val))
                for name in series:
                    series[name].append(_to_float((row.get(name) or "").strip()))
    except OSError:
        return [], {}, {}

    latest: dict[str, float] = {}
    for name, values in series.items():
        for value in reversed(values):
            if value is not None:
                latest[name] = value
                break

    # Keep preferred keys first; drop empty-only series from payload size later.
    return epochs, series, latest


def query_nvidia_smi(*, timeout_sec: float = 1.0) -> dict[str, Any]:
    global _GPU_CACHE
    now = time.monotonic()
    if _GPU_CACHE is not None and now - _GPU_CACHE[0] < _CACHE_TTL_SEC:
        return _GPU_CACHE[1]

    binary = shutil.which("nvidia-smi")
    if not binary:
        payload = {
            "available": False,
            "gpus": [],
            "error": "nvidia-smi not found",
            "queried_at": _now_iso(),
        }
        _GPU_CACHE = (now, payload)
        return payload

    query = (
        "index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [
                binary,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        payload = {
            "available": False,
            "gpus": [],
            "error": str(exc),
            "queried_at": _now_iso(),
        }
        _GPU_CACHE = (now, payload)
        return payload

    if completed.returncode != 0:
        payload = {
            "available": False,
            "gpus": [],
            "error": (completed.stderr or completed.stdout or "nvidia-smi failed").strip()[:400],
            "queried_at": _now_iso(),
        }
        _GPU_CACHE = (now, payload)
        return payload

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue

        def num(idx: int) -> float | None:
            return _to_float(parts[idx])

        index_val = num(0)
        gpus.append(
            {
                "index": int(index_val) if index_val is not None else len(gpus),
                "name": parts[1],
                "utilization_gpu": num(2),
                "memory_used_mb": num(3),
                "memory_total_mb": num(4),
                "temperature_c": num(5),
                "power_w": num(6),
                "power_limit_w": num(7),
            }
        )

    # Best-effort compute processes (optional; ignore failures).
    processes_by_gpu: dict[int, list[dict[str, Any]]] = {}
    try:
        proc = subprocess.run(
            [
                binary,
                "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        # Map by index via a second simpler query if needed — keep processes empty on mismatch.
        if proc.returncode == 0 and proc.stdout.strip():
            # Fallback: attach all listed processes to GPU 0 when only one GPU.
            if len(gpus) == 1:
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 5:
                        continue
                    pid = _to_float(parts[2])
                    mem = _to_float(parts[4])
                    processes_by_gpu.setdefault(0, []).append(
                        {
                            "pid": int(pid) if pid is not None else None,
                            "name": parts[3],
                            "memory_mb": mem,
                        }
                    )
    except (subprocess.TimeoutExpired, OSError):
        pass

    for gpu in gpus:
        gpu["processes"] = processes_by_gpu.get(int(gpu["index"]), [])

    payload = {
        "available": len(gpus) > 0,
        "gpus": gpus,
        "error": None if gpus else "no GPUs reported",
        "queried_at": _now_iso(),
    }
    _GPU_CACHE = (now, payload)
    return payload


def build_live_snapshot(run: TrainingRun, *, include_gpu: bool = True) -> dict[str, Any]:
    cache_key = f"{run.id}:{int(include_gpu)}"
    now = time.monotonic()
    cached = _LIVE_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    config = dict(run.config or {})
    total_epochs = int(config.get("epochs") or 0) or None

    artifact_root = Path(run.artifact_path) if run.artifact_path else None
    csv_path = (artifact_root / "ultralytics" / "results.csv") if artifact_root else None
    epochs, series_raw, latest = parse_results_csv(csv_path) if csv_path else ([], {}, {})

    current_epoch = epochs[-1] if epochs else None
    percent: float | None = None
    if current_epoch is not None and total_epochs and total_epochs > 0:
        percent = round(min(100.0, max(0.0, (current_epoch / total_epochs) * 100.0)), 2)

    elapsed_sec: float | None = None
    if "time" in latest:
        elapsed_sec = latest["time"]
    elif run.started_at is not None:
        started = run.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        elapsed_sec = max(0.0, (datetime.now(UTC) - started).total_seconds())

    eta_sec: float | None = None
    if (
        elapsed_sec is not None
        and current_epoch is not None
        and total_epochs is not None
        and current_epoch > 0
        and total_epochs > current_epoch
    ):
        eta_sec = round(elapsed_sec * (total_epochs - current_epoch) / current_epoch, 1)

    # Trim series payload: preferred keys + any other non-empty series (cap).
    series: dict[str, list[float | None]] = {
        "epoch": [float(e) for e in epochs],
    }
    preferred_set = set(PREFERRED_SERIES)
    for name in PREFERRED_SERIES:
        if name in series_raw:
            series[name] = series_raw[name]
    extras = 0
    for name, values in series_raw.items():
        if name in preferred_set or name == "time":
            if name == "time":
                series[name] = values
            continue
        if extras >= 8:
            break
        if any(v is not None for v in values):
            series[name] = values
            extras += 1

    status_label = {
        "queued": "排队中",
        "preparing": "准备中",
        "running": "训练中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
    }.get(run.status, run.status)

    payload: dict[str, Any] = {
        "run_id": run.id,
        "project_id": run.project_id,
        "status": run.status,
        "status_label": status_label,
        "device": run.device,
        "config": {
            "model": config.get("model"),
            "epochs": total_epochs,
            "image_size": config.get("image_size"),
            "batch_size": config.get("batch_size"),
            "augmentation_preset": config.get("augmentation_preset"),
        },
        "progress": {
            "epoch": current_epoch,
            "total_epochs": total_epochs,
            "percent": percent,
            "elapsed_sec": elapsed_sec,
            "eta_sec": eta_sec,
            "phase": run.status,
        },
        "latest": latest,
        "series": series,
        "log_tail": read_log_tail(run.log_path),
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "updated_at": _now_iso(),
        "gpu": query_nvidia_smi() if include_gpu else None,
    }
    _LIVE_CACHE[cache_key] = (now, payload)
    return payload
