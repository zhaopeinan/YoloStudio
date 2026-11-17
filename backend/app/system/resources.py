from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import os
import time
from typing import Any

from app.core.settings import Settings, get_settings
from app.training.live import query_nvidia_smi

_CPU_SAMPLE: tuple[float, float, float] | None = None  # (monotonic, idle, total)


def _read_proc_stat() -> tuple[float, float] | None:
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            line = handle.readline()
    except OSError:
        return None
    if not line.startswith("cpu "):
        return None
    parts = line.split()
    try:
        values = [float(part) for part in parts[1:8]]
    except ValueError:
        return None
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


def _cpu_percent() -> float | None:
    global _CPU_SAMPLE
    sample = _read_proc_stat()
    if sample is None:
        try:
            load1, _, _ = os.getloadavg()
            ncpu = os.cpu_count() or 1
            return round(min(100.0, (load1 / ncpu) * 100.0), 1)
        except (AttributeError, OSError):
            return None

    idle, total = sample
    now = time.monotonic()
    previous = _CPU_SAMPLE
    _CPU_SAMPLE = (now, idle, total)
    if previous is None:
        time.sleep(0.05)
        second = _read_proc_stat()
        if second is None:
            return None
        idle2, total2 = second
        delta_total = total2 - total
        delta_idle = idle2 - idle
    else:
        _, prev_idle, prev_total = previous
        delta_total = total - prev_total
        delta_idle = idle - prev_idle
    if delta_total <= 0:
        return 0.0
    used = max(0.0, min(100.0, (1.0 - (delta_idle / delta_total)) * 100.0))
    return round(used, 1)


def _memory_stats() -> dict[str, float | None]:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, raw = line.partition(":")
                info[key] = int(raw.strip().split()[0])
        total_kb = info.get("MemTotal")
        available_kb = info.get("MemAvailable")
        if total_kb is None or available_kb is None:
            return {"used_mb": None, "total_mb": None, "percent": None}
        used_kb = max(0, total_kb - available_kb)
        total_mb = total_kb / 1024.0
        used_mb = used_kb / 1024.0
        return {
            "used_mb": round(used_mb, 1),
            "total_mb": round(total_mb, 1),
            "percent": round((used_mb / total_mb) * 100.0, 1) if total_mb else 0.0,
        }
    except OSError:
        return {"used_mb": None, "total_mb": None, "percent": None}


def _disk_stats(path: Path) -> dict[str, Any]:
    try:
        usage = os.statvfs(path)
    except OSError:
        return {"path": str(path), "used_gb": None, "total_gb": None, "percent": None}
    total = usage.f_frsize * usage.f_blocks
    free = usage.f_frsize * usage.f_bavail
    used = max(0, total - free)
    total_gb = total / (1024.0**3)
    used_gb = used / (1024.0**3)
    return {
        "path": str(path),
        "used_gb": round(used_gb, 2),
        "total_gb": round(total_gb, 2),
        "percent": round((used / total) * 100.0, 1) if total else 0.0,
    }


def _load_average() -> list[float] | None:
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def collect_system_resources(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    memory = _memory_stats()
    disk = _disk_stats(Path(cfg.workspace_root))
    return {
        "cpu_percent": _cpu_percent(),
        "load_average": _load_average(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": disk,
        "gpu": query_nvidia_smi(),
        "queried_at": datetime.now(UTC).isoformat(),
    }
