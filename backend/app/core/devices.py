from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    selected: str
    available: list[str]
    details: dict[str, str]


def detect_devices() -> DeviceInfo:
    available = ["cpu"]
    details: dict[str, str] = {}

    try:
        import torch

        if torch.cuda.is_available():
            available.insert(0, "cuda")
            details["cuda"] = torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            insert_at = 1 if "cuda" in available else 0
            available.insert(insert_at, "mps")
            details["mps"] = "Apple Metal Performance Shaders"
    except Exception as exc:
        details["torch"] = f"unavailable: {exc.__class__.__name__}"

    return DeviceInfo(selected=available[0], available=available, details=details)
