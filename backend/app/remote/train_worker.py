#!/usr/bin/env python3
"""Standalone remote training worker executed on GPU05 via SSH."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {message}\n")


def _write_status(status_path: Path, payload: dict) -> None:
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_data_yaml(source: Path, dataset_root: Path) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()
    rewritten = [f"path: {dataset_root}"]
    rewritten.extend(line for line in lines if not line.startswith("path:"))
    destination = dataset_root / "data.yaml"
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    artifact_root = Path(config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(config["log_path"])
    status_path = artifact_root / "remote_status.json"
    metrics_path = artifact_root / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    (artifact_root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_status(
        status_path,
        {"status": "running", "pid": None, "error": None, "started_at": _now()},
    )
    _log(log_path, "remote worker started")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        message = f"Ultralytics is not available: {exc.__class__.__name__}: {exc}"
        _log(log_path, message)
        _write_status(
            status_path,
            {
                "status": "failed",
                "pid": None,
                "error": message,
                "ended_at": _now(),
            },
        )
        return 1

    try:
        version_root = Path(config["version_root"])
        data_yaml = version_root / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"data.yaml missing: {data_yaml}")
        training_yaml = _rewrite_data_yaml(data_yaml, version_root)
        model = YOLO(config["model_path"])
        _log(
            log_path,
            (
                "ultralytics training started "
                f"(model={config.get('model_ref')} -> {config['model_path']})"
            ),
        )
        results = model.train(
            data=str(training_yaml),
            epochs=int(config["epochs"]),
            imgsz=int(config["image_size"]),
            batch=int(config["batch_size"]),
            device=str(config["device"]),
            project=str(artifact_root),
            name="ultralytics",
            exist_ok=True,
            **dict(config.get("augmentation_kwargs") or {}),
        )
        metrics = getattr(results, "results_dict", {}) or {}
        with metrics_path.open("a", encoding="utf-8") as handle:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    handle.write(
                        json.dumps(
                            {
                                "epoch": int(config["epochs"]),
                                "name": name,
                                "value": float(value),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        _log(log_path, "training completed")
        _write_status(
            status_path,
            {
                "status": "completed",
                "pid": None,
                "error": None,
                "ended_at": _now(),
            },
        )
        return 0
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        _log(log_path, f"training failed: {message}")
        _log(log_path, traceback.format_exc())
        _write_status(
            status_path,
            {
                "status": "failed",
                "pid": None,
                "error": message,
                "ended_at": _now(),
            },
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
