from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models import DatasetVersion, ExportArtifact, TrainingRun
from app.exports.rknn_convert import collect_calibration_images


SUPPORTED_FORMATS = {"pt", "onnx", "tensorrt", "rknn"}

_RKNN_PYTHON_CANDIDATES = (
    Path("/root/anaconda3/envs/rknn/bin/python"),
    Path("/opt/conda/envs/rknn/bin/python"),
    Path("/usr/local/bin/python-rknn"),
)
_resolved_rknn_python: Path | None | bool = False


def _now() -> datetime:
    return datetime.now(UTC)


def best_weights_path(run: TrainingRun) -> Path:
    return Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"


def export_root(run: TrainingRun) -> Path:
    return Path(run.artifact_path) / "exports"


def tensor_rt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def _python_has_rknn(python_path: Path) -> bool:
    if not python_path.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python_path), "-c", "from rknn.api import RKNN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def resolve_rknn_python(*, force: bool = False) -> Path | None:
    global _resolved_rknn_python
    if not force and _resolved_rknn_python is not False:
        return _resolved_rknn_python if isinstance(_resolved_rknn_python, Path) else None

    settings = get_settings()
    candidates: list[Path] = []
    if settings.rknn_python is not None:
        candidates.append(Path(settings.rknn_python))
    env_python = os.environ.get("YOLOSTUDIO_RKNN_PYTHON") or os.environ.get(
        "YOLO_TRAINER_RKNN_PYTHON"
    )
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(Path(sys.executable))
    candidates.extend(_RKNN_PYTHON_CANDIDATES)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _python_has_rknn(candidate):
            _resolved_rknn_python = candidate
            return candidate

    _resolved_rknn_python = None
    return None


def rknn_available() -> bool:
    return resolve_rknn_python() is not None


def _has_onnx_artifact(run: TrainingRun) -> bool:
    root = export_root(run)
    if (root / f"run-{run.id}.onnx").is_file():
        return True
    return any(path.is_file() for path in root.glob(f"run-{run.id}*.onnx"))


def export_capabilities(run: TrainingRun) -> dict:
    weights = best_weights_path(run)
    reasons: dict[str, str] = {}
    if not weights.exists():
        reasons["pt"] = f"Model weights were not found: {weights}"
        reasons["onnx"] = f"Model weights were not found: {weights}"
        reasons["tensorrt"] = f"Model weights were not found: {weights}"
        if not _has_onnx_artifact(run):
            reasons["rknn"] = f"Model weights were not found: {weights}"
    if importlib.util.find_spec("ultralytics") is None:
        reasons["onnx"] = "Ultralytics is not installed"
        if "rknn" not in reasons and not _has_onnx_artifact(run):
            reasons["rknn"] = "Ultralytics is not installed"
    if not tensor_rt_available():
        reasons["tensorrt"] = "TensorRT Python package is not installed"
    if not rknn_available():
        reasons["rknn"] = "rknn-toolkit2 is not installed"

    return {
        "pt_available": weights.exists(),
        "onnx_available": weights.exists() and "onnx" not in reasons,
        "tensorrt_available": weights.exists() and "tensorrt" not in reasons,
        "rknn_available": (weights.exists() or _has_onnx_artifact(run)) and "rknn" not in reasons,
        "weights_path": str(weights) if weights.exists() else None,
        "reasons": reasons,
    }


def create_export_artifact(db: Session, run: TrainingRun, export_format: str) -> ExportArtifact:
    if export_format not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported export format")
    artifact = ExportArtifact(
        run_id=run.id,
        project_id=run.project_id,
        format=export_format,
        status="queued",
        artifact_path="",
        metadata_={},
    )
    db.add(artifact)
    db.flush()
    export_root(run).mkdir(parents=True, exist_ok=True)
    artifact.artifact_path = str(export_root(run) / f"run-{run.id}.{export_format}")
    db.commit()
    db.refresh(artifact)
    return artifact


def complete_export(
    db: Session,
    artifact: ExportArtifact,
    artifact_path: Path,
    metadata: dict | None = None,
) -> ExportArtifact:
    artifact.status = "completed"
    artifact.artifact_path = str(artifact_path)
    artifact.metadata_ = metadata or {}
    artifact.ended_at = _now()
    db.commit()
    db.refresh(artifact)
    return artifact


def fail_export(db: Session, artifact: ExportArtifact, message: str) -> ExportArtifact:
    artifact.status = "failed"
    artifact.error_message = message
    artifact.ended_at = _now()
    db.commit()
    db.refresh(artifact)
    return artifact


def execute_export(db: Session, artifact: ExportArtifact, run: TrainingRun) -> ExportArtifact:
    artifact.status = "running"
    artifact.started_at = _now()
    db.commit()
    db.refresh(artifact)

    try:
        if artifact.format == "pt":
            return _export_pt(db, artifact, run)
        if artifact.format == "onnx":
            return _export_onnx(db, artifact, run)
        if artifact.format == "tensorrt":
            return _export_tensorrt(db, artifact, run)
        if artifact.format == "rknn":
            return _export_rknn(db, artifact, run)
    except Exception as exc:
        return fail_export(db, artifact, str(exc))

    return fail_export(db, artifact, "Unsupported export format")


def _export_pt(db: Session, artifact: ExportArtifact, run: TrainingRun) -> ExportArtifact:
    weights = best_weights_path(run)
    if not weights.exists():
        raise RuntimeError(f"Model weights were not found: {weights}")

    destination = export_root(run) / f"run-{run.id}.pt"
    if weights.resolve() != destination.resolve():
        shutil.copy2(weights, destination)
    return complete_export(
        db,
        artifact,
        destination,
        {"source": str(weights), "format": "pt"},
    )


def _export_onnx(db: Session, artifact: ExportArtifact, run: TrainingRun) -> ExportArtifact:
    destination = _ensure_onnx(run)
    return complete_export(
        db,
        artifact,
        destination,
        {"source": str(best_weights_path(run)), "format": "onnx"},
    )


def _export_tensorrt(db: Session, artifact: ExportArtifact, run: TrainingRun) -> ExportArtifact:
    if not tensor_rt_available():
        return fail_export(db, artifact, "TensorRT export is not supported in this environment")

    weights = best_weights_path(run)
    if not weights.exists():
        raise RuntimeError(f"Model weights were not found: {weights}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not available: {exc.__class__.__name__}") from exc

    model = YOLO(str(weights))
    exported = Path(model.export(format="engine"))
    destination = export_root(run) / f"run-{run.id}.engine"
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    return complete_export(
        db,
        artifact,
        destination,
        {"source": str(weights), "format": "tensorrt"},
    )


def _ensure_onnx(run: TrainingRun) -> Path:
    destination = export_root(run) / f"run-{run.id}.onnx"
    if destination.is_file():
        return destination

    # Accept previously generated alternate ONNX names.
    for candidate in export_root(run).glob(f"run-{run.id}*.onnx"):
        if candidate.is_file():
            if candidate.resolve() != destination.resolve():
                shutil.copy2(candidate, destination)
            return destination

    weights = best_weights_path(run)
    if not weights.exists():
        raise RuntimeError(f"Model weights were not found: {weights}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not available: {exc.__class__.__name__}") from exc

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx"))
    export_root(run).mkdir(parents=True, exist_ok=True)
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    return destination


def _calibration_image_roots(db: Session, run: TrainingRun) -> list[Path]:
    roots: list[Path] = []
    version = db.get(DatasetVersion, run.version_id)
    if version and version.artifact_path:
        version_root = Path(version.artifact_path)
        for split in ("val", "train", "test"):
            roots.append(version_root / "images" / split)
    # Fallback: any images next to the run artifact.
    roots.append(Path(run.artifact_path) / "images" / "val")
    roots.append(Path(run.artifact_path) / "images" / "train")
    return roots


def _write_calibration_dataset(images: list[Path], dataset_txt: Path) -> Path:
    dataset_txt.parent.mkdir(parents=True, exist_ok=True)
    dataset_txt.write_text("".join(f"{path}\n" for path in images), encoding="utf-8")
    return dataset_txt


def _export_rknn(db: Session, artifact: ExportArtifact, run: TrainingRun) -> ExportArtifact:
    python_path = resolve_rknn_python()
    if python_path is None:
        return fail_export(db, artifact, "rknn-toolkit2 is not installed")

    settings = get_settings()
    target_platform = settings.rknn_target_platform or "rk3588"
    quant_mode = (settings.rknn_quant_mode or "fp16").lower()
    if quant_mode not in {"hybrid", "fp16", "int8"}:
        quant_mode = "fp16"

    onnx_path = _ensure_onnx(run)
    destination = export_root(run) / f"run-{run.id}.rknn"

    images = collect_calibration_images(_calibration_image_roots(db, run), limit=100)
    needs_dataset = quant_mode in {"hybrid", "int8"}
    if needs_dataset and not images:
        # Without calibration images, fall back to full FP16 rather than a broken INT8 head.
        quant_mode = "fp16"

    with tempfile.TemporaryDirectory(prefix=f"rknn-export-{run.id}-") as tmp:
        tmp_root = Path(tmp)
        dataset_txt = None
        if quant_mode in {"hybrid", "int8"}:
            dataset_txt = _write_calibration_dataset(images, tmp_root / "dataset.txt")

        # Hybrid step1 writes intermediate files next to the output; use exports/.
        convert_module = Path(__file__).with_name("rknn_convert.py")
        command = [
            str(python_path),
            str(convert_module),
            "--onnx",
            str(onnx_path),
            "--output",
            str(destination),
            "--target-platform",
            target_platform,
            "--quant-mode",
            quant_mode,
        ]
        if dataset_txt is not None:
            command.extend(["--dataset", str(dataset_txt)])

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"RKNN conversion failed with code {completed.returncode}")

    return complete_export(
        db,
        artifact,
        destination,
        {
            "source": str(best_weights_path(run)),
            "onnx": str(onnx_path),
            "format": "rknn",
            "target_platform": target_platform,
            "quantization": quant_mode,
            "calibration_images": len(images),
            "rknn_python": str(python_path),
            "note": (
                "YOLOv8 packs box coords and class scores in one tensor; "
                "full INT8 zeros class scores. hybrid/fp16 keeps outputs usable."
            ),
        },
    )
