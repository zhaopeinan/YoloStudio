from __future__ import annotations

import importlib.util
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import ExportArtifact, TrainingRun


SUPPORTED_FORMATS = {"pt", "onnx", "tensorrt"}


def _now() -> datetime:
    return datetime.now(UTC)


def best_weights_path(run: TrainingRun) -> Path:
    return Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"


def export_root(run: TrainingRun) -> Path:
    return Path(run.artifact_path) / "exports"


def tensor_rt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def export_capabilities(run: TrainingRun) -> dict:
    weights = best_weights_path(run)
    reasons: dict[str, str] = {}
    if not weights.exists():
        reasons["pt"] = f"Model weights were not found: {weights}"
        reasons["onnx"] = f"Model weights were not found: {weights}"
        reasons["tensorrt"] = f"Model weights were not found: {weights}"
    if importlib.util.find_spec("ultralytics") is None:
        reasons["onnx"] = "Ultralytics is not installed"
    if not tensor_rt_available():
        reasons["tensorrt"] = "TensorRT Python package is not installed"

    return {
        "pt_available": weights.exists(),
        "onnx_available": weights.exists() and "onnx" not in reasons,
        "tensorrt_available": weights.exists() and "tensorrt" not in reasons,
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
    weights = best_weights_path(run)
    if not weights.exists():
        raise RuntimeError(f"Model weights were not found: {weights}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not available: {exc.__class__.__name__}") from exc

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx"))
    destination = export_root(run) / f"run-{run.id}.onnx"
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    return complete_export(
        db,
        artifact,
        destination,
        {"source": str(weights), "format": "onnx"},
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
