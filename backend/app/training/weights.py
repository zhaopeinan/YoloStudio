from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.db.models import ModelWeight, TrainingRun
from app.storage.visibility import active_entity_predicate


WEIGHT_EXTENSIONS = {".pt"}
MAX_WEIGHT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
BASE_MODEL_OPTIONS = (
    ("base:yolov8n.pt", "内置模型 · YOLOv8n", "yolov8n.pt"),
    ("base:yolov8s.pt", "内置模型 · YOLOv8s", "yolov8s.pt"),
    ("base:yolov8m.pt", "内置模型 · YOLOv8m", "yolov8m.pt"),
)


def model_weight_root(settings: Settings, project_id: int, weight_id: int) -> Path:
    return (
        settings.workspace_root
        / "projects"
        / str(project_id)
        / "model-weights"
        / str(weight_id)
    )


def _validate_weight_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in WEIGHT_EXTENSIONS:
        raise ValueError("仅支持上传 .pt 权重文件")
    return suffix


def _bundled_pretrained_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundled_model_path(ultralytics_name: str) -> Path | None:
    candidates = [
        _bundled_pretrained_dir() / ultralytics_name,
        Path("/opt/YOLO_Trainer/backend") / ultralytics_name,
        Path("/opt/YOLO_Trainer/pretrained") / ultralytics_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def create_model_weight(
    db: Session,
    settings: Settings,
    project_id: int,
    source_filename: str,
    source_bytes: bytes,
) -> ModelWeight:
    if len(source_bytes) > MAX_WEIGHT_BYTES:
        raise ValueError("权重文件不能超过 2 GB")
    if len(source_bytes) == 0:
        raise ValueError("权重文件为空")
    suffix = _validate_weight_filename(source_filename)
    weight = ModelWeight(
        project_id=project_id,
        original_filename=Path(source_filename).name,
        stored_path="",
        size_bytes=len(source_bytes),
    )
    db.add(weight)
    db.flush()
    root = model_weight_root(settings, project_id, weight.id)
    root.mkdir(parents=True, exist_ok=True)
    stored_path = root / f"weights{suffix}"
    stored_path.write_bytes(source_bytes)
    weight.stored_path = str(stored_path)
    db.commit()
    db.refresh(weight)
    return weight


def list_model_weights(db: Session, project_id: int) -> list[ModelWeight]:
    return list(
        db.scalars(
            select(ModelWeight)
            .where(ModelWeight.project_id == project_id)
            .order_by(ModelWeight.id.desc())
        ).all()
    )


def delete_model_weight(db: Session, settings: Settings, weight_id: int) -> None:
    weight = db.get(ModelWeight, weight_id)
    if weight is None:
        raise LookupError("权重不存在")
    root = model_weight_root(settings, weight.project_id, weight.id)
    db.delete(weight)
    db.commit()
    shutil.rmtree(root, ignore_errors=True)


def resolve_training_model(
    db: Session,
    settings: Settings,
    project_id: int,
    model_ref: str,
) -> str:
    """Return a path or Ultralytics model name suitable for YOLO(...)."""
    ref = (model_ref or "").strip()
    if not ref:
        raise ValueError("请选择起始模型")

    for option_ref, label, ultralytics_name in BASE_MODEL_OPTIONS:
        if ref == option_ref or ref == ultralytics_name:
            bundled = _bundled_model_path(ultralytics_name)
            if bundled is not None:
                return str(bundled.resolve())
            raise ValueError(
                f"{label} 的本地权重文件不存在（{ultralytics_name}）。"
                "服务器无法从 GitHub 自动下载，请联系管理员预置权重，"
                "或改用已上传权重 / 已完成训练任务的 best.pt。"
            )

    if ref.startswith("run:") and ref[4:].isdigit():
        run_id = int(ref[4:])
        run = db.scalar(
            select(TrainingRun).where(
                TrainingRun.id == run_id,
                TrainingRun.project_id == project_id,
                TrainingRun.status == "completed",
                active_entity_predicate("training_run", TrainingRun.id),
            )
        )
        if run is None:
            raise ValueError("训练任务不存在或尚未完成")
        weights = Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"
        if not weights.is_file():
            raise ValueError(f"训练任务 #{run_id} 缺少 best.pt")
        return str(weights.resolve())

    if ref.startswith("weight:") and ref[7:].isdigit():
        weight_id = int(ref[7:])
        weight = db.get(ModelWeight, weight_id)
        if weight is None or weight.project_id != project_id:
            raise ValueError("上传权重不存在")
        path = Path(weight.stored_path)
        if not path.is_file():
            raise ValueError("上传权重文件已丢失")
        return str(path.resolve())

    # Legacy absolute / relative paths that still exist on disk.
    candidate = Path(ref)
    if candidate.is_file():
        return str(candidate.resolve())

    # Allow Ultralytics named checkpoints (e.g. historical custom-drone.pt configs
    # that lived as free-text names) to pass through unchanged.
    return ref


def list_training_model_options(db: Session, project_id: int) -> list[dict]:
    items: list[dict] = []
    for model_ref, label, ultralytics_name in BASE_MODEL_OPTIONS:
        available = _bundled_model_path(ultralytics_name) is not None
        items.append(
            {
                "model_ref": model_ref,
                "label": label if available else f"{label}（未预置）",
                "kind": "base",
                "run_id": None,
                "weight_id": None,
                "status": "可用" if available else "未预置",
            }
        )

    runs = db.scalars(
        select(TrainingRun)
        .where(
            TrainingRun.project_id == project_id,
            TrainingRun.status == "completed",
            active_entity_predicate("training_run", TrainingRun.id),
        )
        .order_by(TrainingRun.id.desc())
    ).all()
    for run in runs:
        weights = Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"
        if weights.is_file():
            items.append(
                {
                    "model_ref": f"run:{run.id}",
                    "label": f"训练任务 #{run.id} · best.pt",
                    "kind": "trained",
                    "run_id": run.id,
                    "weight_id": None,
                    "status": run.status,
                }
            )

    for weight in list_model_weights(db, project_id):
        items.append(
            {
                "model_ref": f"weight:{weight.id}",
                "label": f"上传权重 · {weight.original_filename}",
                "kind": "upload",
                "run_id": None,
                "weight_id": weight.id,
                "status": "可用",
            }
        )
    return items
