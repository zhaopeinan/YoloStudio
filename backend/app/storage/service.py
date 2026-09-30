from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import shutil
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.db.models import (
    Annotation,
    ClassDef,
    Dataset,
    DatasetVersion,
    ExportArtifact,
    Image,
    Prediction,
    PredictionJob,
    Project,
    RunMetric,
    TrashItem,
    TrainingRun,
)
from app.storage.schemas import (
    RelatedRunRead,
    StorageBlocker,
    StorageItemDetail,
    StorageItemList,
    StorageItemRead,
    StorageMutationResponse,
    PurgeSummary,
    ReconcileSummary,
    TrashItemList,
    TrashItemRead,
)
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    is_entity_trashed,
    require_active_entity,
)


ENDED_RUN_STATUSES = {"completed", "failed", "cancelled"}
ENDED_CHILD_JOB_STATUSES = {"completed", "failed", "cancelled"}
TRASH_RETENTION_DAYS = 30
CATALOG_ENTITY_TYPES = {"dataset", "dataset_version"}
TRASH_ENTITY_TYPES = {"dataset", "dataset_version", "training_run"}


class StoragePathError(ValueError):
    pass


class StorageConflictError(RuntimeError):
    def __init__(self, message: str, blockers: list[StorageBlocker] | None = None):
        super().__init__(message)
        self.blockers = [blocker.model_dump() for blocker in blockers or []]


class StorageMoveError(RuntimeError):
    pass


class StorageConfirmationError(ValueError):
    pass


def require_workspace_path(path: Path, workspace_root: Path) -> Path:
    resolved = path.resolve()
    root = workspace_root.resolve()
    if not resolved.is_relative_to(root):
        raise StoragePathError("数据路径不在当前工作空间内。")
    return resolved


def _canonical_path(workspace_root: Path, *parts: str) -> Path:
    root = workspace_root.resolve()
    canonical = root.joinpath(*parts)
    resolved = require_workspace_path(canonical, root)
    if resolved != canonical:
        raise StoragePathError("存储记录路径与规范目录不一致")
    return canonical


def _validate_stored_artifact_path(
    stored_path: str,
    expected_path: Path,
    workspace_root: Path,
) -> None:
    stored = Path(stored_path)
    if ".." in stored.parts:
        raise StoragePathError("存储路径不能包含目录穿越")

    root = workspace_root.resolve()
    candidate = stored if stored.is_absolute() else root / stored
    require_workspace_path(candidate, root)
    if candidate != expected_path:
        raise StoragePathError("存储记录路径与规范目录不一致")


def expected_entity_path(
    settings: Settings,
    entity_type: str,
    entity: Dataset | DatasetVersion | TrainingRun,
) -> Path:
    if entity_type == "dataset" and isinstance(entity, Dataset):
        return _canonical_path(
            settings.workspace_root,
            "projects",
            str(entity.project_id),
            "datasets",
            str(entity.id),
        )
    if entity_type == "dataset_version" and isinstance(entity, DatasetVersion):
        expected = _canonical_path(
            settings.workspace_root,
            "projects",
            str(entity.project_id),
            "versions",
            str(entity.id),
        )
        _validate_stored_artifact_path(
            entity.artifact_path, expected, settings.workspace_root
        )
        return expected
    if entity_type == "training_run" and isinstance(entity, TrainingRun):
        expected = _canonical_path(
            settings.workspace_root,
            "projects",
            str(entity.project_id),
            "runs",
            str(entity.id),
        )
        _validate_stored_artifact_path(
            entity.artifact_path, expected, settings.workspace_root
        )
        return expected
    raise StorageEntityNotFoundError("不支持的存储对象类型")


def directory_size(path: Path, workspace_root: Path | None = None) -> int:
    resolved_root = path.resolve()
    if workspace_root is not None and not resolved_root.is_relative_to(workspace_root.resolve()):
        raise StoragePathError("存储路径不在工作区内")
    if not resolved_root.exists():
        return 0
    if resolved_root.is_file():
        return resolved_root.stat().st_size

    total = 0
    for item in resolved_root.rglob("*"):
        if not item.is_file():
            continue
        resolved_item = item.resolve()
        if workspace_root is not None and not resolved_item.is_relative_to(
            workspace_root.resolve()
        ):
            raise StoragePathError("存储目录包含工作区外的文件")
        total += resolved_item.stat().st_size
    return total


def _dataset_path(settings: Settings, dataset: Dataset) -> Path:
    return expected_entity_path(settings, "dataset", dataset)


def _version_path(settings: Settings, version: DatasetVersion) -> Path:
    return expected_entity_path(settings, "dataset_version", version)


def _run_path(settings: Settings, run: TrainingRun) -> Path:
    return expected_entity_path(settings, "training_run", run)


def _active_versions(db: Session, dataset_id: int) -> list[DatasetVersion]:
    return list(
        db.scalars(
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id == dataset_id,
                active_entity_predicate("dataset_version", DatasetVersion.id),
            )
            .order_by(DatasetVersion.id)
        ).all()
    )


def _active_runs(db: Session, version_ids: list[int]) -> list[TrainingRun]:
    if not version_ids:
        return []
    return list(
        db.scalars(
            select(TrainingRun)
            .where(
                TrainingRun.version_id.in_(version_ids),
                active_entity_predicate("training_run", TrainingRun.id),
            )
            .order_by(TrainingRun.id)
        ).all()
    )


def _dataset_counts(db: Session, dataset_id: int) -> tuple[int, int]:
    image_count = db.scalar(
        select(func.count(Image.id)).where(Image.dataset_id == dataset_id)
    ) or 0
    annotation_count = db.scalar(
        select(func.count(Annotation.id))
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id)
    ) or 0
    return int(image_count), int(annotation_count)


def _version_counts(version: DatasetVersion) -> tuple[int, int, dict[str, int]]:
    manifest = version.split_manifest or {}
    images = manifest.get("images") or []
    split_counts_source = manifest.get("split_counts") or {}
    split_counts = {
        split: int(split_counts_source.get(split, 0))
        for split in ("train", "val", "test")
    }
    annotation_count = sum(len(item.get("annotations") or []) for item in images)
    return len(images), annotation_count, split_counts


def _dataset_item(
    db: Session,
    settings: Settings,
    dataset: Dataset,
    project_name: str,
) -> StorageItemRead:
    versions = _active_versions(db, dataset.id)
    blockers = [
        StorageBlocker(
            entity_type="dataset_version",
            entity_id=version.id,
            display_name=version.name,
        )
        for version in versions
    ]
    image_count, annotation_count = _dataset_counts(db, dataset.id)
    artifact_path = _dataset_path(settings, dataset)
    return StorageItemRead(
        entity_type="dataset",
        entity_id=dataset.id,
        display_name=dataset.name,
        project_id=dataset.project_id,
        project_name=project_name,
        dataset_id=dataset.id,
        version_id=None,
        artifact_path=str(artifact_path),
        size_bytes=directory_size(artifact_path, settings.workspace_root),
        image_count=image_count,
        annotation_count=annotation_count,
        split_counts=None,
        created_at=dataset.created_at,
        protected=bool(blockers),
        blockers=blockers,
    )


def _version_item(
    db: Session,
    settings: Settings,
    version: DatasetVersion,
    project_name: str,
) -> StorageItemRead:
    runs = _active_runs(db, [version.id])
    blockers = [
        StorageBlocker(
            entity_type="training_run",
            entity_id=run.id,
            display_name=f"训练任务 #{run.id}",
            status=run.status,
        )
        for run in runs
    ]
    image_count, annotation_count, split_counts = _version_counts(version)
    artifact_path = _version_path(settings, version)
    return StorageItemRead(
        entity_type="dataset_version",
        entity_id=version.id,
        display_name=version.name,
        project_id=version.project_id,
        project_name=project_name,
        dataset_id=version.dataset_id,
        version_id=version.id,
        artifact_path=str(artifact_path),
        size_bytes=directory_size(artifact_path, settings.workspace_root),
        image_count=image_count,
        annotation_count=annotation_count,
        split_counts=split_counts,
        created_at=version.created_at,
        protected=bool(blockers),
        blockers=blockers,
    )


def list_storage_items(db: Session, settings: Settings) -> StorageItemList:
    items: list[StorageItemRead] = []
    datasets = db.execute(
        select(Dataset, Project.name)
        .join(Project, Project.id == Dataset.project_id)
        .where(active_entity_predicate("dataset", Dataset.id))
    ).all()
    for dataset, project_name in datasets:
        items.append(_dataset_item(db, settings, dataset, project_name))

    versions = db.execute(
        select(DatasetVersion, Project.name)
        .join(Project, Project.id == DatasetVersion.project_id)
        .where(
            active_entity_predicate("dataset_version", DatasetVersion.id),
            active_entity_predicate("dataset", DatasetVersion.dataset_id),
        )
    ).all()
    for version, project_name in versions:
        items.append(_version_item(db, settings, version, project_name))

    items.sort(key=lambda item: (item.created_at, item.entity_id), reverse=True)
    return StorageItemList(
        items=items,
        total_size_bytes=sum(item.size_bytes for item in items),
    )


def _class_names_for_dataset(db: Session, dataset_id: int) -> list[str]:
    return list(
        db.scalars(
            select(ClassDef.name)
            .join(Annotation, Annotation.class_id == ClassDef.id)
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.dataset_id == dataset_id)
            .distinct()
            .order_by(ClassDef.name)
        ).all()
    )


def _class_names_for_version(db: Session, version: DatasetVersion) -> list[str]:
    class_ids = [int(class_id) for class_id in (version.class_mapping or {})]
    if not class_ids:
        return []
    return list(
        db.scalars(
            select(ClassDef.name)
            .where(ClassDef.id.in_(class_ids))
            .order_by(ClassDef.id)
        ).all()
    )


def _related_runs(
    db: Session,
    settings: Settings,
    version_ids: list[int],
) -> list[RelatedRunRead]:
    runs = _active_runs(db, version_ids)
    related: list[RelatedRunRead] = []
    for run in runs:
        prediction_job_count = db.scalar(
            select(func.count(PredictionJob.id)).where(PredictionJob.run_id == run.id)
        ) or 0
        export_count = db.scalar(
            select(func.count(ExportArtifact.id)).where(ExportArtifact.run_id == run.id)
        ) or 0
        related.append(
            RelatedRunRead(
                id=run.id,
                status=run.status,
                model=str((run.config or {}).get("model") or ""),
                size_bytes=directory_size(_run_path(settings, run), settings.workspace_root),
                prediction_job_count=int(prediction_job_count),
                export_count=int(export_count),
                created_at=run.created_at,
            )
        )
    related.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    return related


def get_storage_item_detail(
    db: Session,
    settings: Settings,
    entity_type: str,
    entity_id: int,
) -> StorageItemDetail:
    if entity_type not in CATALOG_ENTITY_TYPES:
        raise StorageEntityNotFoundError("不支持的存储对象类型")
    require_active_entity(db, entity_type, entity_id)

    if entity_type == "dataset":
        row = db.execute(
            select(Dataset, Project.name)
            .join(Project, Project.id == Dataset.project_id)
            .where(Dataset.id == entity_id)
        ).one_or_none()
        if row is None:
            raise StorageEntityNotFoundError("存储对象不存在或已移入回收站")
        dataset, project_name = row
        item = _dataset_item(db, settings, dataset, project_name)
        versions = _active_versions(db, dataset.id)
        class_names = _class_names_for_dataset(db, dataset.id)
        related_runs = _related_runs(db, settings, [version.id for version in versions])
    else:
        row = db.execute(
            select(DatasetVersion, Project.name)
            .join(Project, Project.id == DatasetVersion.project_id)
            .where(
                DatasetVersion.id == entity_id,
                active_entity_predicate("dataset", DatasetVersion.dataset_id),
            )
        ).one_or_none()
        if row is None:
            raise StorageEntityNotFoundError("存储对象不存在或已移入回收站")
        version, project_name = row
        item = _version_item(db, settings, version, project_name)
        class_names = _class_names_for_version(db, version)
        related_runs = _related_runs(db, settings, [version.id])

    return StorageItemDetail(
        **item.model_dump(),
        class_names=class_names,
        related_runs=related_runs,
    )


def _load_trash_entity(
    db: Session,
    entity_type: str,
    entity_id: int,
) -> Dataset | DatasetVersion | TrainingRun:
    model_by_type = {
        "dataset": Dataset,
        "dataset_version": DatasetVersion,
        "training_run": TrainingRun,
    }
    model = model_by_type.get(entity_type)
    if model is None:
        raise StorageEntityNotFoundError("不支持的存储对象类型")
    entity = db.get(model, entity_id)
    if entity is None:
        raise StorageEntityNotFoundError("存储对象不存在")
    return entity


def _trash_context(
    db: Session,
    entity_type: str,
    entity: Dataset | DatasetVersion | TrainingRun,
) -> tuple[int, int | None, int | None, str, list[StorageBlocker]]:
    if entity_type == "dataset" and isinstance(entity, Dataset):
        versions = _active_versions(db, entity.id)
        blockers = [
            StorageBlocker(
                entity_type="dataset_version",
                entity_id=version.id,
                display_name=version.name,
            )
            for version in versions
        ]
        return entity.project_id, entity.id, None, entity.name, blockers

    if entity_type == "dataset_version" and isinstance(entity, DatasetVersion):
        runs = _active_runs(db, [entity.id])
        blockers = [
            StorageBlocker(
                entity_type="training_run",
                entity_id=run.id,
                display_name=f"训练任务 #{run.id}",
                status=run.status,
            )
            for run in runs
        ]
        return entity.project_id, entity.dataset_id, entity.id, entity.name, blockers

    if entity_type == "training_run" and isinstance(entity, TrainingRun):
        version = db.get(DatasetVersion, entity.version_id)
        if version is None:
            raise StorageEntityNotFoundError("训练任务关联的标注版本不存在")
        blockers = []
        if entity.status not in ENDED_RUN_STATUSES:
            blockers.append(
                StorageBlocker(
                    entity_type="training_run",
                    entity_id=entity.id,
                    display_name=f"训练任务 #{entity.id}",
                    status=entity.status,
                )
            )
        active_prediction_jobs = db.scalars(
            select(PredictionJob).where(
                PredictionJob.run_id == entity.id,
                PredictionJob.status.not_in(ENDED_CHILD_JOB_STATUSES),
            )
        ).all()
        blockers.extend(
            StorageBlocker(
                entity_type="prediction_job",
                entity_id=job.id,
                display_name=f"预测任务 #{job.id}",
                status=job.status,
            )
            for job in active_prediction_jobs
        )
        active_exports = db.scalars(
            select(ExportArtifact).where(
                ExportArtifact.run_id == entity.id,
                ExportArtifact.status.not_in(ENDED_CHILD_JOB_STATUSES),
            )
        ).all()
        blockers.extend(
            StorageBlocker(
                entity_type="export_artifact",
                entity_id=artifact.id,
                display_name=f"模型导出 #{artifact.id}",
                status=artifact.status,
            )
            for artifact in active_exports
        )
        return (
            entity.project_id,
            version.dataset_id,
            version.id,
            f"训练任务 #{entity.id}",
            blockers,
        )

    raise StorageEntityNotFoundError("不支持的存储对象类型")


def _raise_for_blockers(
    entity_type: str,
    blockers: list[StorageBlocker],
) -> None:
    if not blockers:
        return
    messages = {
        "dataset": "该数据集仍有关联的标注版本，请先将关联版本移入回收站。",
        "dataset_version": "该标注版本仍有关联的训练任务，请先将关联训练任务移入回收站。",
        "training_run": "训练任务仍在运行中，无法移入回收站。",
    }
    message = messages[entity_type]
    if entity_type == "training_run" and any(
        blocker.entity_type in {"prediction_job", "export_artifact"}
        for blocker in blockers
    ):
        message = "训练任务仍有关联任务正在处理中，无法移入回收站。"
    raise StorageConflictError(message, blockers)


def _trash_summary(
    db: Session,
    entity_type: str,
    entity: Dataset | DatasetVersion | TrainingRun,
) -> dict[str, Any]:
    if entity_type == "dataset" and isinstance(entity, Dataset):
        image_count, annotation_count = _dataset_counts(db, entity.id)
        return {
            "name": entity.name,
            "image_count": image_count,
            "annotation_count": annotation_count,
            "class_names": _class_names_for_dataset(db, entity.id),
        }
    if entity_type == "dataset_version" and isinstance(entity, DatasetVersion):
        image_count, annotation_count, split_counts = _version_counts(entity)
        return {
            "name": entity.name,
            "image_count": image_count,
            "annotation_count": annotation_count,
            "split_counts": split_counts,
            "class_names": _class_names_for_version(db, entity),
        }
    if entity_type == "training_run" and isinstance(entity, TrainingRun):
        prediction_count = db.scalar(
            select(func.count(PredictionJob.id)).where(PredictionJob.run_id == entity.id)
        ) or 0
        export_count = db.scalar(
            select(func.count(ExportArtifact.id)).where(ExportArtifact.run_id == entity.id)
        ) or 0
        return {
            "status": entity.status,
            "model": str((entity.config or {}).get("model") or ""),
            "prediction_job_count": int(prediction_count),
            "export_count": int(export_count),
        }
    raise StorageEntityNotFoundError("不支持的存储对象类型")


def move_storage_item_to_trash(
    db: Session,
    settings: Settings,
    entity_type: str,
    entity_id: int,
) -> TrashItemRead:
    if entity_type not in TRASH_ENTITY_TYPES:
        raise StorageEntityNotFoundError("不支持的存储对象类型")

    duplicate = db.scalar(
        select(TrashItem).where(
            TrashItem.entity_type == entity_type,
            TrashItem.entity_id == entity_id,
        )
    )
    if duplicate is not None:
        raise StorageConflictError("该对象已在回收站中。")

    entity = _load_trash_entity(db, entity_type, entity_id)
    project_id, dataset_id, version_id, display_name, blockers = _trash_context(
        db, entity_type, entity
    )
    _raise_for_blockers(entity_type, blockers)

    source_path = expected_entity_path(settings, entity_type, entity)
    if not source_path.is_dir():
        raise StoragePathError("数据目录不存在")
    size_bytes = directory_size(source_path, settings.workspace_root)
    summary = _trash_summary(db, entity_type, entity)
    now = datetime.now(UTC)

    trash = TrashItem(
        entity_type=entity_type,
        entity_id=entity_id,
        project_id=project_id,
        dataset_id=dataset_id,
        version_id=version_id,
        display_name=display_name,
        original_path=str(source_path),
        trash_path="pending",
        size_bytes=size_bytes,
        summary=summary,
        status="pending_move",
        deleted_at=now,
        purge_after=now + timedelta(days=TRASH_RETENTION_DAYS),
    )
    db.add(trash)
    try:
        db.flush()
        trash_root = _canonical_path(settings.workspace_root, ".trash")
        destination = trash_root / f"{trash.id}-{entity_type}-{entity_id}"
        trash.trash_path = str(destination)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise StorageConflictError("该对象已在回收站中。") from exc
    db.refresh(trash)

    try:
        trash_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise OSError("回收站目标目录已存在")
        source_path.rename(destination)
    except OSError as exc:
        if source_path.exists():
            db.delete(trash)
            db.commit()
        raise StorageMoveError(f"移动数据到回收站失败：{exc}") from exc

    trash.status = "active"
    try:
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise StorageMoveError("数据已移动，但回收站状态更新失败") from exc
    db.refresh(trash)
    return TrashItemRead.model_validate(trash)


def _trash_root(settings: Settings) -> Path:
    return _canonical_path(settings.workspace_root, ".trash")


def _expected_trash_path(settings: Settings, item: TrashItem) -> Path:
    return _trash_root(settings) / f"{item.id}-{item.entity_type}-{item.entity_id}"


def _validate_trash_item_paths(
    db: Session,
    settings: Settings,
    item: TrashItem,
) -> tuple[Path, Path]:
    entity = _load_trash_entity(db, item.entity_type, item.entity_id)
    source = expected_entity_path(settings, item.entity_type, entity)
    _validate_stored_artifact_path(item.original_path, source, settings.workspace_root)

    expected_trash = _expected_trash_path(settings, item)
    _validate_stored_artifact_path(item.trash_path, expected_trash, _trash_root(settings))
    if not expected_trash.resolve().is_relative_to(_trash_root(settings).resolve()):
        raise StoragePathError("回收站路径不在规范目录内")
    return source, expected_trash


def list_trash_items(db: Session) -> TrashItemList:
    items = list(
        db.scalars(
            select(TrashItem)
            .order_by(TrashItem.deleted_at.desc(), TrashItem.id.desc())
        ).all()
    )
    payload = [TrashItemRead.model_validate(item) for item in items]
    return TrashItemList(
        items=payload,
        total_size_bytes=sum(item.size_bytes for item in payload),
    )


def _require_restore_parent(db: Session, item: TrashItem) -> None:
    if item.entity_type == "training_run":
        if item.version_id is None or db.get(DatasetVersion, item.version_id) is None:
            raise StorageConflictError("训练任务关联的标注版本不存在，无法恢复。")
        if is_entity_trashed(db, "dataset_version", item.version_id):
            parent = db.scalar(
                select(TrashItem).where(
                    TrashItem.entity_type == "dataset_version",
                    TrashItem.entity_id == item.version_id,
                )
            )
            name = parent.display_name if parent is not None else f"版本 #{item.version_id}"
            raise StorageConflictError(f"请先恢复标注版本“{name}”。")
    elif item.entity_type == "dataset_version":
        if item.dataset_id is None or db.get(Dataset, item.dataset_id) is None:
            raise StorageConflictError("标注版本关联的原始数据集不存在，无法恢复。")
        if is_entity_trashed(db, "dataset", item.dataset_id):
            parent = db.scalar(
                select(TrashItem).where(
                    TrashItem.entity_type == "dataset",
                    TrashItem.entity_id == item.dataset_id,
                )
            )
            name = parent.display_name if parent is not None else f"数据集 #{item.dataset_id}"
            raise StorageConflictError(f"请先恢复原始数据集“{name}”。")


def restore_trash_item(
    db: Session,
    settings: Settings,
    trash_id: int,
) -> StorageMutationResponse:
    item = db.get(TrashItem, trash_id)
    if item is None:
        raise StorageEntityNotFoundError("回收站记录不存在")
    if item.status != "active":
        raise StorageConflictError("该回收站记录当前无法恢复，请先刷新回收站状态。")
    _require_restore_parent(db, item)
    source, trash = _validate_trash_item_paths(db, settings, item)
    if source.exists():
        raise StorageConflictError("原目录已存在，无法恢复以免覆盖现有数据。")
    if not trash.is_dir():
        raise StoragePathError("回收站数据目录不存在")

    item.status = "pending_restore"
    item.error_message = None
    db.commit()
    try:
        source.parent.mkdir(parents=True, exist_ok=True)
        trash.rename(source)
    except OSError as exc:
        item.status = "active"
        item.error_message = str(exc)
        db.commit()
        raise StorageMoveError(f"恢复数据目录失败：{exc}") from exc

    response = StorageMutationResponse(
        trash_id=item.id,
        entity_type=item.entity_type,
        entity_id=item.entity_id,
        status="restored",
        message="数据已恢复。",
    )
    try:
        db.delete(item)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise StorageMoveError("目录已恢复，数据库状态将在下次维护时自动修复。") from exc
    return response


def _purging_path(settings: Settings, trash_id: int) -> Path:
    return _trash_root(settings) / ".purging" / str(trash_id)


def _cascade_entity_model(entity_type: str):
    return {
        "dataset": Dataset,
        "dataset_version": DatasetVersion,
        "training_run": TrainingRun,
    }.get(entity_type)


def _reconcile_abandoned_cascade(
    db: Session,
    settings: Settings,
    staged: Path,
    payload: dict[str, Any],
) -> tuple[int, int]:
    parent_trash_id = payload.get("parent_trash_id")
    entries = payload.get("entries")
    if not isinstance(parent_trash_id, int) or not isinstance(entries, list):
        return 0, 0

    parent_item = db.get(TrashItem, parent_trash_id)
    if parent_item is None:
        database_rows_missing = True
        for entry in entries:
            if not isinstance(entry, dict):
                database_rows_missing = False
                break
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            model = (
                _cascade_entity_model(entity_type)
                if isinstance(entity_type, str)
                else None
            )
            if model is None or not isinstance(entity_id, int):
                database_rows_missing = False
                break
            if db.get(model, entity_id) is not None:
                database_rows_missing = False
                break
            trash_id = entry.get("trash_id")
            if isinstance(trash_id, int) and db.get(TrashItem, trash_id) is not None:
                database_rows_missing = False
                break
        if database_rows_missing:
            shutil.rmtree(staged, ignore_errors=True)
            return 1, 0
        return 0, 1

    try:
        if parent_item.entity_type != "dataset":
            raise StoragePathError("级联删除暂存标记不是数据集记录")
        dataset = db.get(Dataset, parent_item.entity_id)
        if dataset is None:
            raise StoragePathError("级联删除数据库状态不完整，需要人工检查")
        for entry in entries:
            if not isinstance(entry, dict):
                raise StoragePathError("级联删除暂存标记格式无效")
            entity_type = entry.get("entity_type")
            entity_id = entry.get("entity_id")
            if not isinstance(entity_type, str) or not isinstance(entity_id, int):
                raise StoragePathError("级联删除暂存标记格式无效")
            model = _cascade_entity_model(entity_type)
            if model is None or db.get(model, entity_id) is None:
                raise StoragePathError("级联删除数据库状态不完整，需要人工检查")
            trash_id = entry.get("trash_id")
            item = db.get(TrashItem, trash_id) if isinstance(trash_id, int) else None
            if isinstance(trash_id, int) and item is None:
                raise StoragePathError("级联删除关联回收站记录缺失，需要人工检查")
            if item is not None and (
                item.entity_type != entity_type or item.entity_id != entity_id
            ):
                raise StoragePathError("级联删除回收站记录不匹配，需要人工检查")
            target = Path(str(entry.get("source_path")))
            expected = (
                _expected_trash_path(settings, item)
                if item is not None
                else expected_entity_path(settings, entity_type, db.get(model, entity_id))
            )
            _validate_stored_artifact_path(
                str(target), expected, settings.workspace_root
            )
            staged_path = Path(str(entry.get("staged_path")))
            require_workspace_path(staged_path, settings.workspace_root)
            if staged_path.resolve().parent != staged.resolve():
                raise StoragePathError("级联删除暂存目录不在规范目录内")
            if target.exists():
                raise StoragePathError(f"级联删除恢复目标已存在：{target}")
        _restore_cascade_staged(staged, entries)
        return 1, 0
    except Exception as exc:
        db.rollback()
        current = db.get(TrashItem, parent_trash_id)
        if current is not None:
            try:
                _mark_reconcile_error(db, current, str(exc))
            except SQLAlchemyError:
                db.rollback()
        return 0, 1


def _delete_training_run_rows(db: Session, run_id: int) -> None:
    db.execute(delete(Prediction).where(Prediction.run_id == run_id))
    db.execute(delete(PredictionJob).where(PredictionJob.run_id == run_id))
    db.execute(delete(ExportArtifact).where(ExportArtifact.run_id == run_id))
    db.execute(delete(RunMetric).where(RunMetric.run_id == run_id))
    db.execute(delete(TrainingRun).where(TrainingRun.id == run_id))


def _delete_entity_rows(db: Session, item: TrashItem) -> None:
    if item.entity_type == "training_run":
        _delete_training_run_rows(db, item.entity_id)
        return
    if item.entity_type == "dataset_version":
        run_count = db.scalar(
            select(func.count(TrainingRun.id)).where(
                TrainingRun.version_id == item.entity_id
            )
        ) or 0
        if run_count:
            raise StorageConflictError("该标注版本仍有关联训练任务，无法彻底删除。")
        db.execute(delete(DatasetVersion).where(DatasetVersion.id == item.entity_id))
        return
    if item.entity_type == "dataset":
        version_count = db.scalar(
            select(func.count(DatasetVersion.id)).where(
                DatasetVersion.dataset_id == item.entity_id
            )
        ) or 0
        if version_count:
            raise StorageConflictError(
                "该数据集仍有关联标注版本，无法单独彻底删除，请使用“级联彻底删除”。"
            )
        image_ids = select(Image.id).where(Image.dataset_id == item.entity_id)
        db.execute(delete(Annotation).where(Annotation.image_id.in_(image_ids)))
        db.execute(delete(Image).where(Image.dataset_id == item.entity_id))
        db.execute(delete(Dataset).where(Dataset.id == item.entity_id))
        return
    raise StorageEntityNotFoundError("不支持的存储对象类型")


def _cascade_entities(
    db: Session,
    dataset_id: int,
) -> tuple[Dataset, list[DatasetVersion], list[TrainingRun]]:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise StorageEntityNotFoundError("数据集记录不存在")
    versions = list(
        db.scalars(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.id)
        ).all()
    )
    version_ids = [version.id for version in versions]
    runs = list(
        db.scalars(
            select(TrainingRun)
            .where(TrainingRun.version_id.in_(version_ids))
            .order_by(TrainingRun.id)
        ).all()
    ) if version_ids else []
    return dataset, versions, runs


def _cascade_trash_items(
    db: Session,
    dataset: Dataset,
    versions: list[DatasetVersion],
    runs: list[TrainingRun],
) -> dict[tuple[str, int], TrashItem]:
    keys = [("dataset", dataset.id)]
    keys.extend(("dataset_version", version.id) for version in versions)
    keys.extend(("training_run", run.id) for run in runs)
    if not keys:
        return {}
    predicates = [
        (TrashItem.entity_type == entity_type) & (TrashItem.entity_id == entity_id)
        for entity_type, entity_id in keys
    ]
    items = list(db.scalars(select(TrashItem).where(or_(*predicates))).all())
    return {(item.entity_type, item.entity_id): item for item in items}


def _validate_cascade_state(
    db: Session,
    settings: Settings,
    dataset: Dataset,
    versions: list[DatasetVersion],
    runs: list[TrainingRun],
    trash_items: dict[tuple[str, int], TrashItem],
) -> list[tuple[str, int, Path, TrashItem | None]]:
    parent_item = trash_items.get(("dataset", dataset.id))
    if parent_item is None:
        raise StorageConflictError("该数据集不在回收站中，请先移入回收站。")
    if parent_item.status not in {"active", "error"}:
        raise StorageConflictError("该数据集回收站记录正在处理中，暂时无法级联删除。")

    active_runs = [run for run in runs if run.status not in ENDED_RUN_STATUSES]
    active_jobs = list(
        db.scalars(
            select(PredictionJob).where(
                PredictionJob.run_id.in_([run.id for run in runs]),
                PredictionJob.status.not_in(ENDED_CHILD_JOB_STATUSES),
            )
        ).all()
    ) if runs else []
    active_exports = list(
        db.scalars(
            select(ExportArtifact).where(
                ExportArtifact.run_id.in_([run.id for run in runs]),
                ExportArtifact.status.not_in(ENDED_CHILD_JOB_STATUSES),
            )
        ).all()
    ) if runs else []
    blockers = [
        StorageBlocker(
            entity_type="training_run",
            entity_id=run.id,
            display_name=f"训练任务 #{run.id}",
            status=run.status,
        )
        for run in active_runs
    ]
    blockers.extend(
        StorageBlocker(
            entity_type="prediction_job",
            entity_id=job.id,
            display_name=f"预测任务 #{job.id}",
            status=job.status,
        )
        for job in active_jobs
    )
    blockers.extend(
        StorageBlocker(
            entity_type="export_artifact",
            entity_id=artifact.id,
            display_name=f"模型导出 #{artifact.id}",
            status=artifact.status,
        )
        for artifact in active_exports
    )
    if blockers:
        raise StorageConflictError(
            "数据集仍有关联任务正在处理，无法级联彻底删除。",
            blockers,
        )

    entities: list[tuple[str, int, Dataset | DatasetVersion | TrainingRun]] = [
        ("dataset", dataset.id, dataset)
    ]
    entities.extend(
        ("dataset_version", version.id, version) for version in versions
    )
    entities.extend(("training_run", run.id, run) for run in runs)
    paths: list[tuple[str, int, Path, TrashItem | None]] = []
    for entity_type, entity_id, entity in entities:
        item = trash_items.get((entity_type, entity_id))
        if item is not None:
            if item.status not in {"active", "error"}:
                raise StorageConflictError(
                    f"{item.display_name} 的回收站记录正在处理中，暂时无法级联删除。"
                )
            source, trash = _validate_trash_item_paths(db, settings, item)
            if source.exists():
                raise StorageConflictError(f"{item.display_name} 的原目录仍然存在。")
            if not trash.is_dir():
                raise StoragePathError(f"{item.display_name} 的回收站数据目录不存在。")
            paths.append((entity_type, entity_id, trash, item))
            continue

        source = expected_entity_path(settings, entity_type, entity)
        if not source.is_dir():
            raise StoragePathError(f"{entity_type} #{entity_id} 的数据目录不存在。")
        paths.append((entity_type, entity_id, source, None))
    return paths


def _cascade_marker_payload(
    parent_trash_id: int,
    entries: list[tuple[str, int, Path, TrashItem | None]],
    staged: Path,
) -> dict[str, Any]:
    return {
        "kind": "cascade",
        "parent_trash_id": parent_trash_id,
        "entries": [
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "trash_id": item.id if item is not None else None,
                "source_path": str(path),
                "staged_path": str(staged / f"{entity_type}-{entity_id}"),
            }
            for entity_type, entity_id, path, item in entries
        ],
    }


def _restore_cascade_staged(
    staged: Path,
    entries: list[dict[str, Any]],
) -> None:
    restored: list[tuple[Path, Path]] = []
    try:
        for entry in reversed(entries):
            current = Path(str(entry["staged_path"]))
            target = Path(str(entry["source_path"]))
            if not current.exists():
                continue
            if target.exists():
                raise OSError(f"恢复目标已存在：{target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            current.rename(target)
            restored.append((target, current))
    except OSError:
        for target, current in reversed(restored):
            if target.exists() and not current.exists():
                target.rename(current)
        raise
    shutil.rmtree(staged, ignore_errors=True)


def cascade_purge_trash_item(
    db: Session,
    settings: Settings,
    trash_id: int,
    confirm_name: str,
) -> StorageMutationResponse:
    parent_item = db.get(TrashItem, trash_id)
    if parent_item is None:
        raise StorageEntityNotFoundError("回收站记录不存在")
    if parent_item.entity_type != "dataset":
        raise StorageConflictError("级联彻底删除只能用于原始数据集。")
    if confirm_name != parent_item.display_name:
        raise StorageConfirmationError("确认名称不匹配，未执行级联彻底删除。")
    if parent_item.status not in {"active", "error"}:
        raise StorageConflictError("该回收站记录正在处理中，暂时无法级联彻底删除。")

    dataset, versions, runs = _cascade_entities(db, parent_item.entity_id)
    trash_items = _cascade_trash_items(db, dataset, versions, runs)
    entries = _validate_cascade_state(
        db, settings, dataset, versions, runs, trash_items
    )
    staged = _purging_path(settings, parent_item.id)
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        raise StorageConflictError("该数据集已有未完成的级联删除暂存目录。")

    marker_payload = _cascade_marker_payload(parent_item.id, entries, staged)
    try:
        staged.mkdir()
        (staged / ".purge.json").write_text(
            json.dumps(marker_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        for entity_type, entity_id, path, _ in entries:
            destination = staged / f"{entity_type}-{entity_id}"
            if destination.exists():
                raise OSError(f"级联删除暂存目标已存在：{destination}")
            path.rename(destination)
    except OSError as exc:
        try:
            _restore_cascade_staged(staged, marker_payload["entries"])
        except OSError as restore_exc:
            raise StorageMoveError(
                f"级联删除暂存失败，且原目录恢复失败：{restore_exc}"
            ) from exc
        raise StorageMoveError(f"暂存级联删除数据失败：{exc}") from exc

    response = StorageMutationResponse(
        trash_id=parent_item.id,
        entity_type="dataset",
        entity_id=dataset.id,
        status="purged",
        message="数据集及其全部关联数据已彻底删除。",
    )
    try:
        for run in runs:
            _delete_training_run_rows(db, run.id)
        for version in versions:
            db.execute(delete(DatasetVersion).where(DatasetVersion.id == version.id))
        image_ids = select(Image.id).where(Image.dataset_id == dataset.id)
        db.execute(delete(Annotation).where(Annotation.image_id.in_(image_ids)))
        db.execute(delete(Image).where(Image.dataset_id == dataset.id))
        db.execute(delete(Dataset).where(Dataset.id == dataset.id))
        for item in trash_items.values():
            db.delete(item)
        db.commit()
    except Exception as exc:
        db.rollback()
        try:
            _restore_cascade_staged(staged, marker_payload["entries"])
        except OSError as restore_exc:
            raise StorageMoveError(
                f"级联删除数据库失败，且暂存目录恢复失败：{restore_exc}"
            ) from exc
        if isinstance(exc, StorageConflictError):
            raise
        raise StorageMoveError("级联删除数据库失败，数据已恢复到原回收站目录。") from exc

    shutil.rmtree(staged, ignore_errors=True)
    return response


def purge_trash_item(
    db: Session,
    settings: Settings,
    trash_id: int,
    confirm_name: str,
) -> StorageMutationResponse:
    item = db.get(TrashItem, trash_id)
    if item is None:
        raise StorageEntityNotFoundError("回收站记录不存在")
    if confirm_name != item.display_name:
        raise StorageConfirmationError("确认名称不匹配，未执行彻底删除。")
    if item.status not in {"active", "error"}:
        raise StorageConflictError("该回收站记录正在处理中，暂时无法彻底删除。")
    source, trash = _validate_trash_item_paths(db, settings, item)
    if source.exists():
        raise StorageConflictError("原目录仍然存在，无法确认可彻底删除。")
    if not trash.is_dir():
        raise StoragePathError("回收站数据目录不存在")

    staged = _purging_path(settings, item.id)
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        raise StorageConflictError("该对象已有未完成的彻底删除暂存目录。")
    trash_marker = trash / ".purge.json"
    staged_marker = staged / ".purge.json"
    try:
        trash_marker.write_text(
            json.dumps({"entity_type": item.entity_type, "entity_id": item.entity_id}),
            encoding="utf-8",
        )
        trash.rename(staged)
    except OSError as exc:
        trash_marker.unlink(missing_ok=True)
        staged_marker.unlink(missing_ok=True)
        if staged.exists() and not trash.exists():
            staged.rename(trash)
        raise StorageMoveError(f"暂存待删除数据失败：{exc}") from exc

    response = StorageMutationResponse(
        trash_id=item.id,
        entity_type=item.entity_type,
        entity_id=item.entity_id,
        status="purged",
        message="数据已彻底删除。",
    )
    try:
        _delete_entity_rows(db, item)
        db.delete(item)
        db.commit()
    except Exception as exc:
        db.rollback()
        try:
            staged_marker.unlink(missing_ok=True)
            staged.rename(trash)
        except OSError as restore_exc:
            raise StorageMoveError(
                f"数据库删除失败，且暂存目录恢复失败：{restore_exc}"
            ) from exc
        if isinstance(exc, StorageConflictError):
            raise
        raise StorageMoveError("数据库删除失败，数据已恢复到回收站。") from exc

    shutil.rmtree(staged, ignore_errors=True)
    return response


def purge_expired_trash(
    db: Session,
    settings: Settings,
    now: datetime | None = None,
) -> PurgeSummary:
    current = now or datetime.now(UTC)
    priority = {"training_run": 0, "dataset_version": 1, "dataset": 2}
    expired = list(
        db.scalars(
            select(TrashItem).where(TrashItem.purge_after <= current)
        ).all()
    )
    expired.sort(key=lambda item: (priority.get(item.entity_type, 99), item.id))
    purged = 0
    failed = 0
    for candidate in expired:
        item_id = candidate.id
        display_name = candidate.display_name
        try:
            purge_trash_item(db, settings, item_id, display_name)
            purged += 1
        except Exception as exc:
            db.rollback()
            failed += 1
            item = db.get(TrashItem, item_id)
            if item is not None:
                item.status = "error"
                item.error_message = str(exc)
                try:
                    db.commit()
                except SQLAlchemyError:
                    db.rollback()
    return PurgeSummary(purged_count=purged, failed_count=failed)


def _mark_reconcile_error(db: Session, item: TrashItem, message: str) -> None:
    item.status = "error"
    item.error_message = message
    db.commit()


def _cleanup_abandoned_purging(
    db: Session,
    settings: Settings,
) -> tuple[int, int]:
    purging_root = _purging_path(settings, 0).parent
    if not purging_root.is_dir():
        return 0, 0
    reconciled = 0
    errors = 0
    for staged in purging_root.iterdir():
        if not staged.is_dir() or not staged.name.isdigit():
            continue
        trash_id = int(staged.name)
        marker = staged / ".purge.json"
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("kind") == "cascade":
            cascade_reconciled, cascade_errors = _reconcile_abandoned_cascade(
                db, settings, staged, payload
            )
            reconciled += cascade_reconciled
            errors += cascade_errors
            continue
        model = {
            "training_run": TrainingRun,
            "dataset_version": DatasetVersion,
            "dataset": Dataset,
        }.get(payload.get("entity_type"))
        entity_id = payload.get("entity_id")
        if model is None or not isinstance(entity_id, int):
            continue

        item = db.get(TrashItem, trash_id)
        entity = db.get(model, entity_id)
        if item is None:
            if entity is None:
                shutil.rmtree(staged, ignore_errors=True)
                reconciled += 1
            continue

        try:
            if item.entity_type != payload["entity_type"] or item.entity_id != entity_id:
                _mark_reconcile_error(
                    db,
                    item,
                    "彻底删除暂存标记与回收站记录不一致，需要人工检查。",
                )
                errors += 1
                continue
            if entity is None:
                _mark_reconcile_error(
                    db,
                    item,
                    "回收站记录仍存在，但源数据记录缺失，需要人工检查。",
                )
                errors += 1
                continue

            source, trash = _validate_trash_item_paths(db, settings, item)
            conflicts = []
            if trash.exists():
                conflicts.append("规范回收站目录")
            if source.exists():
                conflicts.append("原目录")
            if conflicts:
                _mark_reconcile_error(
                    db,
                    item,
                    f"彻底删除恢复发生路径冲突：{'、'.join(conflicts)}已存在，需要人工检查。",
                )
                errors += 1
                continue

            trash.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(trash)
            (trash / ".purge.json").unlink(missing_ok=True)
            item.status = "active"
            item.error_message = None
            db.commit()
            reconciled += 1
        except Exception as exc:
            db.rollback()
            current = db.get(TrashItem, trash_id)
            if current is not None:
                try:
                    _mark_reconcile_error(db, current, str(exc))
                except SQLAlchemyError:
                    db.rollback()
            errors += 1

    return reconciled, errors


def reconcile_trash(db: Session, settings: Settings) -> ReconcileSummary:
    reconciled = 0
    errors = 0
    items = list(
        db.scalars(
            select(TrashItem).where(
                TrashItem.status.in_({"pending_move", "pending_restore"})
            )
        ).all()
    )
    for item in items:
        try:
            source, trash = _validate_trash_item_paths(db, settings, item)
            source_exists = source.exists()
            trash_exists = trash.exists()
            if source_exists == trash_exists:
                _mark_reconcile_error(
                    db,
                    item,
                    "原目录与回收站目录同时存在或同时缺失，需要人工检查。",
                )
                errors += 1
                continue
            if item.status == "pending_move":
                if source_exists:
                    db.delete(item)
                else:
                    item.status = "active"
                    item.error_message = None
            else:
                if source_exists:
                    db.delete(item)
                else:
                    item.status = "active"
                    item.error_message = None
            db.commit()
            reconciled += 1
        except Exception as exc:
            db.rollback()
            current = db.get(TrashItem, item.id)
            if current is not None:
                try:
                    _mark_reconcile_error(db, current, str(exc))
                except SQLAlchemyError:
                    db.rollback()
            errors += 1
    purging_reconciled, purging_errors = _cleanup_abandoned_purging(db, settings)
    reconciled += purging_reconciled
    errors += purging_errors
    return ReconcileSummary(reconciled_count=reconciled, error_count=errors)
