from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
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
    PredictionJob,
    Project,
    TrashItem,
    TrainingRun,
)
from app.storage.schemas import (
    RelatedRunRead,
    StorageBlocker,
    StorageItemDetail,
    StorageItemList,
    StorageItemRead,
    TrashItemRead,
)
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    require_active_entity,
)


ACTIVE_RUN_STATUSES = {"queued", "preparing", "running"}
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
        if entity.status in ACTIVE_RUN_STATUSES:
            blockers.append(
                StorageBlocker(
                    entity_type="training_run",
                    entity_id=entity.id,
                    display_name=f"训练任务 #{entity.id}",
                    status=entity.status,
                )
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
    raise StorageConflictError(messages[entity_type], blockers)


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
