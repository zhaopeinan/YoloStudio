from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
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
    TrainingRun,
)
from app.storage.schemas import (
    RelatedRunRead,
    StorageBlocker,
    StorageItemDetail,
    StorageItemList,
    StorageItemRead,
)
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    require_active_entity,
)


ACTIVE_RUN_STATUSES = {"queued", "preparing", "running"}
TRASH_RETENTION_DAYS = 30
CATALOG_ENTITY_TYPES = {"dataset", "dataset_version"}


class StoragePathError(ValueError):
    pass


def _workspace_path(settings: Settings, value: str | Path) -> Path:
    workspace_root = settings.workspace_root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = workspace_root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise StoragePathError("存储路径不在工作区内")
    return resolved


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
    return _workspace_path(
        settings,
        Path("projects") / str(dataset.project_id) / "datasets" / str(dataset.id),
    )


def _version_path(settings: Settings, version: DatasetVersion) -> Path:
    return _workspace_path(settings, version.artifact_path)


def _run_path(settings: Settings, run: TrainingRun) -> Path:
    return _workspace_path(settings, run.artifact_path)


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
