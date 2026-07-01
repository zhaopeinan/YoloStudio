from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.models import Dataset, DatasetVersion
from app.db.session import get_db
from app.versions.exporter import VersionExportError, create_dataset_version
from app.versions.schemas import DatasetVersionCreate, DatasetVersionList, DatasetVersionRead


router = APIRouter(prefix="/api/datasets", tags=["versions"])


def _read_version(version: DatasetVersion) -> DatasetVersionRead:
    split_counts = version.split_manifest.get("split_counts", {})
    return DatasetVersionRead(
        id=version.id,
        project_id=version.project_id,
        dataset_id=version.dataset_id,
        name=version.name,
        class_mapping=version.class_mapping,
        split_counts={
            "train": int(split_counts.get("train", 0)),
            "val": int(split_counts.get("val", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        artifact_path=version.artifact_path,
        frozen=bool(version.frozen),
        created_at=version.created_at,
    )


@router.post("/{dataset_id}/versions", response_model=DatasetVersionRead)
def create_version(
    dataset_id: int,
    request: DatasetVersionCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetVersionRead:
    try:
        version = create_dataset_version(db, settings, dataset_id, request.name, request.class_ids)
    except VersionExportError as exc:
        message = str(exc)
        status_code = 404 if message == "Dataset was not found" else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    return _read_version(version)


@router.get("/{dataset_id}/versions", response_model=DatasetVersionList)
def list_versions(dataset_id: int, db: Session = Depends(get_db)) -> DatasetVersionList:
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")

    versions = db.scalars(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.id.desc())
    ).all()
    return DatasetVersionList(items=[_read_version(version) for version in versions])
