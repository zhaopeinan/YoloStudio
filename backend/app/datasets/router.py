from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.datasets.importer import import_dataset
from app.datasets.scanner import scan_dataset_zip
from app.datasets.schemas import (
    DatasetImageList,
    DatasetImageRead,
    DatasetImportRequest,
    DatasetImportSummary,
    DatasetScanRequest,
    DatasetScanSummary,
)
from app.db.models import Annotation, Dataset, Image
from app.db.session import get_db


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/scan", response_model=DatasetScanSummary)
def scan_dataset(request: DatasetScanRequest) -> DatasetScanSummary:
    try:
        return scan_dataset_zip(request.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset archive was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/import", response_model=DatasetImportSummary)
def import_dataset_endpoint(
    request: DatasetImportRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DatasetImportSummary:
    try:
        return import_dataset(db, settings, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset archive was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_id}/images", response_model=DatasetImageList)
def list_dataset_images(
    dataset_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DatasetImageList:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")

    total = db.scalar(select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id)) or 0
    annotation_counts = (
        select(Annotation.image_id, func.count(Annotation.id).label("annotation_count"))
        .group_by(Annotation.image_id)
        .subquery()
    )
    rows = db.execute(
        select(Image, func.coalesce(annotation_counts.c.annotation_count, 0))
        .outerjoin(annotation_counts, Image.id == annotation_counts.c.image_id)
        .where(Image.dataset_id == dataset_id)
        .order_by(Image.id)
        .limit(limit)
        .offset(offset)
    ).all()

    return DatasetImageList(
        items=[
            DatasetImageRead(
                id=image.id,
                relative_path=image.relative_path,
                platform=image.platform,
                altitude=image.altitude,
                timestamp=image.timestamp,
                annotation_count=int(annotation_count),
                image_url=f"/api/images/{image.id}/file",
            )
            for image, annotation_count in rows
        ],
        limit=limit,
        offset=offset,
        total=total,
    )


images_router = APIRouter(prefix="/api/images", tags=["images"])


@images_router.get("/{image_id}/file")
def get_image_file(
    image_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image was not found")

    file_path = (settings.workspace_root / image.relative_path).resolve()
    workspace_root = settings.workspace_root.resolve()
    if not file_path.is_relative_to(workspace_root) or not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file was not found")

    return FileResponse(Path(file_path))
