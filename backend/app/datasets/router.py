from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import exists, func, select, text
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.datasets.importer import import_dataset
from app.datasets.scanner import scan_dataset_source
from app.datasets.schemas import (
    DatasetImageList,
    DatasetImageRead,
    DatasetImportRequest,
    DatasetImportSummary,
    DatasetScanRequest,
    DatasetScanSummary,
    ProjectDatasetRead,
    ProjectList,
    ProjectRead,
)
from app.db.models import Annotation, Dataset, Image, Project
from app.db.session import get_db


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


projects_router = APIRouter(prefix="/api/projects", tags=["projects"])


@projects_router.get("", response_model=ProjectList)
def list_projects(db: Session = Depends(get_db)) -> ProjectList:
    projects = db.scalars(select(Project).order_by(Project.id)).all()
    datasets = db.scalars(select(Dataset).order_by(Dataset.project_id, Dataset.id)).all()
    datasets_by_project: dict[int, list[Dataset]] = {}
    for dataset in datasets:
        datasets_by_project.setdefault(dataset.project_id, []).append(dataset)

    return ProjectList(
        items=[
            ProjectRead(
                id=project.id,
                name=project.name,
                datasets=[
                    ProjectDatasetRead(
                        id=dataset.id,
                        project_id=dataset.project_id,
                        name=dataset.name,
                        source_type=dataset.source_type,
                        import_status=dataset.import_status,
                        image_count=dataset.image_count,
                    )
                    for dataset in datasets_by_project.get(project.id, [])
                ],
            )
            for project in projects
        ]
    )


@router.post("/scan", response_model=DatasetScanSummary)
def scan_dataset(request: DatasetScanRequest) -> DatasetScanSummary:
    try:
        return scan_dataset_source(request.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset source was not found") from exc
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
        raise HTTPException(status_code=404, detail="Dataset source was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_id}/images", response_model=DatasetImageList)
def list_dataset_images(
    dataset_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    platform: str | None = Query(None, max_length=80),
    label_status: str = Query("all", pattern="^(all|annotated|unannotated)$"),
    class_id: int | None = Query(None, ge=1),
    edge_tag: str | None = Query(None, max_length=80),
    altitude_min: float | None = Query(None),
    altitude_max: float | None = Query(None),
    db: Session = Depends(get_db),
) -> DatasetImageList:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")

    annotation_counts = (
        select(Annotation.image_id, func.count(Annotation.id).label("annotation_count"))
        .group_by(Annotation.image_id)
        .subquery()
    )
    conditions = [Image.dataset_id == dataset_id]
    if platform:
        conditions.append(Image.platform == platform)
    if altitude_min is not None:
        conditions.append(Image.altitude >= altitude_min)
    if altitude_max is not None:
        conditions.append(Image.altitude <= altitude_max)
    if label_status == "annotated":
        conditions.append(exists().where(Annotation.image_id == Image.id))
    elif label_status == "unannotated":
        conditions.append(~exists().where(Annotation.image_id == Image.id))
    if class_id is not None:
        conditions.append(
            exists().where(Annotation.image_id == Image.id, Annotation.class_id == class_id)
        )
    if edge_tag:
        conditions.append(
            exists().where(
                Annotation.image_id == Image.id,
                text(
                    "EXISTS (SELECT 1 FROM json_each(annotations.edge_tags) "
                    "WHERE json_each.value = :edge_tag)"
                ).bindparams(edge_tag=edge_tag),
            )
        )

    total = db.scalar(select(func.count()).select_from(Image).where(*conditions)) or 0
    rows = db.execute(
        select(Image, func.coalesce(annotation_counts.c.annotation_count, 0))
        .outerjoin(annotation_counts, Image.id == annotation_counts.c.image_id)
        .where(*conditions)
        .order_by(Image.id)
        .limit(limit)
        .offset(offset)
    ).all()

    return DatasetImageList(
        items=[
            DatasetImageRead(
                id=image.id,
                relative_path=image.relative_path,
                width=image.width,
                height=image.height,
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
