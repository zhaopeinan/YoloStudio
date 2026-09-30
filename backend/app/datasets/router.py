from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.auth.deps import get_current_user, require_admin
from fastapi.responses import FileResponse
from sqlalchemy import exists, func, select, text
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.annotations.status import get_annotation_status
from app.datasets.class_detect import detect_classes_for_dataset
from app.datasets.importer import import_dataset, read_image_dimensions
from app.datasets.scanner import scan_dataset_source
from app.datasets.schemas import (
    ClassCoverageBucket,
    CoverageBucket,
    DatasetCoverageSummary,
    DatasetImageList,
    DatasetImageRead,
    DatasetDimensionRefreshSummary,
    DatasetImportRequest,
    DatasetImportSummary,
    DatasetScanRequest,
    DatasetScanSummary,
    DatasetSourceList,
    DatasetSourceOption,
    DatasetSourceOptionList,
    DatasetSourceRead,
    DetectedClassList,
    DetectedClassSuggestion,
    ProjectDatasetRead,
    ProjectList,
    ProjectRead,
)
from app.datasets.sources import (
    create_dataset_source,
    delete_dataset_source,
    list_dataset_source_options,
    list_dataset_sources,
)
from app.db.models import Annotation, ClassDef, Dataset, DatasetSource, Image, Prediction, Project
from app.db.session import get_db
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    require_active_entity,
)


router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(get_current_user)],
)
FAILURE_TYPE_PATTERN = "^(all|matched|false_positive|false_negative|class_confusion)$"


projects_router = APIRouter(
    prefix="/api/projects",
    tags=["projects"],
    dependencies=[Depends(get_current_user)],
)


def _get_active_dataset(db: Session, dataset_id: int) -> Dataset:
    try:
        require_active_entity(db, "dataset", dataset_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    return dataset


def _altitude_band(altitude: float | None) -> str:
    if altitude is None:
        return "missing"
    if altitude < 20:
        return "<20m"
    if altitude < 50:
        return "20-50m"
    if altitude < 100:
        return "50-100m"
    return ">=100m"


def _coverage_bucket(
    label: str,
    image_ids: set[int],
    annotated_image_ids: set[int],
    annotation_count: int,
) -> CoverageBucket:
    return CoverageBucket(
        label=label,
        image_count=len(image_ids),
        annotated_image_count=len(image_ids & annotated_image_ids),
        annotation_count=annotation_count,
    )


@projects_router.get("", response_model=ProjectList)
def list_projects(db: Session = Depends(get_db)) -> ProjectList:
    projects = db.scalars(select(Project).order_by(Project.id)).all()
    datasets = db.scalars(
        select(Dataset)
        .where(active_entity_predicate("dataset", Dataset.id))
        .order_by(Dataset.project_id, Dataset.id)
    ).all()
    annotation_counts_by_dataset: dict[int, tuple[int, int]] = {}
    if datasets:
        annotation_counts = db.execute(
            select(
                Image.dataset_id,
                func.count(func.distinct(Annotation.image_id)).label("annotated_image_count"),
                func.count(Annotation.id).label("annotation_count"),
            )
            .join(Annotation, Annotation.image_id == Image.id)
            .where(Image.dataset_id.in_([dataset.id for dataset in datasets]))
            .group_by(Image.dataset_id)
        ).all()
        annotation_counts_by_dataset = {
            dataset_id: (annotated_image_count, annotation_count)
            for dataset_id, annotated_image_count, annotation_count in annotation_counts
        }
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
                        annotated_image_count=annotation_counts_by_dataset.get(dataset.id, (0, 0))[0],
                        annotation_count=annotation_counts_by_dataset.get(dataset.id, (0, 0))[1],
                    )
                    for dataset in datasets_by_project.get(project.id, [])
                ],
            )
            for project in projects
        ]
    )


@router.post("/scan", response_model=DatasetScanSummary)
def scan_dataset(
    request: DatasetScanRequest,
    _: object = Depends(require_admin),
) -> DatasetScanSummary:
    try:
        return scan_dataset_source(request.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset source was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _read_dataset_source(source: DatasetSource) -> DatasetSourceRead:
    return DatasetSourceRead(
        id=source.id,
        original_filename=source.original_filename,
        size_bytes=source.size_bytes,
        source_path=source.stored_path,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.get("/sources", response_model=DatasetSourceOptionList)
def list_dataset_source_library(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> DatasetSourceOptionList:
    return DatasetSourceOptionList(
        items=[DatasetSourceOption(**item) for item in list_dataset_source_options(db)]
    )


@router.get("/sources/uploads", response_model=DatasetSourceList)
def list_uploaded_dataset_sources(
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> DatasetSourceList:
    return DatasetSourceList(items=[_read_dataset_source(item) for item in list_dataset_sources(db)])


@router.post("/sources", response_model=DatasetSourceRead)
async def upload_dataset_source(
    archive: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> DatasetSourceRead:
    try:
        saved = await create_dataset_source(
            db,
            settings,
            archive.filename or "dataset.zip",
            archive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _read_dataset_source(saved)


@router.delete("/sources/{source_id}", status_code=204)
def remove_dataset_source(
    source_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> None:
    try:
        delete_dataset_source(db, settings, source_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/import", response_model=DatasetImportSummary)
def import_dataset_endpoint(
    request: DatasetImportRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> DatasetImportSummary:
    try:
        return import_dataset(db, settings, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset source was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_id}/detected-classes", response_model=DetectedClassList)
def get_detected_classes(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DetectedClassList:
    _get_active_dataset(db, dataset_id)
    payload = detect_classes_for_dataset(db, dataset_id)
    return DetectedClassList(
        dataset_id=payload["dataset_id"],
        total_images=payload["total_images"],
        method=payload["method"],
        items=[DetectedClassSuggestion(**item) for item in payload["items"]],
    )


@router.get("/{dataset_id}/images", response_model=DatasetImageList)
def list_dataset_images(
    dataset_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    platform: str | None = Query(None, max_length=80),
    label_status: str = Query("all", pattern="^(all|annotated|unannotated)$"),
    class_id: int | None = Query(None, ge=1),
    edge_tag: str | None = Query(None, max_length=80),
    failure_type: str = Query("all", pattern=FAILURE_TYPE_PATTERN),
    altitude_min: float | None = Query(None),
    altitude_max: float | None = Query(None),
    db: Session = Depends(get_db),
) -> DatasetImageList:
    _get_active_dataset(db, dataset_id)

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
    if failure_type != "all":
        conditions.append(
            exists().where(
                Prediction.image_id == Image.id,
                Prediction.failure_type == failure_type,
                active_entity_predicate("training_run", Prediction.run_id),
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
                annotation_status=get_annotation_status(image),
                image_url=f"/api/images/{image.id}/file",
            )
            for image, annotation_count in rows
        ],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get("/{dataset_id}/coverage", response_model=DatasetCoverageSummary)
def get_dataset_coverage(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DatasetCoverageSummary:
    dataset = _get_active_dataset(db, dataset_id)

    images = db.scalars(select(Image).where(Image.dataset_id == dataset_id).order_by(Image.id)).all()
    image_ids = {image.id for image in images}
    rows = db.execute(
        select(Annotation, Image, ClassDef)
        .join(Image, Annotation.image_id == Image.id)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Image.dataset_id == dataset_id)
        .order_by(ClassDef.id, Annotation.id)
    ).all()
    project_classes = db.scalars(
        select(ClassDef)
        .where(ClassDef.project_id == dataset.project_id, ClassDef.active == 1)
        .order_by(ClassDef.id)
    ).all()
    annotated_image_ids = {annotation.image_id for annotation, _, _ in rows}
    annotation_count = len(rows)

    platform_image_ids: dict[str, set[int]] = {}
    altitude_image_ids: dict[str, set[int]] = {}
    for image in images:
        platform_label = image.platform or "missing"
        platform_image_ids.setdefault(platform_label, set()).add(image.id)
        altitude_image_ids.setdefault(_altitude_band(image.altitude), set()).add(image.id)

    platform_annotation_counts = {label: 0 for label in platform_image_ids}
    altitude_annotation_counts = {label: 0 for label in altitude_image_ids}
    class_rows: dict[int, dict] = {}
    edge_tag_rows: dict[str, dict[str, set[int] | int]] = {}
    for annotation, image, class_def in rows:
        platform_annotation_counts[image.platform or "missing"] += 1
        altitude_annotation_counts[_altitude_band(image.altitude)] += 1

        class_row = class_rows.setdefault(
            class_def.id,
            {
                "class": class_def,
                "image_ids": set(),
                "annotation_count": 0,
            },
        )
        class_row["image_ids"].add(image.id)
        class_row["annotation_count"] += 1

        for raw_tag in annotation.edge_tags or []:
            tag = str(raw_tag).strip()
            if not tag:
                continue
            tag_row = edge_tag_rows.setdefault(
                tag,
                {
                    "image_ids": set(),
                    "annotation_count": 0,
                },
            )
            tag_row["image_ids"].add(image.id)
            tag_row["annotation_count"] += 1

    platforms = [
        _coverage_bucket(
            label,
            platform_image_ids[label],
            annotated_image_ids,
            platform_annotation_counts[label],
        )
        for label in sorted(platform_image_ids)
    ]
    altitude_order = {"<20m": 0, "20-50m": 1, "50-100m": 2, ">=100m": 3, "missing": 4}
    altitude_bands = [
        _coverage_bucket(
            label,
            altitude_image_ids[label],
            annotated_image_ids,
            altitude_annotation_counts[label],
        )
        for label in sorted(altitude_image_ids, key=lambda item: altitude_order.get(item, 99))
    ]
    classes = []
    for class_def in project_classes:
        row = class_rows.get(
            class_def.id,
            {
                "image_ids": set(),
                "annotation_count": 0,
            },
        )
        classes.append(
            ClassCoverageBucket(
                class_id=class_def.id,
                class_name=class_def.name,
                class_color=class_def.color,
                image_count=len(row["image_ids"]),
                annotation_count=int(row["annotation_count"]),
            )
        )
    edge_tags = [
        {
            "tag": tag,
            "image_count": len(row["image_ids"]),
            "annotation_count": int(row["annotation_count"]),
        }
        for tag, row in sorted(
            edge_tag_rows.items(),
            key=lambda item: (-int(item[1]["annotation_count"]), item[0]),
        )
    ]

    return DatasetCoverageSummary(
        dataset_id=dataset.id,
        image_count=len(image_ids),
        annotated_image_count=len(annotated_image_ids),
        annotation_count=annotation_count,
        platforms=platforms,
        altitude_bands=altitude_bands,
        classes=classes,
        edge_tags=edge_tags,
    )


@router.post("/{dataset_id}/refresh-image-dimensions", response_model=DatasetDimensionRefreshSummary)
def refresh_dataset_image_dimensions(
    dataset_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> DatasetDimensionRefreshSummary:
    dataset = _get_active_dataset(db, dataset_id)

    workspace_root = settings.workspace_root.resolve()
    images = db.scalars(
        select(Image)
        .where(Image.dataset_id == dataset_id, (Image.width.is_(None) | Image.height.is_(None)))
        .order_by(Image.id)
    ).all()
    updated_count = 0
    missing_count = 0

    for image in images:
        file_path = (settings.workspace_root / image.relative_path).resolve()
        if not file_path.is_relative_to(workspace_root) or not file_path.exists():
            missing_count += 1
            continue

        width, height = read_image_dimensions(file_path)
        if width is None or height is None:
            missing_count += 1
            continue

        image.width = width
        image.height = height
        updated_count += 1

    db.commit()
    return DatasetDimensionRefreshSummary(
        dataset_id=dataset.id,
        scanned_count=len(images),
        updated_count=updated_count,
        missing_count=missing_count,
    )


images_router = APIRouter(
    prefix="/api/images",
    tags=["images"],
    dependencies=[Depends(get_current_user)],
)


@images_router.get("/{image_id}/file")
def get_image_file(
    image_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image was not found")
    _get_active_dataset(db, image.dataset_id)

    file_path = (settings.workspace_root / image.relative_path).resolve()
    workspace_root = settings.workspace_root.resolve()
    if not file_path.is_relative_to(workspace_root) or not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file was not found")

    return FileResponse(Path(file_path))
