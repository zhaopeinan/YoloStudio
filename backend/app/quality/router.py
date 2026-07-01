from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Annotation, ClassDef, Dataset, Image
from app.db.session import get_db
from app.prediction.matching import annotation_box, iou
from app.quality.schemas import DatasetQualityIssue, DatasetQualityIssueList, DatasetQualitySummary


router = APIRouter(prefix="/api/datasets", tags=["quality"])
duplicate_iou_threshold = 0.95
QUALITY_ISSUE_TYPES = {
    "unannotated_image",
    "tiny_box",
    "invalid_box",
    "duplicate_box",
    "missing_metadata",
    "missing_image_dimensions",
    "unknown_class_reference",
}


def is_invalid_box(annotation: Annotation) -> bool:
    return (
        annotation.width <= 0
        or annotation.height <= 0
        or annotation.x_center < 0
        or annotation.y_center < 0
        or annotation.x_center > 1
        or annotation.y_center > 1
        or annotation.width > 1
        or annotation.height > 1
        or annotation.x_center - annotation.width / 2 < 0
        or annotation.x_center + annotation.width / 2 > 1
        or annotation.y_center - annotation.height / 2 < 0
        or annotation.y_center + annotation.height / 2 > 1
    )


def is_tiny_box(annotation: Annotation, image: Image) -> bool:
    if image.width is None or image.height is None:
        return False
    return annotation.width * image.width < 10 or annotation.height * image.height < 10


def _duplicate_box_count(annotations: list[Annotation]) -> int:
    seen_by_image_class: dict[tuple[int, int], list[Annotation]] = {}
    duplicate_count = 0
    for annotation in annotations:
        key = (annotation.image_id, annotation.class_id)
        existing_annotations = seen_by_image_class.setdefault(key, [])
        if any(
            iou(annotation_box(existing), annotation_box(annotation)) >= duplicate_iou_threshold
            for existing in existing_annotations
        ):
            duplicate_count += 1
        existing_annotations.append(annotation)
    return duplicate_count


def missing_metadata_fields(image: Image) -> list[str]:
    fields: list[str] = []
    if not image.metadata_:
        fields.append("source metadata row")
    if not image.platform:
        fields.append("platform")
    if image.altitude is None:
        fields.append("altitude")
    if image.timestamp is None:
        fields.append("timestamp")
    return fields


def has_missing_image_dimensions(image: Image) -> bool:
    return image.width is None or image.height is None


def _import_warnings(dataset: Dataset, warning_type: str | None = None) -> list[dict]:
    warnings = dataset.metadata_.get("import_warnings", [])
    if not isinstance(warnings, list):
        return []
    typed_warnings = [warning for warning in warnings if isinstance(warning, dict)]
    if warning_type is None:
        return typed_warnings
    return [warning for warning in typed_warnings if warning.get("type") == warning_type]


def build_quality_summary(db: Session, dataset_id: int) -> DatasetQualitySummary:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")

    image_count = (
        db.scalar(select(func.count()).select_from(Image).where(Image.dataset_id == dataset_id))
        or 0
    )
    annotated_image_count = (
        db.scalar(
            select(func.count(func.distinct(Annotation.image_id)))
            .select_from(Annotation)
            .join(Image, Annotation.image_id == Image.id)
            .where(Image.dataset_id == dataset_id)
        )
        or 0
    )
    annotation_count = (
        db.scalar(
            select(func.count())
            .select_from(Annotation)
            .join(Image, Annotation.image_id == Image.id)
            .where(Image.dataset_id == dataset_id)
        )
        or 0
    )
    class_count = (
        db.scalar(
            select(func.count())
            .select_from(ClassDef)
            .where(ClassDef.project_id == dataset.project_id, ClassDef.active == 1)
        )
        or 0
    )

    rows = db.execute(
        select(Annotation, Image)
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id)
    ).all()
    invalid_box_count = sum(1 for annotation, _ in rows if is_invalid_box(annotation))
    tiny_box_count = sum(1 for annotation, image in rows if is_tiny_box(annotation, image))
    duplicate_box_count = _duplicate_box_count([annotation for annotation, _ in rows])
    images = db.scalars(select(Image).where(Image.dataset_id == dataset_id).order_by(Image.id)).all()
    missing_metadata_count = sum(1 for image in images if missing_metadata_fields(image))
    missing_image_dimensions_count = sum(1 for image in images if has_missing_image_dimensions(image))
    unannotated_image_count = image_count - annotated_image_count
    unknown_class_reference_count = sum(
        int(warning.get("count") or 0)
        for warning in _import_warnings(dataset, "unknown_class_reference")
    )

    issues: list[str] = []
    if image_count == 0:
        issues.append("Dataset has no images.")
    if class_count == 0:
        issues.append("Project has no active classes.")
    if annotation_count == 0:
        issues.append("Dataset has no saved annotations.")
    if unannotated_image_count > 0:
        noun = "image has" if unannotated_image_count == 1 else "images have"
        issues.append(f"{unannotated_image_count} {noun} no annotations.")
    if invalid_box_count > 0:
        noun = "box has" if invalid_box_count == 1 else "boxes have"
        issues.append(f"{invalid_box_count} {noun} invalid geometry.")
    if tiny_box_count > 0:
        noun = "box is" if tiny_box_count == 1 else "boxes are"
        issues.append(f"{tiny_box_count} {noun} smaller than 10x10 pixels.")
    if duplicate_box_count > 0:
        noun = "box duplicates" if duplicate_box_count == 1 else "boxes duplicate"
        issues.append(f"{duplicate_box_count} {noun} another box on the same image and class.")
    if missing_metadata_count > 0:
        noun = "image is" if missing_metadata_count == 1 else "images are"
        issues.append(
            f"{missing_metadata_count} {noun} missing platform, altitude, timestamp, or source metadata."
        )
    if missing_image_dimensions_count > 0:
        noun = "image has" if missing_image_dimensions_count == 1 else "images have"
        issues.append(f"{missing_image_dimensions_count} {noun} unreadable image dimensions.")
    if unknown_class_reference_count > 0:
        noun = "label references" if unknown_class_reference_count == 1 else "label references"
        issues.append(f"{unknown_class_reference_count} {noun} unknown class indexes.")

    ready_for_training = (
        image_count > 0
        and class_count > 0
        and annotation_count > 0
        and invalid_box_count == 0
        and duplicate_box_count == 0
    )
    return DatasetQualitySummary(
        dataset_id=dataset.id,
        image_count=image_count,
        annotated_image_count=annotated_image_count,
        unannotated_image_count=unannotated_image_count,
        annotation_count=annotation_count,
        class_count=class_count,
        tiny_box_count=tiny_box_count,
        invalid_box_count=invalid_box_count,
        duplicate_box_count=duplicate_box_count,
        missing_metadata_count=missing_metadata_count,
        missing_image_dimensions_count=missing_image_dimensions_count,
        unknown_class_reference_count=unknown_class_reference_count,
        ready_for_training=ready_for_training,
        issues=issues,
    )


def _annotation_issue(
    issue_type: str,
    severity: str,
    message: str,
    annotation: Annotation,
    image: Image,
    class_def: ClassDef,
) -> DatasetQualityIssue:
    return DatasetQualityIssue(
        issue_type=issue_type,
        severity=severity,
        message=message,
        image_id=image.id,
        image_path=image.relative_path,
        image_url=f"/api/images/{image.id}/file",
        annotation_id=annotation.id,
        class_id=annotation.class_id,
        class_name=class_def.name,
        x_center=annotation.x_center,
        y_center=annotation.y_center,
        width=annotation.width,
        height=annotation.height,
    )


def _duplicate_box_issues(
    rows: list[tuple[Annotation, Image, ClassDef]],
) -> list[DatasetQualityIssue]:
    seen_by_image_class: dict[tuple[int, int], list[Annotation]] = {}
    issues: list[DatasetQualityIssue] = []
    for annotation, image, class_def in rows:
        key = (image.id, annotation.class_id)
        existing_annotations = seen_by_image_class.setdefault(key, [])
        matched_annotation = next(
            (
                existing
                for existing in existing_annotations
                if iou(annotation_box(existing), annotation_box(annotation)) >= duplicate_iou_threshold
            ),
            None,
        )
        if matched_annotation is not None:
            issues.append(
                _annotation_issue(
                    "duplicate_box",
                    "error",
                    f"Box duplicates annotation {matched_annotation.id} on the same image and class.",
                    annotation,
                    image,
                    class_def,
                )
            )
        existing_annotations.append(annotation)
    return issues


def build_quality_issues(
    db: Session,
    dataset_id: int,
    issue_type: str = "all",
    limit: int = 50,
    offset: int = 0,
) -> DatasetQualityIssueList:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset was not found")
    if issue_type != "all" and issue_type not in QUALITY_ISSUE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported quality issue type")

    issues: list[DatasetQualityIssue] = []
    if issue_type in {"all", "unannotated_image"}:
        annotation_counts = (
            select(Annotation.image_id, func.count(Annotation.id).label("annotation_count"))
            .group_by(Annotation.image_id)
            .subquery()
        )
        images = db.scalars(
            select(Image)
            .outerjoin(annotation_counts, Image.id == annotation_counts.c.image_id)
            .where(
                Image.dataset_id == dataset_id,
                func.coalesce(annotation_counts.c.annotation_count, 0) == 0,
            )
            .order_by(Image.id)
        ).all()
        issues.extend(
            DatasetQualityIssue(
                issue_type="unannotated_image",
                severity="warning",
                message="Image has no saved annotations.",
                image_id=image.id,
                image_path=image.relative_path,
                image_url=f"/api/images/{image.id}/file",
            )
            for image in images
        )

    rows = db.execute(
        select(Annotation, Image, ClassDef)
        .join(Image, Annotation.image_id == Image.id)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Image.dataset_id == dataset_id)
        .order_by(Image.id, Annotation.id)
    ).all()
    for annotation, image, class_def in rows:
        if issue_type in {"all", "invalid_box"} and is_invalid_box(annotation):
            issues.append(
                _annotation_issue(
                    "invalid_box",
                    "error",
                    "Box geometry falls outside normalized image bounds.",
                    annotation,
                    image,
                    class_def,
                )
            )
        if issue_type in {"all", "tiny_box"} and is_tiny_box(annotation, image):
            issues.append(
                _annotation_issue(
                    "tiny_box",
                    "warning",
                    "Box is smaller than 10x10 pixels.",
                    annotation,
                    image,
                    class_def,
                )
            )

    if issue_type in {"all", "duplicate_box"}:
        issues.extend(_duplicate_box_issues(rows))

    if issue_type in {"all", "missing_metadata"}:
        images = db.scalars(select(Image).where(Image.dataset_id == dataset_id).order_by(Image.id)).all()
        for image in images:
            missing_fields = missing_metadata_fields(image)
            if not missing_fields:
                continue
            issues.append(
                DatasetQualityIssue(
                    issue_type="missing_metadata",
                    severity="warning",
                    message=f"Image is missing {', '.join(missing_fields)}.",
                    image_id=image.id,
                    image_path=image.relative_path,
                    image_url=f"/api/images/{image.id}/file",
                )
            )

    if issue_type in {"all", "missing_image_dimensions"}:
        images = db.scalars(select(Image).where(Image.dataset_id == dataset_id).order_by(Image.id)).all()
        for image in images:
            if not has_missing_image_dimensions(image):
                continue
            issues.append(
                DatasetQualityIssue(
                    issue_type="missing_image_dimensions",
                    severity="warning",
                    message="Image width or height could not be read.",
                    image_id=image.id,
                    image_path=image.relative_path,
                    image_url=f"/api/images/{image.id}/file",
                )
            )

    if issue_type in {"all", "unknown_class_reference"}:
        images_by_id = {
            image.id: image
            for image in db.scalars(select(Image).where(Image.dataset_id == dataset_id)).all()
        }
        for warning in _import_warnings(dataset, "unknown_class_reference"):
            image_id = warning.get("image_id")
            image = images_by_id.get(image_id) if isinstance(image_id, int) else None
            class_index = warning.get("class_index")
            count = int(warning.get("count") or 0)
            image_path = str(warning.get("image_path") or warning.get("source_image") or "")
            issues.append(
                DatasetQualityIssue(
                    issue_type="unknown_class_reference",
                    severity="error",
                    message=f"{count} label row references unknown class index {class_index}.",
                    image_id=image.id if image is not None else 0,
                    image_path=image.relative_path if image is not None else image_path,
                    image_url=f"/api/images/{image.id}/file" if image is not None else "",
                    class_id=None,
                    class_name=f"YOLO class {class_index}",
                )
            )

    total = len(issues)
    return DatasetQualityIssueList(
        dataset_id=dataset_id,
        limit=limit,
        offset=offset,
        total=total,
        items=issues[offset : offset + limit],
    )


@router.get("/{dataset_id}/quality", response_model=DatasetQualitySummary)
def get_dataset_quality(dataset_id: int, db: Session = Depends(get_db)) -> DatasetQualitySummary:
    return build_quality_summary(db, dataset_id)


@router.get("/{dataset_id}/quality/issues", response_model=DatasetQualityIssueList)
def get_dataset_quality_issues(
    dataset_id: int,
    issue_type: str = Query("all"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DatasetQualityIssueList:
    return build_quality_issues(db, dataset_id, issue_type, limit, offset)
