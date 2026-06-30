from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Annotation, ClassDef, Dataset, Image
from app.db.session import get_db
from app.quality.schemas import DatasetQualitySummary


router = APIRouter(prefix="/api/datasets", tags=["quality"])


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
    unannotated_image_count = image_count - annotated_image_count

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

    ready_for_training = (
        image_count > 0 and class_count > 0 and annotation_count > 0 and invalid_box_count == 0
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
        ready_for_training=ready_for_training,
        issues=issues,
    )


@router.get("/{dataset_id}/quality", response_model=DatasetQualitySummary)
def get_dataset_quality(dataset_id: int, db: Session = Depends(get_db)) -> DatasetQualitySummary:
    return build_quality_summary(db, dataset_id)
