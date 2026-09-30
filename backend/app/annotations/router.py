from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.annotations.schemas import AnnotationList, AnnotationRead, AnnotationReplace
from app.annotations.status import get_annotation_status, set_annotation_status
from app.db.models import Annotation, ClassDef, Dataset, Image
from app.db.session import get_db
from app.storage.visibility import StorageEntityNotFoundError, require_active_entity


router = APIRouter(
    prefix="/api/images",
    tags=["annotations"],
    dependencies=[Depends(get_current_user)],
)


def _get_active_image(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image was not found")
    try:
        require_active_entity(db, "dataset", image.dataset_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return image


def _image_project_id(db: Session, image: Image) -> int:
    dataset = db.get(Dataset, image.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Image dataset was not found")
    return dataset.project_id


def _read_annotation(annotation: Annotation, class_def: ClassDef) -> AnnotationRead:
    return AnnotationRead(
        id=annotation.id,
        image_id=annotation.image_id,
        class_id=annotation.class_id,
        class_name=class_def.name,
        class_color=class_def.color,
        x_center=annotation.x_center,
        y_center=annotation.y_center,
        width=annotation.width,
        height=annotation.height,
        track_id=annotation.track_id,
        edge_tags=annotation.edge_tags or [],
    )


def _list_annotations(db: Session, image_id: int) -> AnnotationList:
    image = _get_active_image(db, image_id)
    rows = db.execute(
        select(Annotation, ClassDef)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.id)
    ).all()
    return AnnotationList(
        items=[_read_annotation(annotation, class_def) for annotation, class_def in rows],
        annotation_status=get_annotation_status(image),
    )


@router.get("/{image_id}/annotations", response_model=AnnotationList)
def list_image_annotations(image_id: int, db: Session = Depends(get_db)) -> AnnotationList:
    return _list_annotations(db, image_id)


@router.put("/{image_id}/annotations", response_model=AnnotationList)
def replace_image_annotations(
    image_id: int,
    request: AnnotationReplace,
    db: Session = Depends(get_db),
) -> AnnotationList:
    image = _get_active_image(db, image_id)

    if request.annotations and request.annotation_status == "negative":
        raise HTTPException(
            status_code=400,
            detail="带有边界框的图像不能标记为无目标",
        )
    if request.annotations and request.annotation_status not in {None, "annotated"}:
        raise HTTPException(
            status_code=400,
            detail="带有边界框的图像状态必须为已标注",
        )

    project_id = _image_project_id(db, image)
    class_ids = {annotation.class_id for annotation in request.annotations}
    if class_ids:
        classes = db.scalars(select(ClassDef).where(ClassDef.id.in_(class_ids))).all()
        classes_by_id = {class_def.id: class_def for class_def in classes}
        missing_ids = class_ids - set(classes_by_id)
        if missing_ids:
            raise HTTPException(status_code=400, detail="Annotation class was not found")
        if any(class_def.project_id != project_id for class_def in classes_by_id.values()):
            raise HTTPException(
                status_code=400,
                detail="Annotation class must belong to the image project",
            )

    db.execute(delete(Annotation).where(Annotation.image_id == image_id))
    for annotation in request.annotations:
        db.add(
            Annotation(
                image_id=image_id,
                class_id=annotation.class_id,
                x_center=annotation.x_center,
                y_center=annotation.y_center,
                width=annotation.width,
                height=annotation.height,
                track_id=annotation.track_id,
                edge_tags=annotation.edge_tags,
            )
        )
    next_status = "annotated" if request.annotations else request.annotation_status or "unreviewed"
    set_annotation_status(image, next_status)
    db.commit()
    return _list_annotations(db, image_id)
