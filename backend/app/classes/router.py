from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user, require_admin
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.classes.schemas import ClassCreate, ClassList, ClassRead, ClassUpdate
from app.db.models import Annotation, ClassDef, DatasetVersion, Project
from app.db.session import get_db


router = APIRouter(
    prefix="/api/projects",
    tags=["classes"],
    dependencies=[Depends(get_current_user)],
)

DEFAULT_COLORS = [
    "#2f80ed",
    "#27ae60",
    "#eb5757",
    "#f2994a",
    "#9b51e0",
    "#00a3a3",
]


def _class_usage_counts(
    db: Session,
    project_id: int,
) -> tuple[dict[int, int], dict[int, int]]:
    class_ids = list(
        db.scalars(select(ClassDef.id).where(ClassDef.project_id == project_id)).all()
    )
    annotation_counts = {class_id: 0 for class_id in class_ids}
    version_counts = {class_id: 0 for class_id in class_ids}
    if class_ids:
        annotation_rows = db.execute(
            select(Annotation.class_id, func.count(Annotation.id))
            .where(Annotation.class_id.in_(class_ids))
            .group_by(Annotation.class_id)
        ).all()
        annotation_counts.update({int(class_id): int(count) for class_id, count in annotation_rows})

        versions = db.scalars(
            select(DatasetVersion).where(DatasetVersion.project_id == project_id)
        ).all()
        for version in versions:
            mapping = {str(class_id) for class_id in (version.class_mapping or {})}
            for class_id in class_ids:
                if str(class_id) in mapping:
                    version_counts[class_id] += 1
    return annotation_counts, version_counts


def _read_class(
    class_def: ClassDef,
    annotation_count: int = 0,
    version_count: int = 0,
) -> ClassRead:
    return ClassRead(
        id=class_def.id,
        project_id=class_def.project_id,
        name=class_def.name,
        color=class_def.color,
        description=class_def.description,
        active=bool(class_def.active),
        annotation_count=annotation_count,
        version_count=version_count,
    )


@router.get("/{project_id}/classes", response_model=ClassList)
def list_project_classes(project_id: int, db: Session = Depends(get_db)) -> ClassList:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")

    classes = db.scalars(
        select(ClassDef).where(ClassDef.project_id == project_id).order_by(ClassDef.id)
    ).all()
    annotation_counts, version_counts = _class_usage_counts(db, project_id)
    return ClassList(
        items=[
            _read_class(
                class_def,
                annotation_counts.get(class_def.id, 0),
                version_counts.get(class_def.id, 0),
            )
            for class_def in classes
        ]
    )


@router.post("/{project_id}/classes", response_model=ClassRead)
def create_project_class(
    project_id: int,
    request: ClassCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> ClassRead:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")

    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Class name cannot be empty")

    existing_count = (
        db.scalar(select(func.count()).select_from(ClassDef).where(ClassDef.project_id == project_id))
        or 0
    )
    color = request.color or DEFAULT_COLORS[existing_count % len(DEFAULT_COLORS)]
    class_def = ClassDef(
        project_id=project_id,
        name=name,
        color=color,
        description=request.description,
        active=1,
    )
    db.add(class_def)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Class name already exists for project") from exc
    db.refresh(class_def)
    return _read_class(class_def)


@router.patch("/{project_id}/classes/{class_id}", response_model=ClassRead)
def update_project_class(
    project_id: int,
    class_id: int,
    request: ClassUpdate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> ClassRead:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")

    class_def = db.get(ClassDef, class_id)
    if class_def is None or class_def.project_id != project_id:
        raise HTTPException(status_code=404, detail="Class was not found")

    if request.name is not None:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")
        class_def.name = name
    if request.color is not None:
        class_def.color = request.color
    if "description" in request.model_fields_set:
        class_def.description = request.description

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Class name already exists for project") from exc
    db.refresh(class_def)
    annotation_counts, version_counts = _class_usage_counts(db, project_id)
    return _read_class(
        class_def,
        annotation_counts.get(class_def.id, 0),
        version_counts.get(class_def.id, 0),
    )


@router.delete("/{project_id}/classes/{class_id}", response_model=ClassRead)
def delete_project_class(
    project_id: int,
    class_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> ClassRead:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")

    class_def = db.get(ClassDef, class_id)
    if class_def is None or class_def.project_id != project_id:
        raise HTTPException(status_code=404, detail="Class was not found")

    annotation_counts, version_counts = _class_usage_counts(db, project_id)
    annotation_count = annotation_counts.get(class_id, 0)
    version_count = version_counts.get(class_id, 0)
    if annotation_count or version_count:
        reasons = []
        if annotation_count:
            reasons.append(f"{annotation_count} 条标注")
        if version_count:
            reasons.append(f"{version_count} 个数据集版本")
        raise HTTPException(
            status_code=409,
            detail=f"类别“{class_def.name}”仍被{'、'.join(reasons)}使用，不能删除。",
        )

    deleted = _read_class(class_def, annotation_count, version_count)
    db.delete(class_def)
    db.commit()
    return deleted
