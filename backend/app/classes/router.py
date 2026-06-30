from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.classes.schemas import ClassCreate, ClassList, ClassRead
from app.db.models import ClassDef, Project
from app.db.session import get_db


router = APIRouter(prefix="/api/projects", tags=["classes"])

DEFAULT_COLORS = [
    "#2f80ed",
    "#27ae60",
    "#eb5757",
    "#f2994a",
    "#9b51e0",
    "#00a3a3",
]


def _read_class(class_def: ClassDef) -> ClassRead:
    return ClassRead(
        id=class_def.id,
        project_id=class_def.project_id,
        name=class_def.name,
        color=class_def.color,
        description=class_def.description,
        active=bool(class_def.active),
    )


@router.get("/{project_id}/classes", response_model=ClassList)
def list_project_classes(project_id: int, db: Session = Depends(get_db)) -> ClassList:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")

    classes = db.scalars(
        select(ClassDef).where(ClassDef.project_id == project_id).order_by(ClassDef.id)
    ).all()
    return ClassList(items=[_read_class(class_def) for class_def in classes])


@router.post("/{project_id}/classes", response_model=ClassRead)
def create_project_class(
    project_id: int,
    request: ClassCreate,
    db: Session = Depends(get_db),
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
