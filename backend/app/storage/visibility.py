from __future__ import annotations

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.db.models import TrashItem


TRASHED_STATUSES = {"pending_move", "active", "pending_restore", "error"}
SUPPORTED_ENTITY_TYPES = {"dataset", "dataset_version", "training_run"}


class StorageEntityNotFoundError(LookupError):
    pass


def active_entity_predicate(entity_type: str, entity_id_column):
    return ~exists().where(
        TrashItem.entity_type == entity_type,
        TrashItem.entity_id == entity_id_column,
        TrashItem.status.in_(TRASHED_STATUSES),
    )


def is_entity_trashed(db: Session, entity_type: str, entity_id: int) -> bool:
    return bool(
        db.scalar(
            select(TrashItem.id).where(
                TrashItem.entity_type == entity_type,
                TrashItem.entity_id == entity_id,
                TrashItem.status.in_(TRASHED_STATUSES),
            )
        )
    )


def require_active_entity(db: Session, entity_type: str, entity_id: int) -> None:
    if entity_type not in SUPPORTED_ENTITY_TYPES or is_entity_trashed(
        db, entity_type, entity_id
    ):
        raise StorageEntityNotFoundError("存储对象不存在或已移入回收站")
