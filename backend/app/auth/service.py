from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.schemas import UserCreate, UserRead
from app.auth.security import hash_password
from app.core.settings import Settings
from app.db.models import User


def to_user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        username=user.username,
        role=user.role,  # type: ignore[arg-type]
        is_active=bool(user.is_active),
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def create_user(db: Session, payload: UserCreate) -> User:
    user = User(
        username=payload.username.strip(),
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def ensure_default_admin(db: Session, settings: Settings) -> User | None:
    count = db.scalar(select(func.count()).select_from(User)) or 0
    if count > 0:
        return None
    return create_user(
        db,
        UserCreate(
            username=settings.default_admin_username,
            password=settings.default_admin_password,
            role="admin",
            is_active=True,
        ),
    )
