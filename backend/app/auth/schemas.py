from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RoleName = Literal["admin", "annotator"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserRead"


class UserRead(BaseModel):
    id: int
    username: str
    role: RoleName
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=6, max_length=128)
    role: RoleName = "annotator"
    is_active: bool = True


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=128)
    role: RoleName | None = None
    is_active: bool | None = None


class UserList(BaseModel):
    items: list[UserRead]
