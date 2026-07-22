from __future__ import annotations

from pydantic import BaseModel, Field


class ClassCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    color: str | None = Field(default=None, max_length=24)
    description: str | None = None


class ClassUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    color: str | None = Field(default=None, max_length=24)
    description: str | None = None


class ClassRead(BaseModel):
    id: int
    project_id: int
    name: str
    color: str
    description: str | None
    active: bool
    annotation_count: int = 0
    version_count: int = 0


class ClassList(BaseModel):
    items: list[ClassRead]
