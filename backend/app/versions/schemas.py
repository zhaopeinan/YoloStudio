from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DatasetVersionCreate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    class_ids: list[int] | None = None


class DatasetVersionRead(BaseModel):
    id: int
    project_id: int
    dataset_id: int
    name: str
    class_mapping: dict[str, int]
    split_counts: dict[str, int]
    artifact_path: str
    frozen: bool
    created_at: datetime


class DatasetVersionList(BaseModel):
    items: list[DatasetVersionRead]
