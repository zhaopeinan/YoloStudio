from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DatasetVersionCreate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    class_ids: list[int] | None = None
    image_scope: Literal["annotated", "all"] = "annotated"
    dataset_ids: list[int] | None = Field(
        default=None,
        description="Optional extra/same-project dataset IDs to merge into one training version",
    )


class DatasetVersionRead(BaseModel):
    id: int
    project_id: int
    dataset_id: int
    name: str
    class_mapping: dict[str, int]
    image_scope: Literal["annotated", "all"]
    split_counts: dict[str, int]
    source_dataset_ids: list[int] = Field(default_factory=list)
    merged: bool = False
    artifact_path: str
    frozen: bool
    created_at: datetime


class DatasetVersionList(BaseModel):
    items: list[DatasetVersionRead]
