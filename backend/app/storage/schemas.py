from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class StorageBlocker(BaseModel):
    entity_type: str
    entity_id: int
    display_name: str
    status: str | None = None


class StorageItemRead(BaseModel):
    entity_type: str
    entity_id: int
    display_name: str
    project_id: int
    project_name: str
    dataset_id: int | None
    version_id: int | None
    artifact_path: str
    size_bytes: int
    image_count: int | None
    annotation_count: int | None
    split_counts: dict[str, int] | None
    created_at: datetime
    protected: bool
    blockers: list[StorageBlocker]


class StorageItemList(BaseModel):
    items: list[StorageItemRead]
    total_size_bytes: int


class RelatedRunRead(BaseModel):
    id: int
    status: str
    model: str
    size_bytes: int
    prediction_job_count: int
    export_count: int
    created_at: datetime


class StorageItemDetail(StorageItemRead):
    class_names: list[str]
    related_runs: list[RelatedRunRead]
