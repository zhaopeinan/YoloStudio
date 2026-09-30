from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ExportCreate(BaseModel):
    format: str = Field(pattern="^(pt|onnx|tensorrt)$")


class ExportArtifactRead(BaseModel):
    id: int
    run_id: int
    project_id: int
    format: str
    status: str
    artifact_path: str
    download_url: str | None = None
    error_message: str | None
    metadata: dict
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ExportArtifactList(BaseModel):
    items: list[ExportArtifactRead]


class ExportCapabilities(BaseModel):
    pt_available: bool
    onnx_available: bool
    tensorrt_available: bool
    weights_path: str | None
    reasons: dict[str, str]
