from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PreviewModelOption(BaseModel):
    model_ref: str
    label: str
    kind: Literal["base", "trained"]
    run_id: int | None = None
    status: str | None = None


class PreviewModelList(BaseModel):
    items: list[PreviewModelOption]


class PreviewBox(BaseModel):
    class_id: int
    class_name: str
    color: str
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class ImagePreviewRequest(BaseModel):
    model_ref: str = Field(min_length=1, max_length=240)
    confidence_threshold: float = Field(default=0.25, ge=0.01, le=0.99)


class ImagePreviewResponse(BaseModel):
    image_id: int
    filename: str
    image_url: str
    width: int
    height: int
    model_ref: str
    confidence_threshold: float
    annotations: list[PreviewBox]
    predictions: list[PreviewBox]


class PreviewJobRead(BaseModel):
    id: int
    project_id: int
    dataset_id: int | None
    run_id: int | None
    video_id: int | None = None
    model_ref: str
    kind: str
    source_filename: str
    status: str
    confidence_threshold: float
    frame_step: int
    fps: float | None
    total_frames: int
    processed_frames: int
    error_message: str | None
    result_url: str | None
    stream_url: str | None
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None


class PreviewJobList(BaseModel):
    items: list[PreviewJobRead]


class PreviewVideoRead(BaseModel):
    id: int
    project_id: int
    original_filename: str
    size_bytes: int
    created_at: datetime


class PreviewVideoList(BaseModel):
    items: list[PreviewVideoRead]