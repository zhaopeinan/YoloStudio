from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TrainingAugmentationConfig(BaseModel):
    mosaic: float = Field(default=1.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)
    copy_paste: float = Field(default=0.0, ge=0, le=1)
    hsv_h: float = Field(default=0.015, ge=0, le=1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    translate: float = Field(default=0.1, ge=0, le=1)
    scale: float = Field(default=0.5, ge=0, le=2)
    fliplr: float = Field(default=0.5, ge=0, le=1)
    erasing: float = Field(default=0.4, ge=0, le=1)
    gridmask: bool = False


class TrainingRunCreate(BaseModel):
    version_id: int
    model: str = Field(default="yolov8n.pt", min_length=1, max_length=240)
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=640, ge=32, le=4096)
    batch_size: int = Field(default=8, ge=1, le=256)
    device: str | None = Field(default=None, max_length=40)
    augmentation_preset: str = Field(default="balanced", max_length=80)
    augmentation: TrainingAugmentationConfig = Field(default_factory=TrainingAugmentationConfig)
    tta: bool = False
    threshold_scan: bool = False


class TrainingRunRead(BaseModel):
    id: int
    project_id: int
    version_id: int
    status: str
    device: str
    config: dict
    artifact_path: str
    log_path: str
    error_message: str | None
    latest_metrics: dict[str, float]
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TrainingRunList(BaseModel):
    items: list[TrainingRunRead]


class TrainingRunLogs(BaseModel):
    run_id: int
    text: str
