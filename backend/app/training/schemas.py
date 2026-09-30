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
    model: str = Field(default="base:yolov8n.pt", min_length=1, max_length=240)
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


class TrainingRunArtifact(BaseModel):
    relative_path: str
    category: str
    size_bytes: int


class TrainingRunArtifactSummary(BaseModel):
    run_id: int
    artifact_root: str
    total_count: int
    items: list[TrainingRunArtifact]


class ModelWeightRead(BaseModel):
    id: int
    project_id: int
    original_filename: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime


class ModelWeightList(BaseModel):
    items: list[ModelWeightRead]


class TrainingModelOption(BaseModel):
    model_ref: str
    label: str
    kind: str
    run_id: int | None = None
    weight_id: int | None = None
    status: str


class TrainingModelList(BaseModel):
    items: list[TrainingModelOption]


class TrainingLiveProgress(BaseModel):
    epoch: int | None = None
    total_epochs: int | None = None
    percent: float | None = None
    elapsed_sec: float | None = None
    eta_sec: float | None = None
    phase: str


class GpuProcessInfo(BaseModel):
    pid: int | None = None
    name: str
    memory_mb: float | None = None


class GpuDeviceInfo(BaseModel):
    index: int
    name: str
    utilization_gpu: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    processes: list[GpuProcessInfo] = Field(default_factory=list)


class GpuStatus(BaseModel):
    available: bool
    gpus: list[GpuDeviceInfo] = Field(default_factory=list)
    error: str | None = None
    queried_at: str


class TrainingLiveSnapshot(BaseModel):
    run_id: int
    project_id: int
    status: str
    status_label: str
    device: str
    config: dict
    progress: TrainingLiveProgress
    latest: dict[str, float]
    series: dict[str, list[float | None]]
    log_tail: list[str]
    error_message: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    updated_at: str
    gpu: GpuStatus | None = None
