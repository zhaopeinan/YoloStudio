from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RemoteGpuInfo(BaseModel):
    index: int
    name: str


class RemoteHostStatus(BaseModel):
    id: str
    label: str
    enabled: bool
    online: bool
    ssh_host: str
    remote_root: str
    python: str
    ultralytics_ok: bool = False
    cuda_available: bool = False
    gpu_count: int = 0
    gpus: list[RemoteGpuInfo] = Field(default_factory=list)
    torch_version: str | None = None
    ultralytics_version: str | None = None
    disk: str | None = None
    hostname: str | None = None
    error: str | None = None


class RemoteHostList(BaseModel):
    items: list[RemoteHostStatus]
    local: dict = Field(
        default_factory=lambda: {
            "id": "local",
            "label": "本机 T4",
        }
    )


class RemoteSyncCreate(BaseModel):
    version_id: int
    model_ref: str = Field(min_length=1, max_length=240)
    host: str = Field(default="gpu05", max_length=40)


class RemoteSyncJobRead(BaseModel):
    id: int
    project_id: int
    version_id: int
    model_ref: str
    host: str
    status: str
    bytes_total: int
    bytes_done: int
    files_total: int
    files_done: int
    speed_bps: float
    percent: float | None = None
    fingerprint: str
    remote_fingerprint: str | None
    error_message: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class RemoteSyncJobList(BaseModel):
    items: list[RemoteSyncJobRead]
