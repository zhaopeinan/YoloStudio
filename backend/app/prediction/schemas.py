from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionJobCreate(BaseModel):
    image_scope: str = Field(default="all", pattern="^(all|train|val|test)$")
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)


class PredictionJobRead(BaseModel):
    id: int
    run_id: int
    project_id: int
    status: str
    image_scope: str
    confidence_threshold: float
    artifact_path: str
    log_path: str
    image_count: int
    prediction_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    error_message: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class PredictionJobList(BaseModel):
    items: list[PredictionJobRead]


class PredictionRead(BaseModel):
    id: int
    run_id: int
    job_id: int
    image_id: int
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    matched_annotation_id: int | None
    failure_type: str


class PredictionList(BaseModel):
    items: list[PredictionRead]


class PredictionJobLogs(BaseModel):
    job_id: int
    text: str
