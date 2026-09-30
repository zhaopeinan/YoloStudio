from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PredictionImageFilters(BaseModel):
    platform: str | None = Field(default=None, max_length=80)
    label_status: str = Field(default="all", pattern="^(all|annotated|unannotated)$")
    class_id: int | None = Field(default=None, ge=1)
    edge_tag: str | None = Field(default=None, max_length=80)
    failure_type: str = Field(
        default="all",
        pattern="^(all|matched|false_positive|false_negative|class_confusion)$",
    )
    altitude_min: float | None = None
    altitude_max: float | None = None


class PredictionJobCreate(BaseModel):
    image_scope: str = Field(default="all", pattern="^(all|train|val|test)$")
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    version_id: int | None = Field(default=None, ge=1)
    image_filters: PredictionImageFilters | None = None


class PredictionThresholdScanCreate(BaseModel):
    image_scope: str = Field(default="all", pattern="^(all|train|val|test)$")
    thresholds: list[float] = Field(..., min_length=1, max_length=20)
    version_id: int | None = Field(default=None, ge=1)
    image_filters: PredictionImageFilters | None = None

    @field_validator("thresholds")
    @classmethod
    def normalize_thresholds(cls, value: list[float]) -> list[float]:
        thresholds = sorted({round(threshold, 4) for threshold in value})
        if any(threshold < 0 or threshold > 1 for threshold in thresholds):
            raise ValueError("Thresholds must be between 0 and 1")
        if not thresholds:
            raise ValueError("At least one threshold is required")
        return thresholds


class PredictionJobRead(BaseModel):
    id: int
    run_id: int
    project_id: int
    version_id: int | None
    status: str
    image_scope: str
    confidence_threshold: float
    image_filters: PredictionImageFilters | None
    artifact_path: str
    log_path: str
    image_count: int
    prediction_count: int
    matched_count: int
    false_positive_count: int
    false_negative_count: int
    class_confusion_count: int
    error_message: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class PredictionJobList(BaseModel):
    items: list[PredictionJobRead]


class PredictionThresholdScanRead(BaseModel):
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


class PredictionReviewImage(BaseModel):
    id: int
    relative_path: str
    image_url: str
    platform: str | None
    altitude: float | None
    timestamp: float | None


class PredictionReviewAnnotation(BaseModel):
    id: int
    image_id: int
    class_id: int
    class_name: str
    class_color: str
    x_center: float
    y_center: float
    width: float
    height: float
    track_id: str | None
    edge_tags: list[str]


class PredictionImageReview(BaseModel):
    image: PredictionReviewImage
    annotations: list[PredictionReviewAnnotation]
    predictions: list[PredictionRead]
    counts: dict[str, int]
