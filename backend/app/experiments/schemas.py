from __future__ import annotations

from pydantic import BaseModel


class MetricPoint(BaseModel):
    epoch: int | None
    step: int | None
    value: float


class MetricSeries(BaseModel):
    name: str
    points: list[MetricPoint]
    latest: float | None


class ClassOutcome(BaseModel):
    class_id: int
    class_name: str
    matched: int
    false_positive: int
    false_negative: int
    class_confusion: int


class ConfusionCell(BaseModel):
    actual_class_id: int
    actual_class_name: str
    predicted_class_id: int
    predicted_class_name: str
    count: int


class ThresholdPoint(BaseModel):
    job_id: int
    confidence_threshold: float
    matched: int
    false_positive: int
    false_negative: int
    class_confusion: int
    precision: float
    recall: float
    f1: float


class ThresholdRecommendation(BaseModel):
    job_id: int
    confidence_threshold: float
    precision: float
    recall: float
    f1: float


class RunExperimentSummary(BaseModel):
    run_id: int
    metric_series: list[MetricSeries]
    class_outcomes: list[ClassOutcome]
    confusion_matrix: list[ConfusionCell]
    threshold_scan: list[ThresholdPoint]
    threshold_recommendation: ThresholdRecommendation | None
    latest_prediction_job_id: int | None


class RunComparisonRow(BaseModel):
    run_id: int
    status: str
    model: str
    epochs: int | None
    device: str
    artifact_path: str
    map50: float | None
    box_loss: float | None
    latest_prediction_job_id: int | None
    matched: int
    false_positive: int
    false_negative: int
    class_confusion: int
    best_threshold: float | None
    best_f1: float | None


class ProjectExperimentSummary(BaseModel):
    project_id: int
    runs: list[RunComparisonRow]
