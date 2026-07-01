from __future__ import annotations

from pydantic import BaseModel


class DatasetQualitySummary(BaseModel):
    dataset_id: int
    image_count: int
    annotated_image_count: int
    unannotated_image_count: int
    annotation_count: int
    class_count: int
    tiny_box_count: int
    invalid_box_count: int
    duplicate_box_count: int
    missing_metadata_count: int
    unknown_class_reference_count: int
    ready_for_training: bool
    issues: list[str]


class DatasetQualityIssue(BaseModel):
    issue_type: str
    severity: str
    message: str
    image_id: int
    image_path: str
    image_url: str
    annotation_id: int | None = None
    class_id: int | None = None
    class_name: str | None = None
    x_center: float | None = None
    y_center: float | None = None
    width: float | None = None
    height: float | None = None


class DatasetQualityIssueList(BaseModel):
    dataset_id: int
    limit: int
    offset: int
    total: int
    items: list[DatasetQualityIssue]
