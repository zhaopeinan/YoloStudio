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
    ready_for_training: bool
    issues: list[str]
