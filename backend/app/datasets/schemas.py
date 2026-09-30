from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetScanRequest(BaseModel):
    source_path: Path = Field(..., description="Absolute path to a local dataset zip file or folder")


class DatasetGroupSummary(BaseModel):
    name: str
    image_count: int
    metadata_rows: int
    altitude_min: float | None = None
    altitude_max: float | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None


class DatasetScanSummary(BaseModel):
    source_path: Path
    archive_name: str
    total_files: int
    total_images: int
    total_yolo_labels: int
    total_yaml_files: int
    total_json_files: int
    total_metadata_rows: int
    has_yolo_labels: bool
    has_data_yaml: bool
    groups: list[DatasetGroupSummary]
    warnings: list[str]


class DatasetSourceRead(BaseModel):
    id: int
    original_filename: str
    size_bytes: int
    source_path: str
    created_at: datetime
    updated_at: datetime


class DatasetSourceList(BaseModel):
    items: list[DatasetSourceRead]


class DatasetSourceOption(BaseModel):
    source_ref: str
    label: str
    kind: str
    source_path: str
    original_filename: str
    size_bytes: int
    source_id: int | None = None


class DatasetSourceOptionList(BaseModel):
    items: list[DatasetSourceOption]


class DetectedClassSuggestion(BaseModel):
    name: str
    image_count: int
    color: str
    sample_filenames: list[str] = Field(default_factory=list)


class DetectedClassList(BaseModel):
    dataset_id: int
    total_images: int
    method: str
    items: list[DetectedClassSuggestion]


class DatasetImportRequest(BaseModel):
    source_path: Path = Field(..., description="Absolute path to a local dataset zip file or folder")
    project_name: str = Field(..., min_length=1, max_length=160)
    dataset_name: str = Field(..., min_length=1, max_length=160)


class DatasetImportGroupSummary(BaseModel):
    name: str
    image_count: int
    metadata_rows: int


class DatasetImportSummary(BaseModel):
    project_id: int
    dataset_id: int
    project_name: str
    dataset_name: str
    image_count: int
    groups: list[DatasetImportGroupSummary]


class ProjectDatasetRead(BaseModel):
    id: int
    project_id: int
    name: str
    source_type: str
    import_status: str
    image_count: int
    annotated_image_count: int
    annotation_count: int


class ProjectRead(BaseModel):
    id: int
    name: str
    datasets: list[ProjectDatasetRead]


class ProjectList(BaseModel):
    items: list[ProjectRead]


class DatasetImageRead(BaseModel):
    id: int
    relative_path: str
    width: int | None
    height: int | None
    platform: str | None
    altitude: float | None
    timestamp: float | None
    annotation_count: int
    annotation_status: Literal["unreviewed", "annotated", "negative"]
    image_url: str


class DatasetImageList(BaseModel):
    items: list[DatasetImageRead]
    limit: int
    offset: int
    total: int


class DatasetDimensionRefreshSummary(BaseModel):
    dataset_id: int
    scanned_count: int
    updated_count: int
    missing_count: int


class CoverageBucket(BaseModel):
    label: str
    image_count: int
    annotated_image_count: int
    annotation_count: int


class ClassCoverageBucket(BaseModel):
    class_id: int
    class_name: str
    class_color: str
    image_count: int
    annotation_count: int


class EdgeTagCoverageBucket(BaseModel):
    tag: str
    image_count: int
    annotation_count: int


class DatasetCoverageSummary(BaseModel):
    dataset_id: int
    image_count: int
    annotated_image_count: int
    annotation_count: int
    platforms: list[CoverageBucket]
    altitude_bands: list[CoverageBucket]
    classes: list[ClassCoverageBucket]
    edge_tags: list[EdgeTagCoverageBucket]
