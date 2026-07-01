from pathlib import Path

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


class DatasetImageRead(BaseModel):
    id: int
    relative_path: str
    platform: str | None
    altitude: float | None
    timestamp: float | None
    annotation_count: int
    image_url: str


class DatasetImageList(BaseModel):
    items: list[DatasetImageRead]
    limit: int
    offset: int
    total: int
