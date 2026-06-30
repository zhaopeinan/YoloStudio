from pathlib import Path

from pydantic import BaseModel, Field


class DatasetScanRequest(BaseModel):
    source_path: Path = Field(..., description="Absolute path to a local dataset zip file")


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
