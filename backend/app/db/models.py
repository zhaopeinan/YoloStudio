from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project")
    classes: Mapped[list["ClassDef"]] = relationship(back_populates="project")
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="project")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="project")


class TrashItem(TimestampMixin, Base):
    __tablename__ = "trash_items"
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_trash_items_entity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(Integer, index=True)
    version_id: Mapped[int | None] = mapped_column(Integer, index=True)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    trash_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    purge_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    import_status: Mapped[str] = mapped_column(String(40), nullable=False, default="scanned")
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    project: Mapped["Project"] = relationship(back_populates="datasets")
    images: Mapped[list["Image"]] = relationship(back_populates="dataset")
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset")


class Image(TimestampMixin, Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    platform: Mapped[str | None] = mapped_column(String(80), index=True)
    altitude: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    dataset: Mapped["Dataset"] = relationship(back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="image")


class ClassDef(TimestampMixin, Base):
    __tablename__ = "class_defs"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_class_defs_project_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    color: Mapped[str] = mapped_column(String(24), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    project: Mapped["Project"] = relationship(back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="class_def")


class Annotation(TimestampMixin, Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("class_defs.id"), nullable=False, index=True)
    x_center: Mapped[float] = mapped_column(Float, nullable=False)
    y_center: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    track_id: Mapped[str | None] = mapped_column(String(120))
    edge_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    image: Mapped["Image"] = relationship(back_populates="annotations")
    class_def: Mapped["ClassDef"] = relationship(back_populates="annotations")


class DatasetVersion(TimestampMixin, Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    class_mapping: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    split_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    frozen: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    project: Mapped["Project"] = relationship(back_populates="versions")
    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="version")


class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="queued", index=True)
    device: Mapped[str] = mapped_column(String(40), nullable=False, default="cpu")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped["Project"] = relationship(back_populates="training_runs")
    version: Mapped["DatasetVersion"] = relationship(back_populates="training_runs")
    metrics: Mapped[list["RunMetric"]] = relationship(back_populates="run")
    prediction_jobs: Mapped[list["PredictionJob"]] = relationship(back_populates="run")
    exports: Mapped[list["ExportArtifact"]] = relationship(back_populates="run")


class RunMetric(TimestampMixin, Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch: Mapped[int | None] = mapped_column(Integer)
    step: Mapped[int | None] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped["TrainingRun"] = relationship(back_populates="metrics")


class PredictionJob(TimestampMixin, Base):
    __tablename__ = "prediction_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="queued", index=True)
    image_scope: Mapped[str] = mapped_column(String(40), nullable=False, default="all")
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prediction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped["TrainingRun"] = relationship(back_populates="prediction_jobs")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="job")


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("prediction_jobs.id"), nullable=False, index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("class_defs.id"), nullable=False, index=True)
    x_center: Mapped[float] = mapped_column(Float, nullable=False)
    y_center: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    matched_annotation_id: Mapped[int | None] = mapped_column(ForeignKey("annotations.id"))
    failure_type: Mapped[str] = mapped_column(String(40), nullable=False)

    job: Mapped["PredictionJob"] = relationship(back_populates="predictions")


class ExportArtifact(TimestampMixin, Base):
    __tablename__ = "export_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    format: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="queued", index=True)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    run: Mapped["TrainingRun"] = relationship(back_populates="exports")


class PreviewVideo(TimestampMixin, Base):
    __tablename__ = "preview_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(240), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    jobs: Mapped[list["PreviewJob"]] = relationship(back_populates="video")


class ModelWeight(TimestampMixin, Base):
    __tablename__ = "model_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(240), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class DatasetSource(TimestampMixin, Base):
    """Uploaded dataset zip archives available for scan/import."""

    __tablename__ = "dataset_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(240), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class PreviewJob(TimestampMixin, Base):
    __tablename__ = "preview_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("training_runs.id"), index=True)
    video_id: Mapped[int | None] = mapped_column(ForeignKey("preview_videos.id"), index=True)
    model_ref: Mapped[str] = mapped_column(String(240), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, default="video")
    source_filename: Mapped[str] = mapped_column(String(240), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_path: Mapped[str | None] = mapped_column(Text)
    frame_directory: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="queued", index=True)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    frame_step: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    fps: Mapped[float | None] = mapped_column(Float)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    video: Mapped[PreviewVideo | None] = relationship(back_populates="jobs")