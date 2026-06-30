from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
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


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project")
    classes: Mapped[list["ClassDef"]] = relationship(back_populates="project")
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="project")
    training_runs: Mapped[list["TrainingRun"]] = relationship(back_populates="project")


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


class RunMetric(TimestampMixin, Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch: Mapped[int | None] = mapped_column(Integer)
    step: Mapped[int | None] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped["TrainingRun"] = relationship(back_populates="metrics")
