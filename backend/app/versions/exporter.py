from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.db.models import Annotation, ClassDef, Dataset, DatasetVersion, Image
from app.quality.router import build_quality_summary


SPLITS = ("train", "val", "test")


class VersionExportError(ValueError):
    pass


def _split_images(images: list[Image]) -> dict[str, list[Image]]:
    ordered = sorted(images, key=lambda image: image.id)
    total = len(ordered)
    if total == 0:
        return {split: [] for split in SPLITS}
    if total == 1:
        return {"train": ordered, "val": [], "test": []}
    if total == 2:
        return {"train": ordered[:1], "val": ordered[1:], "test": []}

    train_count = max(1, int(total * 0.8))
    val_count = max(1, int(total * 0.1))
    if train_count + val_count > total:
        train_count = total - val_count
    test_count = total - train_count - val_count
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count : train_count + val_count + test_count],
    }


def _safe_name(path: str) -> str:
    source = Path(path)
    parts = [part for part in source.with_suffix("").parts if part not in {"", ".", ".."}]
    stem = "__".join(parts[-4:]) if parts else source.stem
    return f"{stem}{source.suffix.lower()}"


def _label_name(image: Image) -> str:
    return Path(_safe_name(image.relative_path)).with_suffix(".txt").name


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _write_data_yaml(artifact_root: Path, classes: list[ClassDef]) -> None:
    lines = [
        f"path: {artifact_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, class_def in enumerate(classes):
        lines.append(f"  {index}: {class_def.name}")
    (artifact_root / "data.yaml").write_text("\n".join(lines) + "\n")


def _copy_image(settings: Settings, image: Image, destination: Path) -> None:
    source = (settings.workspace_root / image.relative_path).resolve()
    workspace_root = settings.workspace_root.resolve()
    if not source.is_relative_to(workspace_root) or not source.exists():
        raise VersionExportError(f"Image file is missing for image {image.id}")
    shutil.copy2(source, destination)


def create_dataset_version(
    db: Session,
    settings: Settings,
    dataset_id: int,
    name: str | None = None,
    class_ids: list[int] | None = None,
) -> DatasetVersion:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise VersionExportError("Dataset was not found")

    quality = build_quality_summary(db, dataset_id)
    blockers = []
    if quality.image_count == 0:
        blockers.append("dataset has no images")
    if quality.class_count == 0:
        blockers.append("project has no active classes")
    if quality.annotation_count == 0:
        blockers.append("dataset has no saved annotations")
    if quality.invalid_box_count > 0:
        blockers.append("dataset has invalid boxes")
    if quality.duplicate_box_count > 0:
        blockers.append("dataset has duplicate boxes")
    if blockers:
        raise VersionExportError("Cannot create dataset version: " + ", ".join(blockers))

    requested_class_ids = list(dict.fromkeys(class_ids or []))
    class_query = select(ClassDef).where(
        ClassDef.project_id == dataset.project_id,
        ClassDef.active == 1,
    )
    if requested_class_ids:
        class_query = class_query.where(ClassDef.id.in_(requested_class_ids))
    classes = db.scalars(class_query.order_by(ClassDef.id)).all()
    if requested_class_ids and {class_def.id for class_def in classes} != set(requested_class_ids):
        raise VersionExportError("Selected classes must belong to the dataset project")
    if not classes:
        raise VersionExportError("Cannot create dataset version: no selected active classes")
    class_mapping = {str(class_def.id): index for index, class_def in enumerate(classes)}
    selected_class_ids = [class_def.id for class_def in classes]

    annotated_images = db.scalars(
        select(Image)
        .join(Annotation, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset_id, Annotation.class_id.in_(selected_class_ids))
        .group_by(Image.id)
        .order_by(Image.id)
    ).all()
    if not annotated_images:
        raise VersionExportError("Cannot create dataset version: selected classes have no annotations")
    splits = _split_images(annotated_images)

    version = DatasetVersion(
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        name=name.strip() if name and name.strip() else f"{dataset.name}-v1",
        class_mapping=class_mapping,
        split_manifest={},
        artifact_path="",
        frozen=1,
    )
    db.add(version)
    db.flush()

    artifact_root = (
        settings.workspace_root / "projects" / str(dataset.project_id) / "versions" / str(version.id)
    )
    for split in SPLITS:
        (artifact_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (artifact_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[Annotation]] = {}
    annotations = db.scalars(
        select(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.dataset_id == dataset.id, Annotation.class_id.in_(selected_class_ids))
        .order_by(Annotation.id)
    ).all()
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    manifest_images = []
    for split, split_images in splits.items():
        for image in split_images:
            image_name = _safe_name(image.relative_path)
            label_name = _label_name(image)
            _copy_image(settings, image, artifact_root / "images" / split / image_name)

            image_annotations = annotations_by_image.get(image.id, [])
            label_lines = []
            manifest_annotations = []
            for annotation in image_annotations:
                yolo_class = class_mapping[str(annotation.class_id)]
                label_lines.append(
                    " ".join(
                        [
                            str(yolo_class),
                            _format_float(annotation.x_center),
                            _format_float(annotation.y_center),
                            _format_float(annotation.width),
                            _format_float(annotation.height),
                        ]
                    )
                )
                manifest_annotations.append(
                    {
                        "annotation_id": annotation.id,
                        "class_id": annotation.class_id,
                        "yolo_class": yolo_class,
                        "track_id": annotation.track_id,
                        "edge_tags": annotation.edge_tags or [],
                    }
                )

            (artifact_root / "labels" / split / label_name).write_text(
                "\n".join(label_lines) + "\n"
            )
            manifest_images.append(
                {
                    "image_id": image.id,
                    "source_relative_path": image.relative_path,
                    "export_image": str(Path("images") / split / image_name),
                    "export_label": str(Path("labels") / split / label_name),
                    "split": split,
                    "annotations": manifest_annotations,
                }
            )

    split_counts = {split: len(split_images) for split, split_images in splits.items()}
    manifest = {
        "version_id": version.id,
        "project_id": dataset.project_id,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "class_mapping": class_mapping,
        "selected_class_ids": selected_class_ids,
        "split_counts": split_counts,
        "images": manifest_images,
    }
    _write_data_yaml(artifact_root, classes)
    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    version.artifact_path = str(artifact_root)
    version.split_manifest = manifest
    db.commit()
    db.refresh(version)
    return version
