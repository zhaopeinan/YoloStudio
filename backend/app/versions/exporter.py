from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.db.models import Annotation, ClassDef, Dataset, DatasetVersion, Image
from app.quality.router import build_quality_summary
from app.storage.visibility import (
    StorageEntityNotFoundError,
    require_active_entity,
)


SPLITS = ("train", "val", "test")
MIN_ANNOTATED_IMAGES = 2


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


def _unique_export_name(image: Image, used_names: set[str]) -> str:
    base = f"ds{image.dataset_id}__{_safe_name(image.relative_path)}"
    if base not in used_names:
        used_names.add(base)
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    index = 2
    while True:
        candidate = f"{stem}__{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _label_name(image_name: str) -> str:
    return Path(image_name).with_suffix(".txt").name


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


def _resolve_datasets(
    db: Session,
    primary_dataset_id: int,
    dataset_ids: list[int] | None,
) -> list[Dataset]:
    requested = list(dict.fromkeys(dataset_ids or [primary_dataset_id]))
    if primary_dataset_id not in requested:
        requested.insert(0, primary_dataset_id)
    else:
        # Keep primary first for FK / naming.
        requested = [primary_dataset_id, *[item for item in requested if item != primary_dataset_id]]

    datasets: list[Dataset] = []
    for dataset_id in requested:
        try:
            require_active_entity(db, "dataset", dataset_id)
        except StorageEntityNotFoundError as exc:
            raise VersionExportError(str(exc)) from exc
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise VersionExportError(f"Dataset was not found: {dataset_id}")
        datasets.append(dataset)

    project_ids = {dataset.project_id for dataset in datasets}
    if len(project_ids) != 1:
        raise VersionExportError("Cannot merge datasets from different projects")
    return datasets


def _validate_dataset_quality(db: Session, dataset: Dataset, *, multi: bool) -> None:
    quality = build_quality_summary(db, dataset.id)
    blockers = []
    if quality.image_count == 0:
        blockers.append("dataset has no images" if not multi else "has no images")
    if quality.class_count == 0:
        blockers.append("project has no active classes")
    if quality.invalid_box_count > 0:
        blockers.append("dataset has invalid boxes" if not multi else "has invalid boxes")
    if quality.duplicate_box_count > 0:
        blockers.append("dataset has duplicate boxes" if not multi else "has duplicate boxes")
    if blockers:
        if multi:
            raise VersionExportError(
                f"Cannot create dataset version for '{dataset.name}': " + ", ".join(blockers)
            )
        raise VersionExportError("Cannot create dataset version: " + ", ".join(blockers))


def create_dataset_version(
    db: Session,
    settings: Settings,
    dataset_id: int,
    name: str | None = None,
    class_ids: list[int] | None = None,
    image_scope: str = "annotated",
    dataset_ids: list[int] | None = None,
) -> DatasetVersion:
    if image_scope not in {"annotated", "all"}:
        raise VersionExportError("Unsupported dataset version image scope")

    datasets = _resolve_datasets(db, dataset_id, dataset_ids)
    primary = datasets[0]
    multi = len(datasets) > 1
    for dataset in datasets:
        _validate_dataset_quality(db, dataset, multi=multi)

    source_dataset_ids = [dataset.id for dataset in datasets]

    requested_class_ids = list(dict.fromkeys(class_ids or []))
    class_query = select(ClassDef).where(
        ClassDef.project_id == primary.project_id,
        ClassDef.active == 1,
    )
    if requested_class_ids:
        class_query = class_query.where(ClassDef.id.in_(requested_class_ids))
    classes = list(db.scalars(class_query.order_by(ClassDef.id)).all())
    if requested_class_ids and {class_def.id for class_def in classes} != set(requested_class_ids):
        raise VersionExportError("Selected classes must belong to the dataset project")
    if not classes:
        raise VersionExportError("Cannot create dataset version: no selected active classes")
    class_mapping = {str(class_def.id): index for index, class_def in enumerate(classes)}
    selected_class_ids = [class_def.id for class_def in classes]

    annotated_images = list(
        db.scalars(
            select(Image)
            .join(Annotation, Annotation.image_id == Image.id)
            .where(
                Image.dataset_id.in_(source_dataset_ids),
                Annotation.class_id.in_(selected_class_ids),
            )
            .group_by(Image.id)
            .order_by(Image.id)
        ).all()
    )
    if image_scope == "all":
        export_images = list(
            db.scalars(
                select(Image)
                .where(Image.dataset_id.in_(source_dataset_ids))
                .order_by(Image.id)
            ).all()
        )
    else:
        export_images = annotated_images

    if len(export_images) < MIN_ANNOTATED_IMAGES:
        if image_scope == "annotated":
            raise VersionExportError(
                "Cannot create dataset version: at least "
                f"{MIN_ANNOTATED_IMAGES} annotated images are required for training and "
                f"validation; current selection has {len(export_images)}."
            )
        raise VersionExportError(
            "Cannot create dataset version: at least "
            f"{MIN_ANNOTATED_IMAGES} images are required for training and validation; "
            f"current selection has {len(export_images)}."
        )
    if not annotated_images:
        raise VersionExportError(
            "Cannot create dataset version: at least one annotated image is required; "
            "current selection has 0."
        )
    splits = _split_images(export_images)

    if name and name.strip():
        version_name = name.strip()
    elif len(datasets) == 1:
        version_name = f"{primary.name}-v1"
    else:
        joined = "+".join(dataset.name for dataset in datasets[:3])
        if len(datasets) > 3:
            joined = f"{joined}+{len(datasets) - 3}more"
        version_name = f"merged-{joined}"

    version = DatasetVersion(
        project_id=primary.project_id,
        dataset_id=primary.id,
        name=version_name,
        class_mapping=class_mapping,
        split_manifest={},
        artifact_path="",
        frozen=1,
    )
    db.add(version)
    db.flush()

    artifact_root = (
        settings.workspace_root / "projects" / str(primary.project_id) / "versions" / str(version.id)
    )
    for split in SPLITS:
        (artifact_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (artifact_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[Annotation]] = {}
    annotations = db.scalars(
        select(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .where(
            Image.dataset_id.in_(source_dataset_ids),
            Annotation.class_id.in_(selected_class_ids),
        )
        .order_by(Annotation.id)
    ).all()
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    used_names: set[str] = set()
    manifest_images = []
    for split, split_images in splits.items():
        for image in split_images:
            image_name = _unique_export_name(image, used_names)
            label_name = _label_name(image_name)
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
                    "dataset_id": image.dataset_id,
                    "source_relative_path": image.relative_path,
                    "export_image": str(Path("images") / split / image_name),
                    "export_label": str(Path("labels") / split / label_name),
                    "split": split,
                    "width": image.width,
                    "height": image.height,
                    "platform": image.platform,
                    "altitude": image.altitude,
                    "timestamp": image.timestamp,
                    "metadata": image.metadata_ or {},
                    "annotations": manifest_annotations,
                }
            )

    split_counts = {split: len(split_images) for split, split_images in splits.items()}
    dataset_summaries = [
        {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "image_count": sum(1 for image in export_images if image.dataset_id == dataset.id),
            "annotated_image_count": sum(
                1 for image in annotated_images if image.dataset_id == dataset.id
            ),
        }
        for dataset in datasets
    ]
    manifest = {
        "version_id": version.id,
        "project_id": primary.project_id,
        "dataset_id": primary.id,
        "dataset_name": primary.name,
        "source_dataset_ids": source_dataset_ids,
        "source_datasets": dataset_summaries,
        "merged": len(source_dataset_ids) > 1,
        "class_mapping": class_mapping,
        "selected_class_ids": selected_class_ids,
        "image_scope": image_scope,
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
