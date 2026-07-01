from __future__ import annotations

import json
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.datasets.scanner import IMAGE_EXTENSIONS, _collection_name, _safe_float
from app.datasets.schemas import (
    DatasetImportGroupSummary,
    DatasetImportRequest,
    DatasetImportSummary,
)
from app.db.models import Annotation, ClassDef, Dataset, Image, Project


DEFAULT_CLASS_COLORS = [
    "#2f80ed",
    "#27ae60",
    "#eb5757",
    "#f2994a",
    "#9b51e0",
    "#00a3a3",
]


@dataclass(frozen=True)
class SourceImage:
    name: str
    group: str
    member_path: Path
    local_path: Path | None = None


@dataclass(frozen=True)
class SourceAnnotation:
    class_index: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class DatasetSourceBundle:
    source_type: str
    source_name: str
    images: list[SourceImage]
    metadata_by_file: dict[tuple[str, str], dict]
    class_names: list[str]
    labels_by_image_key: dict[tuple[str, str], list[SourceAnnotation]]


def _safe_archive_parts(name: str) -> list[str]:
    return [part for part in Path(name).parts if part not in {"", ".", ".."}]


def _image_relative_member_path(zip_name: str) -> Path:
    parts = _safe_archive_parts(zip_name)
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[index + 1 :])
    return Path(parts[-1])


def _label_relative_member_path(label_name: str) -> Path:
    parts = _safe_archive_parts(label_name)
    if "labels" in parts:
        index = parts.index("labels")
        return Path(*parts[index + 1 :]).with_suffix("")
    return Path(parts[-1]).with_suffix("")


def _label_collection_name(label_name: str) -> str:
    parts = [part for part in label_name.split("/") if part]
    if "labels" in parts:
        index = parts.index("labels")
        if index > 0:
            return parts[index - 1]
    return _collection_name(label_name)


def _metadata_key(group: str, row: dict) -> tuple[str, str] | None:
    file_name = row.get("file") or row.get("filename") or row.get("image")
    if not isinstance(file_name, str) or not file_name:
        return None
    return group, Path(file_name).name


def _strip_yaml_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _parse_inline_names(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_strip_yaml_scalar(item) for item in body.split(",") if _strip_yaml_scalar(item)]
    if value.startswith("{") and value.endswith("}"):
        entries: dict[int, str] = {}
        body = value[1:-1].strip()
        if not body:
            return []
        for item in body.split(","):
            if ":" not in item:
                continue
            raw_key, raw_name = item.split(":", 1)
            try:
                key = int(_strip_yaml_scalar(raw_key))
            except ValueError:
                continue
            name = _strip_yaml_scalar(raw_name)
            if name:
                entries[key] = name
        return [entries[index] for index in sorted(entries)]
    return []


def _parse_class_names_from_yaml(text: str) -> list[str]:
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.startswith("names:"):
            continue
        inline = stripped.split(":", 1)[1].strip()
        if inline:
            return _parse_inline_names(inline)

        entries: dict[int, str] = {}
        list_items: list[str] = []
        for child_line in lines[index + 1 :]:
            if not child_line.startswith((" ", "\t")):
                break
            child = child_line.strip()
            if not child:
                continue
            if child.startswith("- "):
                name = _strip_yaml_scalar(child[2:])
                if name:
                    list_items.append(name)
                continue
            if ":" not in child:
                continue
            raw_key, raw_name = child.split(":", 1)
            try:
                key = int(_strip_yaml_scalar(raw_key))
            except ValueError:
                continue
            name = _strip_yaml_scalar(raw_name)
            if name:
                entries[key] = name
        if entries:
            return [entries[key] for key in sorted(entries)]
        return list_items
    return []


def _parse_yolo_label_lines(lines: list[str]) -> list[SourceAnnotation]:
    annotations: list[SourceAnnotation] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        if class_index < 0:
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            continue
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            continue
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            continue
        annotations.append(
            SourceAnnotation(
                class_index=class_index,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return annotations


def _metadata_from_lines(group: str, lines: list[str]) -> dict[tuple[str, str], dict]:
    metadata: dict[tuple[str, str], dict] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = _metadata_key(group, row)
        if key:
            metadata[key] = row
    return metadata


def _load_archive_metadata_by_group_and_file(archive: ZipFile) -> dict[tuple[str, str], dict]:
    metadata: dict[tuple[str, str], dict] = {}
    for name in archive.namelist():
        if Path(name).name != "meta.jsonl" or name.endswith("/"):
            continue
        group = _label_collection_name(name)
        with archive.open(name) as handle:
            lines = [raw_line.decode("utf-8") for raw_line in handle]
        metadata.update(_metadata_from_lines(group, lines))
    return metadata


def _load_folder_metadata_by_group_and_file(source_path: Path) -> dict[tuple[str, str], dict]:
    metadata: dict[tuple[str, str], dict] = {}
    for path in sorted(source_path.rglob("meta.jsonl")):
        if not path.is_file():
            continue
        name = path.relative_to(source_path).as_posix()
        group = _collection_name(name)
        metadata.update(_metadata_from_lines(group, path.read_text(encoding="utf-8").splitlines()))
    return metadata


def _label_keys(group: str, member_path: Path) -> list[tuple[str, str]]:
    stem_path = member_path.with_suffix("").as_posix()
    return [
        (group, stem_path),
        (group, member_path.name),
        ("__any__", stem_path),
        ("__any__", member_path.name),
    ]


def _load_archive_class_names(archive: ZipFile) -> list[str]:
    for name in archive.namelist():
        if Path(name).name.lower() not in {"data.yaml", "dataset.yaml"} or name.endswith("/"):
            continue
        with archive.open(name) as handle:
            return _parse_class_names_from_yaml(handle.read().decode("utf-8"))
    return []


def _load_folder_class_names(source_path: Path) -> list[str]:
    for yaml_name in ("data.yaml", "dataset.yaml"):
        path = next(
            (candidate for candidate in source_path.rglob(yaml_name) if candidate.is_file()),
            None,
        )
        if path is not None:
            return _parse_class_names_from_yaml(path.read_text(encoding="utf-8"))
    return []


def _load_archive_labels_by_image_key(
    archive: ZipFile,
) -> dict[tuple[str, str], list[SourceAnnotation]]:
    labels: dict[tuple[str, str], list[SourceAnnotation]] = defaultdict(list)
    for name in archive.namelist():
        if name.endswith("/") or Path(name).suffix.lower() != ".txt":
            continue
        if "labels" not in _safe_archive_parts(name):
            continue
        group = _collection_name(name)
        member_path = _label_relative_member_path(name)
        with archive.open(name) as handle:
            annotations = _parse_yolo_label_lines(
                [raw_line.decode("utf-8") for raw_line in handle]
            )
        for key in _label_keys(group, member_path):
            labels[key].extend(annotations)
    return dict(labels)


def _load_folder_labels_by_image_key(
    source_path: Path,
) -> dict[tuple[str, str], list[SourceAnnotation]]:
    labels: dict[tuple[str, str], list[SourceAnnotation]] = defaultdict(list)
    for path in sorted(source_path.rglob("*.txt")):
        if not path.is_file():
            continue
        relative_name = path.relative_to(source_path).as_posix()
        if "labels" not in _safe_archive_parts(relative_name):
            continue
        group = _label_collection_name(relative_name)
        member_path = _label_relative_member_path(relative_name)
        annotations = _parse_yolo_label_lines(path.read_text(encoding="utf-8").splitlines())
        for key in _label_keys(group, member_path):
            labels[key].extend(annotations)
    return dict(labels)


def _png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)
    return None, None


def _jpeg_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as handle:
            data = handle.read()
    except OSError:
        return None, None
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None, None

    offset = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset < len(data):
        while offset < len(data) and data[offset] != 0xFF:
            offset += 1
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return None, None

        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            return None, None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            return None, None
        if offset + 2 > len(data):
            return None, None

        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            return None, None
        if marker in sof_markers:
            if segment_length < 7:
                return None, None
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_length
    return None, None


def _image_size(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() == ".png":
        return _png_size(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return _jpeg_size(path)
    return None, None


def read_image_dimensions(path: Path) -> tuple[int | None, int | None]:
    return _image_size(path)


def _open_zip_source(source_path: Path) -> tuple[ZipFile, DatasetSourceBundle]:
    try:
        archive = ZipFile(source_path)
    except BadZipFile as exc:
        raise ValueError(f"Invalid zip archive: {source_path}") from exc

    names = [name for name in archive.namelist() if not name.endswith("/")]
    image_names = [name for name in names if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
    images = [
        SourceImage(
            name=name,
            group=_collection_name(name),
            member_path=_image_relative_member_path(name),
        )
        for name in image_names
    ]
    return archive, DatasetSourceBundle(
        source_type="zip",
        source_name=source_path.name,
        images=images,
        metadata_by_file=_load_archive_metadata_by_group_and_file(archive),
        class_names=_load_archive_class_names(archive),
        labels_by_image_key=_load_archive_labels_by_image_key(archive),
    )


def _load_folder_source(source_path: Path) -> DatasetSourceBundle:
    paths = sorted(path for path in source_path.rglob("*") if path.is_file())
    images: list[SourceImage] = []
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = path.relative_to(source_path).as_posix()
        images.append(
            SourceImage(
                name=name,
                group=_collection_name(name),
                member_path=_image_relative_member_path(name),
                local_path=path,
            )
        )

    return DatasetSourceBundle(
        source_type="folder",
        source_name=source_path.name,
        images=images,
        metadata_by_file=_load_folder_metadata_by_group_and_file(source_path),
        class_names=_load_folder_class_names(source_path),
        labels_by_image_key=_load_folder_labels_by_image_key(source_path),
    )


def _copy_source_image(archive: ZipFile | None, image: SourceImage, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if image.local_path is not None:
        shutil.copy2(image.local_path, destination)
        return
    if archive is None:
        raise ValueError(f"Image source is unavailable for {image.name}")
    with archive.open(image.name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def _ensure_imported_classes(
    db: Session,
    project_id: int,
    class_names: list[str],
) -> dict[int, ClassDef]:
    class_by_name = {
        class_def.name: class_def
        for class_def in db.scalars(
            select(ClassDef).where(ClassDef.project_id == project_id).order_by(ClassDef.id)
        ).all()
    }
    next_color_index = len(class_by_name)
    class_by_index: dict[int, ClassDef] = {}
    for class_index, raw_name in enumerate(class_names):
        name = raw_name.strip()
        if not name:
            continue
        class_def = class_by_name.get(name)
        if class_def is None:
            class_def = ClassDef(
                project_id=project_id,
                name=name,
                color=DEFAULT_CLASS_COLORS[next_color_index % len(DEFAULT_CLASS_COLORS)],
                active=1,
            )
            next_color_index += 1
            db.add(class_def)
            db.flush()
            class_by_name[name] = class_def
        class_by_index[class_index] = class_def
    return class_by_index


def _image_label_keys(source_image: SourceImage) -> list[tuple[str, str]]:
    stem_path = source_image.member_path.with_suffix("").as_posix()
    return [
        (source_image.group, stem_path),
        (source_image.group, source_image.member_path.stem),
        (source_image.group, source_image.member_path.name),
        ("__any__", stem_path),
        ("__any__", source_image.member_path.stem),
        ("__any__", source_image.member_path.name),
    ]


def _annotations_for_source_image(
    bundle: DatasetSourceBundle,
    source_image: SourceImage,
) -> list[SourceAnnotation]:
    for key in _image_label_keys(source_image):
        annotations = bundle.labels_by_image_key.get(key)
        if annotations:
            return annotations
    return []


def import_dataset(
    db: Session,
    settings: Settings,
    request: DatasetImportRequest,
) -> DatasetImportSummary:
    source_path = request.source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir():
        archive = None
        bundle = _load_folder_source(source_path)
    elif source_path.suffix.lower() == ".zip":
        archive, bundle = _open_zip_source(source_path)
    else:
        raise ValueError(f"Expected a .zip file or directory, got {source_path.name}")

    settings.ensure_workspace()
    try:
        project = db.scalar(select(Project).where(Project.name == request.project_name))
        if project is None:
            project = Project(name=request.project_name, workspace_path="")
            db.add(project)
            db.flush()
            project.workspace_path = str(Path("projects") / str(project.id))
        class_by_index = _ensure_imported_classes(db, project.id, bundle.class_names)

        dataset = Dataset(
            project_id=project.id,
            name=request.dataset_name,
            source_type=bundle.source_type,
            import_status="imported",
            image_count=0,
            metadata_={
                "source_path": str(source_path),
                "source_name": bundle.source_name,
                "source_type": bundle.source_type,
            },
        )
        db.add(dataset)
        db.flush()

        group_counts: dict[str, int] = defaultdict(int)
        metadata_counts: dict[str, int] = defaultdict(int)
        import_warnings: list[dict] = []
        for group, _ in bundle.metadata_by_file:
            metadata_counts[group] += 1

        image_root = (
            settings.workspace_root
            / "projects"
            / str(project.id)
            / "datasets"
            / str(dataset.id)
            / "images"
        )
        for source_image in bundle.images:
            destination = image_root / source_image.group / source_image.member_path
            _copy_source_image(archive, source_image, destination)

            metadata = bundle.metadata_by_file.get((source_image.group, Path(source_image.name).name), {})
            width, height = _image_size(destination)
            relative_path = destination.relative_to(settings.workspace_root)
            image = Image(
                dataset_id=dataset.id,
                relative_path=str(relative_path),
                width=width,
                height=height,
                platform=metadata.get("drone") or metadata.get("platform") or source_image.group,
                altitude=_safe_float(metadata.get("z") or metadata.get("altitude")),
                timestamp=_safe_float(metadata.get("t") or metadata.get("timestamp")),
                metadata_=metadata,
            )
            db.add(image)
            db.flush()
            unknown_class_counts: dict[int, int] = defaultdict(int)
            for source_annotation in _annotations_for_source_image(bundle, source_image):
                class_def = class_by_index.get(source_annotation.class_index)
                if class_def is None:
                    unknown_class_counts[source_annotation.class_index] += 1
                    continue
                db.add(
                    Annotation(
                        image_id=image.id,
                        class_id=class_def.id,
                        x_center=source_annotation.x_center,
                        y_center=source_annotation.y_center,
                        width=source_annotation.width,
                        height=source_annotation.height,
                        edge_tags=[],
                    )
                )
            for class_index, count in sorted(unknown_class_counts.items()):
                import_warnings.append(
                    {
                        "type": "unknown_class_reference",
                        "image_id": image.id,
                        "image_path": str(relative_path),
                        "source_image": source_image.name,
                        "class_index": class_index,
                        "count": count,
                    }
                )
            group_counts[source_image.group] += 1

        dataset.image_count = sum(group_counts.values())
        if import_warnings:
            dataset.metadata_ = {**dataset.metadata_, "import_warnings": import_warnings}
        db.commit()
        db.refresh(project)
        db.refresh(dataset)
    finally:
        if archive is not None:
            archive.close()

    groups = [
        DatasetImportGroupSummary(
            name=group,
            image_count=group_counts[group],
            metadata_rows=metadata_counts.get(group, 0),
        )
        for group in sorted(set(group_counts) | set(metadata_counts))
    ]
    return DatasetImportSummary(
        project_id=project.id,
        dataset_id=dataset.id,
        project_name=project.name,
        dataset_name=dataset.name,
        image_count=dataset.image_count,
        groups=groups,
    )
