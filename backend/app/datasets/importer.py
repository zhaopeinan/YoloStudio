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
from app.db.models import Dataset, Image, Project


@dataclass(frozen=True)
class SourceImage:
    name: str
    group: str
    member_path: Path
    local_path: Path | None = None


@dataclass(frozen=True)
class DatasetSourceBundle:
    source_type: str
    source_name: str
    images: list[SourceImage]
    metadata_by_file: dict[tuple[str, str], dict]


def _safe_archive_parts(name: str) -> list[str]:
    return [part for part in Path(name).parts if part not in {"", ".", ".."}]


def _image_relative_member_path(zip_name: str) -> Path:
    parts = _safe_archive_parts(zip_name)
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[index + 1 :])
    return Path(parts[-1])


def _metadata_key(group: str, row: dict) -> tuple[str, str] | None:
    file_name = row.get("file") or row.get("filename") or row.get("image")
    if not isinstance(file_name, str) or not file_name:
        return None
    return group, Path(file_name).name


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
        group = _collection_name(name)
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


def _png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)
    return None, None


def _image_size(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() == ".png":
        return _png_size(path)
    return None, None


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
            db.add(
                Image(
                    dataset_id=dataset.id,
                    relative_path=str(relative_path),
                    width=width,
                    height=height,
                    platform=metadata.get("drone") or metadata.get("platform") or group,
                    altitude=_safe_float(metadata.get("z") or metadata.get("altitude")),
                    timestamp=_safe_float(metadata.get("t") or metadata.get("timestamp")),
                    metadata_=metadata,
                )
            )
            group_counts[source_image.group] += 1

        dataset.image_count = sum(group_counts.values())
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
