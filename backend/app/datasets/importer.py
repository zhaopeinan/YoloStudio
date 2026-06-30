from __future__ import annotations

import json
import shutil
import struct
from collections import defaultdict
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


def _load_metadata_by_group_and_file(archive: ZipFile) -> dict[tuple[str, str], dict]:
    metadata: dict[tuple[str, str], dict] = {}
    for name in archive.namelist():
        if Path(name).name != "meta.jsonl" or name.endswith("/"):
            continue
        group = _collection_name(name)
        with archive.open(name) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
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


def import_dataset(
    db: Session,
    settings: Settings,
    request: DatasetImportRequest,
) -> DatasetImportSummary:
    source_path = request.source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file, got {source_path.name}")

    try:
        archive = ZipFile(source_path)
    except BadZipFile as exc:
        raise ValueError(f"Invalid zip archive: {source_path}") from exc

    settings.ensure_workspace()
    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        image_names = [name for name in names if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
        metadata_by_file = _load_metadata_by_group_and_file(archive)

        project = db.scalar(select(Project).where(Project.name == request.project_name))
        if project is None:
            project = Project(name=request.project_name, workspace_path="")
            db.add(project)
            db.flush()
            project.workspace_path = str(Path("projects") / str(project.id))

        dataset = Dataset(
            project_id=project.id,
            name=request.dataset_name,
            source_type="zip",
            import_status="imported",
            image_count=0,
            metadata_={"source_path": str(source_path), "archive_name": source_path.name},
        )
        db.add(dataset)
        db.flush()

        group_counts: dict[str, int] = defaultdict(int)
        metadata_counts: dict[str, int] = defaultdict(int)
        for group, _ in metadata_by_file:
            metadata_counts[group] += 1

        image_root = (
            settings.workspace_root
            / "projects"
            / str(project.id)
            / "datasets"
            / str(dataset.id)
            / "images"
        )
        for zip_name in image_names:
            group = _collection_name(zip_name)
            member_path = _image_relative_member_path(zip_name)
            destination = image_root / group / member_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(zip_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

            metadata = metadata_by_file.get((group, Path(zip_name).name), {})
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
            group_counts[group] += 1

        dataset.image_count = sum(group_counts.values())
        db.commit()
        db.refresh(project)
        db.refresh(dataset)

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
