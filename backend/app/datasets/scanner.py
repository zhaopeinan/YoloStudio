from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.datasets.schemas import DatasetGroupSummary, DatasetScanSummary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YAML_NAMES = {"data.yaml", "dataset.yaml"}


def _collection_name(zip_name: str) -> str:
    parts = [part for part in zip_name.split("/") if part]
    if "images" in parts:
        index = parts.index("images")
        if index > 0:
            return parts[index - 1]
    if len(parts) >= 2:
        return parts[1]
    return "ungrouped"


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_metadata_lines(lines: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_parse_error": line})
    return rows


def _scan_metadata_rows(archive: ZipFile, name: str) -> list[dict]:
    with archive.open(name) as handle:
        return _parse_metadata_lines(raw_line.decode("utf-8") for raw_line in handle)


def _scan_folder_metadata_rows(source_path: Path, name: str) -> list[dict]:
    with (source_path / name).open("r", encoding="utf-8") as handle:
        return _parse_metadata_lines(handle)


def _build_scan_summary(
    source_path: Path,
    source_name: str,
    names: list[str],
    load_metadata_rows: Callable[[str], list[dict]],
) -> DatasetScanSummary:
    image_names = [name for name in names if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
    label_names = [name for name in names if Path(name).suffix.lower() == ".txt"]
    yaml_names = [name for name in names if Path(name).name.lower() in YAML_NAMES]
    json_names = [name for name in names if Path(name).suffix.lower() in {".json", ".jsonl"}]

    image_counts: dict[str, int] = defaultdict(int)
    metadata_rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for image_name in image_names:
        image_counts[_collection_name(image_name)] += 1

    warnings: list[str] = []
    for json_name in json_names:
        if Path(json_name).name != "meta.jsonl":
            continue
        group_name = _collection_name(json_name)
        rows = load_metadata_rows(json_name)
        metadata_rows_by_group[group_name].extend(rows)
        parse_errors = [row for row in rows if "_parse_error" in row]
        if parse_errors:
            warnings.append(f"{json_name} contains {len(parse_errors)} unparsable rows")

    group_names = sorted(set(image_counts) | set(metadata_rows_by_group))
    groups: list[DatasetGroupSummary] = []
    for group_name in group_names:
        rows = metadata_rows_by_group.get(group_name, [])
        altitudes = [
            altitude
            for altitude in (_safe_float(row.get("z")) for row in rows)
            if altitude is not None
        ]
        timestamps = [
            timestamp
            for timestamp in (_safe_float(row.get("t")) for row in rows)
            if timestamp is not None
        ]
        groups.append(
            DatasetGroupSummary(
                name=group_name,
                image_count=image_counts.get(group_name, 0),
                metadata_rows=len(rows),
                altitude_min=min(altitudes) if altitudes else None,
                altitude_max=max(altitudes) if altitudes else None,
                first_timestamp=min(timestamps) if timestamps else None,
                last_timestamp=max(timestamps) if timestamps else None,
            )
        )

    if not label_names:
        warnings.append("No YOLO label .txt files were found")
    if not yaml_names:
        warnings.append("No data.yaml or dataset.yaml file was found")

    return DatasetScanSummary(
        source_path=source_path,
        archive_name=source_name,
        total_files=len(names),
        total_images=len(image_names),
        total_yolo_labels=len(label_names),
        total_yaml_files=len(yaml_names),
        total_json_files=len(json_names),
        total_metadata_rows=sum(group.metadata_rows for group in groups),
        has_yolo_labels=bool(label_names),
        has_data_yaml=bool(yaml_names),
        groups=groups,
        warnings=warnings,
    )


def scan_dataset_zip(source_path: Path) -> DatasetScanSummary:
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file, got {source_path.name}")

    try:
        archive = ZipFile(source_path)
    except BadZipFile as exc:
        raise ValueError(f"Invalid zip archive: {source_path}") from exc

    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        return _build_scan_summary(
            source_path,
            source_path.name,
            names,
            lambda name: _scan_metadata_rows(archive, name),
        )


def scan_dataset_folder(source_path: Path) -> DatasetScanSummary:
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not source_path.is_dir():
        raise ValueError(f"Expected a directory, got {source_path.name}")

    paths = sorted(path for path in source_path.rglob("*") if path.is_file())
    names = [path.relative_to(source_path).as_posix() for path in paths]
    return _build_scan_summary(
        source_path,
        source_path.name,
        names,
        lambda name: _scan_folder_metadata_rows(source_path, name),
    )


def scan_dataset_source(source_path: Path) -> DatasetScanSummary:
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir():
        return scan_dataset_folder(source_path)
    if source_path.suffix.lower() == ".zip":
        return scan_dataset_zip(source_path)
    raise ValueError(f"Expected a .zip file or directory, got {source_path.name}")
