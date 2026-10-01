from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import REPO_ROOT, Settings
from app.db.models import DatasetSource


ZIP_EXTENSIONS = {".zip"}
MAX_DATASET_ZIP_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB
CHUNK_SIZE = 8 * 1024 * 1024


def dataset_source_root(settings: Settings, source_id: int) -> Path:
    return settings.workspace_root / "dataset-sources" / str(source_id)


def _validate_zip_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ZIP_EXTENSIONS:
        raise ValueError("仅支持上传 .zip 数据集压缩包")
    return suffix


def _known_server_zips() -> list[Path]:
    candidates = [
        REPO_ROOT / "image_dataset.zip",
        Path("/opt/YoloStudio/image_dataset.zip"),
    ]
    seen: set[Path] = set()
    result: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


async def create_dataset_source(
    db: Session,
    settings: Settings,
    source_filename: str,
    upload: UploadFile,
) -> DatasetSource:
    suffix = _validate_zip_filename(source_filename)
    source = DatasetSource(
        original_filename=Path(source_filename).name,
        stored_path="",
        size_bytes=0,
    )
    db.add(source)
    db.flush()
    root = dataset_source_root(settings, source.id)
    root.mkdir(parents=True, exist_ok=True)
    stored_path = root / f"dataset{suffix}"
    total = 0
    try:
        with stored_path.open("wb") as handle:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DATASET_ZIP_BYTES:
                    raise ValueError("数据集压缩包不能超过 20 GB")
                handle.write(chunk)
        if total == 0:
            raise ValueError("数据集压缩包为空")
        try:
            with ZipFile(stored_path) as archive:
                if archive.testzip() is not None:
                    raise ValueError("ZIP 压缩包已损坏")
        except BadZipFile as exc:
            raise ValueError("无效的 ZIP 压缩包") from exc
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        db.rollback()
        raise

    source.stored_path = str(stored_path)
    source.size_bytes = total
    db.commit()
    db.refresh(source)
    return source


def list_dataset_sources(db: Session) -> list[DatasetSource]:
    return list(db.scalars(select(DatasetSource).order_by(DatasetSource.id.desc())).all())


def delete_dataset_source(db: Session, settings: Settings, source_id: int) -> None:
    source = db.get(DatasetSource, source_id)
    if source is None:
        raise LookupError("数据集压缩包不存在")
    root = dataset_source_root(settings, source.id)
    db.delete(source)
    db.commit()
    shutil.rmtree(root, ignore_errors=True)


def list_dataset_source_options(db: Session) -> list[dict]:
    items: list[dict] = []
    seen_paths: set[str] = set()

    for path in _known_server_zips():
        path_str = str(path)
        seen_paths.add(path_str)
        items.append(
            {
                "source_ref": f"server:{path_str}",
                "label": f"服务器文件 · {path.name}",
                "kind": "server",
                "source_path": path_str,
                "original_filename": path.name,
                "size_bytes": path.stat().st_size,
                "source_id": None,
            }
        )

    for source in list_dataset_sources(db):
        path = Path(source.stored_path)
        if not path.is_file():
            continue
        path_str = str(path.resolve()) if path.exists() else source.stored_path
        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)
        items.append(
            {
                "source_ref": f"upload:{source.id}",
                "label": f"已上传 · {source.original_filename}",
                "kind": "upload",
                "source_path": source.stored_path,
                "original_filename": source.original_filename,
                "size_bytes": source.size_bytes,
                "source_id": source.id,
            }
        )
    return items
