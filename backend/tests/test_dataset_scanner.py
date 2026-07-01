from pathlib import Path
from zipfile import ZipFile

import pytest

from app.datasets.scanner import scan_dataset_source, scan_dataset_zip


def create_sample_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z25m_00001.jpg", b"fake")
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z26m_00002.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/vtol/meta.jsonl",
            (
                '{"file":"frame_vtol_z25m_00001.jpg","drone":"vtol","z":25.5,"t":1.0}\n'
                '{"file":"frame_vtol_z26m_00002.jpg","drone":"vtol","z":26.5,"t":2.0}\n'
            ),
        )
        archive.writestr("yolo_dataset/iris/images/raw/frame_iris_z12m_00001.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/iris/meta.jsonl",
            '{"file":"frame_iris_z12m_00001.jpg","drone":"iris","z":12.0,"t":3.0}\n',
        )


def create_sample_folder(path: Path) -> None:
    (path / "yolo_dataset/vtol/images/raw").mkdir(parents=True)
    (path / "yolo_dataset/iris/images/raw").mkdir(parents=True)
    (path / "yolo_dataset/vtol/images/raw/frame_vtol_z25m_00001.jpg").write_bytes(b"fake")
    (path / "yolo_dataset/vtol/images/raw/frame_vtol_z26m_00002.jpg").write_bytes(b"fake")
    (path / "yolo_dataset/iris/images/raw/frame_iris_z12m_00001.jpg").write_bytes(b"fake")
    (path / "yolo_dataset/vtol/meta.jsonl").write_text(
        (
            '{"file":"frame_vtol_z25m_00001.jpg","drone":"vtol","z":25.5,"t":1.0}\n'
            '{"file":"frame_vtol_z26m_00002.jpg","drone":"vtol","z":26.5,"t":2.0}\n'
        ),
        encoding="utf-8",
    )
    (path / "yolo_dataset/iris/meta.jsonl").write_text(
        '{"file":"frame_iris_z12m_00001.jpg","drone":"iris","z":12.0,"t":3.0}\n',
        encoding="utf-8",
    )


def test_scan_dataset_zip_counts_images_and_metadata(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_sample_zip(zip_path)

    result = scan_dataset_zip(zip_path)

    assert result.source_path == zip_path
    assert result.total_images == 3
    assert result.total_yolo_labels == 0
    assert result.total_yaml_files == 0
    assert result.total_metadata_rows == 3
    assert [group.name for group in result.groups] == ["iris", "vtol"]
    assert result.groups[0].image_count == 1
    assert result.groups[0].altitude_min == 12.0
    assert result.groups[1].image_count == 2
    assert result.groups[1].altitude_max == 26.5


def test_scan_dataset_zip_rejects_missing_zip(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_dataset_zip(tmp_path / "missing.zip")


def test_scan_dataset_source_counts_folder_images_and_metadata(tmp_path: Path):
    source_path = tmp_path / "sample-folder"
    create_sample_folder(source_path)

    result = scan_dataset_source(source_path)

    assert result.source_path == source_path
    assert result.archive_name == "sample-folder"
    assert result.total_images == 3
    assert result.total_yolo_labels == 0
    assert result.total_yaml_files == 0
    assert result.total_metadata_rows == 3
    assert [group.name for group in result.groups] == ["iris", "vtol"]
    assert result.groups[0].image_count == 1
    assert result.groups[0].altitude_min == 12.0
    assert result.groups[1].image_count == 2
    assert result.groups[1].altitude_max == 26.5
