from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.main import app


def create_sample_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z25m_00001.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/vtol/meta.jsonl",
            '{"file":"frame_vtol_z25m_00001.jpg","drone":"vtol","z":25.5,"t":1.0}\n',
        )


def test_scan_dataset_endpoint_returns_summary(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_sample_zip(zip_path)
    client = TestClient(app)

    response = client.post("/api/datasets/scan", json={"source_path": str(zip_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["archive_name"] == "sample.zip"
    assert payload["total_images"] == 1
    assert payload["groups"][0]["name"] == "vtol"
    assert "No YOLO label .txt files were found" in payload["warnings"]


def test_scan_dataset_endpoint_reports_missing_file(tmp_path: Path):
    client = TestClient(app)

    response = client.post(
        "/api/datasets/scan",
        json={"source_path": str(tmp_path / "missing.zip")},
    )

    assert response.status_code == 404
    assert "Dataset archive was not found" in response.json()["detail"]


def test_scan_dataset_endpoint_rejects_non_zip_path(tmp_path: Path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("not a zip", encoding="utf-8")
    client = TestClient(app)

    response = client.post("/api/datasets/scan", json={"source_path": str(text_path)})

    assert response.status_code == 400
    assert "Expected a .zip file" in response.json()["detail"]


def test_scan_dataset_endpoint_rejects_invalid_zip(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    zip_path.write_text("not a zip", encoding="utf-8")
    client = TestClient(app)

    response = client.post("/api/datasets/scan", json={"source_path": str(zip_path)})

    assert response.status_code == 400
    assert "Invalid zip archive" in response.json()["detail"]
