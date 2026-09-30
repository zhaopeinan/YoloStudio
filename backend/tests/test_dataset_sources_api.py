from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from test_dataset_import_api import create_import_zip, isolated_client


def _make_minimal_zip_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.jpg", b"fake-image")
        archive.writestr("labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    return buffer.getvalue()


def test_dataset_source_upload_list_delete_and_options(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        options = client.get("/api/datasets/sources")
        assert options.status_code == 200
        # Known server zip may or may not exist in isolated workspace.
        assert isinstance(options.json()["items"], list)

        upload = client.post(
            "/api/datasets/sources",
            files={"archive": ("my-data.zip", zip_path.read_bytes(), "application/zip")},
        )
        assert upload.status_code == 200, upload.text
        source = upload.json()
        assert source["original_filename"] == "my-data.zip"
        assert source["size_bytes"] > 0
        assert Path(source["source_path"]).is_file()

        listed = client.get("/api/datasets/sources")
        assert listed.status_code == 200
        upload_options = [item for item in listed.json()["items"] if item["kind"] == "upload"]
        assert len(upload_options) == 1
        assert upload_options[0]["source_ref"] == f"upload:{source['id']}"
        assert upload_options[0]["source_path"] == source["source_path"]

        scan = client.post(
            "/api/datasets/scan",
            json={"source_path": source["source_path"]},
        )
        assert scan.status_code == 200
        assert scan.json()["total_images"] >= 1

        bad = client.post(
            "/api/datasets/sources",
            files={"archive": ("notes.txt", b"not-a-zip", "text/plain")},
        )
        assert bad.status_code == 400

        delete = client.delete(f"/api/datasets/sources/{source['id']}")
        assert delete.status_code == 204
        empty = client.get("/api/datasets/sources/uploads")
        assert empty.status_code == 200
        assert empty.json()["items"] == []


def test_dataset_source_rejects_invalid_zip(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/sources",
            files={"archive": ("broken.zip", b"not-really-zip", "application/zip")},
        )
        assert response.status_code == 400
