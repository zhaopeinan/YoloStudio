from __future__ import annotations

import base64
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import Base
from app.db.session import create_engine_for_settings, get_db
from app.main import app


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def create_import_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_00001.png", PNG_1X1)
        archive.writestr("yolo_dataset/iris/images/raw/frame_iris_00001.png", PNG_1X1)
        archive.writestr(
            "yolo_dataset/vtol/meta.jsonl",
            '{"file":"frame_vtol_00001.png","drone":"vtol","z":25.5,"t":1.0}\n',
        )
        archive.writestr(
            "yolo_dataset/iris/meta.jsonl",
            '{"file":"frame_iris_00001.png","drone":"iris","z":12.0,"t":2.0}\n',
        )


def create_import_folder(path: Path) -> None:
    (path / "yolo_dataset/vtol/images/raw").mkdir(parents=True)
    (path / "yolo_dataset/iris/images/raw").mkdir(parents=True)
    (path / "yolo_dataset/vtol/images/raw/frame_vtol_00001.png").write_bytes(PNG_1X1)
    (path / "yolo_dataset/iris/images/raw/frame_iris_00001.png").write_bytes(PNG_1X1)
    (path / "yolo_dataset/vtol/meta.jsonl").write_text(
        '{"file":"frame_vtol_00001.png","drone":"vtol","z":25.5,"t":1.0}\n',
        encoding="utf-8",
    )
    (path / "yolo_dataset/iris/meta.jsonl").write_text(
        '{"file":"frame_iris_00001.png","drone":"iris","z":12.0,"t":2.0}\n',
        encoding="utf-8",
    )


@contextmanager
def isolated_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    settings = Settings(workspace_root=tmp_path / "workspace")
    engine = create_engine_for_settings(settings)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_import_dataset_persists_images_and_serves_files(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "API Test Project",
                "dataset_name": "sample",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["project_name"] == "API Test Project"
        assert payload["dataset_name"] == "sample"
        assert payload["image_count"] == 2
        assert payload["groups"] == [
            {"name": "iris", "image_count": 1, "metadata_rows": 1},
            {"name": "vtol", "image_count": 1, "metadata_rows": 1},
        ]

        images_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")

        assert images_response.status_code == 200
        images = images_response.json()["items"]
        assert len(images) == 2
        assert images[0]["image_url"].startswith("/api/images/")
        assert images[0]["annotation_count"] == 0
        assert {image["platform"] for image in images} == {"iris", "vtol"}
        assert {image["altitude"] for image in images} == {12.0, 25.5}

        file_response = client.get(images[0]["image_url"])

        assert file_response.status_code == 200
        assert file_response.headers["content-type"] == "image/png"
        assert file_response.content == PNG_1X1


def test_import_dataset_accepts_folder_and_serves_files(tmp_path: Path):
    source_path = tmp_path / "sample-folder"
    create_import_folder(source_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(source_path),
                "project_name": "Folder Import Project",
                "dataset_name": "folder-sample",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["project_name"] == "Folder Import Project"
        assert payload["dataset_name"] == "folder-sample"
        assert payload["image_count"] == 2
        assert payload["groups"] == [
            {"name": "iris", "image_count": 1, "metadata_rows": 1},
            {"name": "vtol", "image_count": 1, "metadata_rows": 1},
        ]

        images_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")

        assert images_response.status_code == 200
        images = images_response.json()["items"]
        assert len(images) == 2
        assert {image["platform"] for image in images} == {"iris", "vtol"}
        assert {image["altitude"] for image in images} == {12.0, 25.5}

        file_response = client.get(images[0]["image_url"])

        assert file_response.status_code == 200
        assert file_response.headers["content-type"] == "image/png"
        assert file_response.content == PNG_1X1


def test_list_dataset_images_filters_by_metadata_and_annotations(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Filter Test Project",
                "dataset_name": "sample",
            },
        )
        assert response.status_code == 200
        dataset = response.json()
        class_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "drone", "color": "#ef4444"},
        )
        assert class_response.status_code == 200
        class_payload = class_response.json()
        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
        iris_image = next(image for image in images if image["platform"] == "iris")

        annotation_response = client.put(
            f"/api/images/{iris_image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "edge_tags": ["occluded", "small"],
                    }
                ]
            },
        )
        assert annotation_response.status_code == 200

        platform_response = client.get(f"/api/datasets/{dataset['dataset_id']}/images?platform=iris")
        assert platform_response.status_code == 200
        assert platform_response.json()["total"] == 1
        assert platform_response.json()["items"][0]["id"] == iris_image["id"]

        annotated_response = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images?label_status=annotated"
        )
        assert annotated_response.status_code == 200
        assert [item["id"] for item in annotated_response.json()["items"]] == [iris_image["id"]]

        unannotated_response = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images?label_status=unannotated"
        )
        assert unannotated_response.status_code == 200
        assert unannotated_response.json()["total"] == 1

        class_response = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images?class_id={class_payload['id']}"
        )
        assert class_response.status_code == 200
        assert class_response.json()["items"][0]["id"] == iris_image["id"]

        tag_response = client.get(f"/api/datasets/{dataset['dataset_id']}/images?edge_tag=occluded")
        assert tag_response.status_code == 200
        assert tag_response.json()["items"][0]["id"] == iris_image["id"]

        altitude_response = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images?altitude_min=20&altitude_max=30"
        )
        assert altitude_response.status_code == 200
        assert altitude_response.json()["total"] == 1
        assert altitude_response.json()["items"][0]["platform"] == "vtol"
