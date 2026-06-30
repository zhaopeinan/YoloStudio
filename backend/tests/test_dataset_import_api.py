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
