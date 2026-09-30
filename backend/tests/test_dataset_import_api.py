from __future__ import annotations

import base64
from contextlib import contextmanager
from collections.abc import Generator
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import Base, Image
from app.db.session import create_engine_for_settings, get_db
from app.main import app


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
JPEG_32X24 = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        *b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00",
        0xFF,
        0xC0,
        0x00,
        0x11,
        0x08,
        0x00,
        0x18,
        0x00,
        0x20,
        0x03,
        0x01,
        0x11,
        0x00,
        0x02,
        0x11,
        0x00,
        0x03,
        0x11,
        0x00,
        0xFF,
        0xD9,
    ]
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


def create_labeled_yolo_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("dataset/data.yaml", "names:\n  0: target\n  1: decoy\n")
        archive.writestr("dataset/images/train/frame001.png", PNG_1X1)
        archive.writestr("dataset/labels/train/frame001.txt", "0 0.5 0.5 0.4 0.4\n")


def create_unknown_class_yolo_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("dataset/data.yaml", "names: [target]\n")
        archive.writestr("dataset/images/train/frame001.png", PNG_1X1)
        archive.writestr("dataset/labels/train/frame001.txt", "2 0.5 0.5 0.4 0.4\n")


def create_jpeg_import_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("dataset/images/train/frame001.jpg", JPEG_32X24)
        archive.writestr(
            "dataset/meta.jsonl",
            '{"file":"frame001.jpg","drone":"iris","z":12.0,"t":1.0}\n',
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

    from app.auth.deps import get_current_user
    from types import SimpleNamespace

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1,
        username="test-admin",
        role="admin",
        is_active=True,
        password_hash="unused",
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_settings, None)
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


def test_import_dataset_skips_macos_metadata_files(tmp_path: Path):
    zip_path = tmp_path / "camo.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("prius_hybrid_camo/images/raw/prius_hybrid_camo_h20_a045.jpg", PNG_1X1)
        archive.writestr("suv_camo/images/raw/suv_camo_h20_a045.jpg", PNG_1X1)
        archive.writestr(
            "._prius_hybrid_camo/images/raw/._prius_hybrid_camo_h20_a045.jpg",
            b"appledouble",
        )
        archive.writestr("suv_camo/images/raw/._suv_camo_h20_a045.jpg", b"appledouble")
        archive.writestr("__MACOSX/suv_camo/._suv_camo_h20_a045.jpg", b"appledouble")

    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Camo Project",
                "dataset_name": "camo",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["image_count"] == 2
        assert [group["name"] for group in payload["groups"]] == [
            "prius_hybrid_camo",
            "suv_camo",
        ]

        images = client.get(f"/api/datasets/{payload['dataset_id']}/images").json()["items"]
        assert len(images) == 2
        assert all("._" not in image["relative_path"] for image in images)


def test_list_projects_returns_imported_dataset_summaries(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Restore Test Project",
                "dataset_name": "restore-sample",
            },
        )
        assert response.status_code == 200
        imported = response.json()

        projects_response = client.get("/api/projects")

        assert projects_response.status_code == 200
        assert projects_response.json()["items"] == [
            {
                "id": imported["project_id"],
                "name": "Restore Test Project",
                "datasets": [
                    {
                        "id": imported["dataset_id"],
                        "project_id": imported["project_id"],
                        "name": "restore-sample",
                        "source_type": "zip",
                        "import_status": "imported",
                        "image_count": 2,
                        "annotated_image_count": 0,
                        "annotation_count": 0,
                    }
                ],
            }
        ]

        class_response = client.post(
            f"/api/projects/{imported['project_id']}/classes",
            json={"name": "target", "color": "#2f80ed"},
        )
        assert class_response.status_code == 200
        images = client.get(f"/api/datasets/{imported['dataset_id']}/images").json()["items"]
        annotation_response = client.put(
            f"/api/images/{images[0]['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_response.json()["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.2,
                        "height": 0.2,
                    },
                    {
                        "class_id": class_response.json()["id"],
                        "x_center": 0.2,
                        "y_center": 0.2,
                        "width": 0.1,
                        "height": 0.1,
                    },
                ]
            },
        )
        assert annotation_response.status_code == 200

        updated_projects_response = client.get("/api/projects")

        assert updated_projects_response.status_code == 200
        updated_dataset = updated_projects_response.json()["items"][0]["datasets"][0]
        assert updated_dataset["annotated_image_count"] == 1
        assert updated_dataset["annotation_count"] == 2


def test_dataset_coverage_summary_reports_diversity(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Coverage Test Project",
                "dataset_name": "coverage-sample",
            },
        )
        assert response.status_code == 200
        dataset = response.json()
        drone_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "drone", "color": "#ef4444"},
        )
        decoy_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "decoy", "color": "#22c55e"},
        )
        vehicle_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "vehicle", "color": "#2f80ed"},
        )
        assert drone_response.status_code == 200
        assert decoy_response.status_code == 200
        assert vehicle_response.status_code == 200
        drone_class = drone_response.json()
        decoy_class = decoy_response.json()
        vehicle_class = vehicle_response.json()

        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
        iris_image = next(image for image in images if image["platform"] == "iris")
        vtol_image = next(image for image in images if image["platform"] == "vtol")

        iris_annotations = client.put(
            f"/api/images/{iris_image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": drone_class["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "edge_tags": ["occluded", "camouflaged"],
                    }
                ]
            },
        )
        vtol_annotations = client.put(
            f"/api/images/{vtol_image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": decoy_class["id"],
                        "x_center": 0.4,
                        "y_center": 0.4,
                        "width": 0.3,
                        "height": 0.3,
                        "edge_tags": ["occluded"],
                    }
                ]
            },
        )
        assert iris_annotations.status_code == 200
        assert vtol_annotations.status_code == 200

        coverage_response = client.get(f"/api/datasets/{dataset['dataset_id']}/coverage")

        assert coverage_response.status_code == 200
        coverage = coverage_response.json()
        assert coverage["dataset_id"] == dataset["dataset_id"]
        assert coverage["image_count"] == 2
        assert coverage["annotated_image_count"] == 2
        assert coverage["annotation_count"] == 2
        assert coverage["platforms"] == [
            {
                "label": "iris",
                "image_count": 1,
                "annotated_image_count": 1,
                "annotation_count": 1,
            },
            {
                "label": "vtol",
                "image_count": 1,
                "annotated_image_count": 1,
                "annotation_count": 1,
            },
        ]
        assert coverage["altitude_bands"] == [
            {
                "label": "<20m",
                "image_count": 1,
                "annotated_image_count": 1,
                "annotation_count": 1,
            },
            {
                "label": "20-50m",
                "image_count": 1,
                "annotated_image_count": 1,
                "annotation_count": 1,
            },
        ]
        assert coverage["classes"] == [
            {
                "class_id": drone_class["id"],
                "class_name": "drone",
                "class_color": "#ef4444",
                "image_count": 1,
                "annotation_count": 1,
            },
            {
                "class_id": decoy_class["id"],
                "class_name": "decoy",
                "class_color": "#22c55e",
                "image_count": 1,
                "annotation_count": 1,
            },
            {
                "class_id": vehicle_class["id"],
                "class_name": "vehicle",
                "class_color": "#2f80ed",
                "image_count": 0,
                "annotation_count": 0,
            },
        ]
        assert coverage["edge_tags"] == [
            {"tag": "occluded", "image_count": 2, "annotation_count": 2},
            {"tag": "camouflaged", "image_count": 1, "annotation_count": 1},
        ]


def test_import_dataset_loads_yolo_classes_and_labels(tmp_path: Path):
    zip_path = tmp_path / "labeled.zip"
    create_labeled_yolo_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Labeled Import Project",
                "dataset_name": "labeled",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        classes_response = client.get(f"/api/projects/{payload['project_id']}/classes")
        assert classes_response.status_code == 200
        classes = classes_response.json()["items"]
        assert [class_item["name"] for class_item in classes] == ["target", "decoy"]

        images_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")
        assert images_response.status_code == 200
        images = images_response.json()["items"]
        assert len(images) == 1
        assert images[0]["annotation_count"] == 1

        annotations_response = client.get(f"/api/images/{images[0]['id']}/annotations")
        assert annotations_response.status_code == 200
        annotations = annotations_response.json()["items"]
        assert annotations == [
            {
                "id": annotations[0]["id"],
                "image_id": images[0]["id"],
                "class_id": classes[0]["id"],
                "class_name": "target",
                "class_color": classes[0]["color"],
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.4,
                "height": 0.4,
                "track_id": None,
                "edge_tags": [],
            }
        ]


def test_import_dataset_reads_jpeg_dimensions(tmp_path: Path):
    zip_path = tmp_path / "jpeg.zip"
    create_jpeg_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "JPEG Import Project",
                "dataset_name": "jpeg",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        images_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")

        assert images_response.status_code == 200
        images = images_response.json()["items"]
        assert len(images) == 1
        assert images[0]["width"] == 32
        assert images[0]["height"] == 24


def test_refresh_image_dimensions_repairs_legacy_missing_sizes(tmp_path: Path):
    zip_path = tmp_path / "jpeg.zip"
    create_jpeg_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "JPEG Refresh Project",
                "dataset_name": "jpeg-refresh",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        db = next(app.dependency_overrides[get_db]())
        try:
            image = db.query(Image).filter(Image.dataset_id == payload["dataset_id"]).one()
            image.width = None
            image.height = None
            db.commit()
        finally:
            db.close()

        before_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")
        assert before_response.status_code == 200
        assert before_response.json()["items"][0]["width"] is None
        assert before_response.json()["items"][0]["height"] is None

        refresh_response = client.post(
            f"/api/datasets/{payload['dataset_id']}/refresh-image-dimensions"
        )

        assert refresh_response.status_code == 200
        assert refresh_response.json() == {
            "dataset_id": payload["dataset_id"],
            "scanned_count": 1,
            "updated_count": 1,
            "missing_count": 0,
        }

        after_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")
        assert after_response.status_code == 200
        after_image = after_response.json()["items"][0]
        assert after_image["width"] == 32
        assert after_image["height"] == 24


def test_refresh_image_dimensions_reports_missing_dataset(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        response = client.post("/api/datasets/999/refresh-image-dimensions")

        assert response.status_code == 404
        assert "Dataset was not found" in response.text


def test_import_dataset_reports_unknown_yolo_class_references(tmp_path: Path):
    zip_path = tmp_path / "unknown-class.zip"
    create_unknown_class_yolo_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Unknown Class Project",
                "dataset_name": "unknown-class",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        images_response = client.get(f"/api/datasets/{payload['dataset_id']}/images")
        assert images_response.status_code == 200
        images = images_response.json()["items"]
        assert len(images) == 1
        assert images[0]["annotation_count"] == 0

        quality_response = client.get(f"/api/datasets/{payload['dataset_id']}/quality")
        assert quality_response.status_code == 200
        quality = quality_response.json()
        assert quality["unknown_class_reference_count"] == 1
        assert "1 label references unknown class indexes." in quality["issues"]

        issues_response = client.get(f"/api/datasets/{payload['dataset_id']}/quality/issues")
        assert issues_response.status_code == 200
        issues = issues_response.json()["items"]
        unknown_issue = next(
            issue for issue in issues if issue["issue_type"] == "unknown_class_reference"
        )
        assert unknown_issue["image_id"] == images[0]["id"]
        assert unknown_issue["image_path"] == images[0]["relative_path"]
        assert unknown_issue["class_name"] == "YOLO class 2"
        assert unknown_issue["message"] == "1 label row references unknown class index 2."


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
