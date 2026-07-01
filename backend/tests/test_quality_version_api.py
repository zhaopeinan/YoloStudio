from __future__ import annotations

import json
from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def _import_dataset(client, zip_path: Path) -> dict:
    response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Version Test Project",
            "dataset_name": "sample",
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_class(client, project_id: int, name: str) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/classes",
        json={"name": name, "color": "#ef4444"},
    )
    assert response.status_code == 200
    return response.json()


def test_dataset_quality_reports_training_readiness(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path)
        class_payload = _create_class(client, dataset["project_id"], "drone")
        image_id = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"][0][
            "id"
        ]

        quality_before = client.get(f"/api/datasets/{dataset['dataset_id']}/quality")

        assert quality_before.status_code == 200
        assert quality_before.json()["ready_for_training"] is False
        assert "Dataset has no saved annotations." in quality_before.json()["issues"]

        annotation_response = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                        "track_id": "track-001",
                        "edge_tags": ["occluded"],
                    },
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.2,
                        "y_center": 0.2,
                        "width": 0.005,
                        "height": 0.005,
                        "edge_tags": ["small"],
                    },
                ]
            },
        )
        assert annotation_response.status_code == 200

        quality_after = client.get(f"/api/datasets/{dataset['dataset_id']}/quality")

        assert quality_after.status_code == 200
        payload = quality_after.json()
        assert payload["image_count"] == 2
        assert payload["annotated_image_count"] == 1
        assert payload["unannotated_image_count"] == 1
        assert payload["annotation_count"] == 2
        assert payload["class_count"] == 1
        assert payload["tiny_box_count"] == 2
        assert payload["invalid_box_count"] == 0
        assert payload["ready_for_training"] is True
        assert payload["issues"] == [
            "1 image has no annotations.",
            "2 boxes are smaller than 10x10 pixels.",
        ]


def test_create_dataset_version_exports_yolo_artifacts(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path)
        class_payload = _create_class(client, dataset["project_id"], "drone")
        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
        first_image_id = images[0]["id"]
        second_image_id = images[1]["id"]

        for image_id, center in [(first_image_id, 0.5), (second_image_id, 0.35)]:
            response = client.put(
                f"/api/images/{image_id}/annotations",
                json={
                    "annotations": [
                        {
                            "class_id": class_payload["id"],
                            "x_center": center,
                            "y_center": center,
                            "width": 0.25,
                            "height": 0.25,
                            "track_id": f"track-{image_id}",
                            "edge_tags": ["occluded"],
                        }
                    ]
                },
            )
            assert response.status_code == 200

        version_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "smoke-export"},
        )

        assert version_response.status_code == 200
        version = version_response.json()
        assert version["name"] == "smoke-export"
        assert version["frozen"] is True
        assert version["class_mapping"] == {str(class_payload["id"]): 0}
        assert version["split_counts"] == {"train": 1, "val": 1, "test": 0}

        artifact_root = Path(version["artifact_path"])
        assert artifact_root.exists()
        assert (artifact_root / "images" / "train").exists()
        assert (artifact_root / "images" / "val").exists()
        assert (artifact_root / "labels" / "train").exists()
        assert (artifact_root / "labels" / "val").exists()

        data_yaml = (artifact_root / "data.yaml").read_text()
        assert f"path: {artifact_root}" in data_yaml
        assert "train: images/train" in data_yaml
        assert "val: images/val" in data_yaml
        assert "test: images/test" in data_yaml
        assert "names:" in data_yaml
        assert "  0: drone" in data_yaml

        manifest = json.loads((artifact_root / "manifest.json").read_text())
        assert manifest["dataset_id"] == dataset["dataset_id"]
        assert manifest["project_id"] == dataset["project_id"]
        assert manifest["class_mapping"] == {str(class_payload["id"]): 0}
        assert len(manifest["images"]) == 2
        assert {item["split"] for item in manifest["images"]} == {"train", "val"}
        assert all(item["annotations"] for item in manifest["images"])

        label_files = sorted((artifact_root / "labels").glob("*/*.txt"))
        assert len(label_files) == 2
        assert all(label.read_text().startswith("0 ") for label in label_files)

        list_response = client.get(f"/api/datasets/{dataset['dataset_id']}/versions")

        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["id"] == version["id"]


def test_create_dataset_version_can_freeze_selected_class_subset(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path)
        drone_class = _create_class(client, dataset["project_id"], "drone")
        decoy_class = _create_class(client, dataset["project_id"], "decoy")
        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
        first_image_id = images[0]["id"]
        second_image_id = images[1]["id"]

        first_response = client.put(
            f"/api/images/{first_image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": drone_class["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.25,
                        "height": 0.25,
                    },
                    {
                        "class_id": decoy_class["id"],
                        "x_center": 0.2,
                        "y_center": 0.2,
                        "width": 0.2,
                        "height": 0.2,
                    },
                ]
            },
        )
        assert first_response.status_code == 200
        second_response = client.put(
            f"/api/images/{second_image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": decoy_class["id"],
                        "x_center": 0.35,
                        "y_center": 0.35,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ]
            },
        )
        assert second_response.status_code == 200

        version_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "drone-only", "class_ids": [drone_class["id"]]},
        )

        assert version_response.status_code == 200
        version = version_response.json()
        assert version["class_mapping"] == {str(drone_class["id"]): 0}
        assert version["split_counts"] == {"train": 1, "val": 0, "test": 0}

        artifact_root = Path(version["artifact_path"])
        data_yaml = (artifact_root / "data.yaml").read_text()
        assert "  0: drone" in data_yaml
        assert "decoy" not in data_yaml

        manifest = json.loads((artifact_root / "manifest.json").read_text())
        assert manifest["selected_class_ids"] == [drone_class["id"]]
        assert len(manifest["images"]) == 1
        assert manifest["images"][0]["image_id"] == first_image_id
        assert manifest["images"][0]["annotations"] == [
            {
                "annotation_id": first_response.json()["items"][0]["id"],
                "class_id": drone_class["id"],
                "yolo_class": 0,
                "track_id": None,
                "edge_tags": [],
            }
        ]

        label_files = sorted((artifact_root / "labels").glob("*/*.txt"))
        assert len(label_files) == 1
        assert label_files[0].read_text().startswith("0 ")

        missing_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "missing", "class_ids": [9999]},
        )

        assert missing_response.status_code == 400
        assert "Selected classes must belong" in missing_response.text
