from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def import_dataset(client, zip_path: Path) -> dict:
    response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Annotation Test Project",
            "dataset_name": "sample",
        },
    )
    assert response.status_code == 200
    return response.json()


def test_create_class_and_replace_image_annotations(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = import_dataset(client, zip_path)

        create_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "drone", "color": "#2f80ed", "description": "aircraft"},
        )

        assert create_response.status_code == 200
        class_payload = create_response.json()
        assert class_payload["name"] == "drone"
        assert class_payload["color"] == "#2f80ed"

        list_response = client.get(f"/api/projects/{dataset['project_id']}/classes")

        assert list_response.status_code == 200
        assert list_response.json()["items"] == [class_payload]

        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
        image_id = images[0]["id"]
        assert images[0]["annotation_status"] == "unreviewed"

        replace_response = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                        "track_id": "target-001",
                        "edge_tags": ["occluded", "small"],
                    }
                ]
            },
        )

        assert replace_response.status_code == 200
        annotations = replace_response.json()["items"]
        assert len(annotations) == 1
        assert replace_response.json()["annotation_status"] == "annotated"
        assert annotations[0]["class_name"] == "drone"
        assert annotations[0]["class_color"] == "#2f80ed"
        assert annotations[0]["edge_tags"] == ["occluded", "small"]

        get_response = client.get(f"/api/images/{image_id}/annotations")

        assert get_response.status_code == 200
        assert get_response.json()["items"] == annotations

        empty_replace = client.put(
            f"/api/images/{image_id}/annotations",
            json={"annotations": []},
        )

        assert empty_replace.status_code == 200
        assert empty_replace.json()["items"] == []
        assert empty_replace.json()["annotation_status"] == "unreviewed"

        negative_replace = client.put(
            f"/api/images/{image_id}/annotations",
            json={"annotations": [], "annotation_status": "negative"},
        )
        assert negative_replace.status_code == 200
        assert negative_replace.json()["annotation_status"] == "negative"
        assert client.get(f"/api/images/{image_id}/annotations").json()["annotation_status"] == "negative"

        images_after_negative = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images"
        ).json()["items"]
        assert images_after_negative[0]["annotation_status"] == "negative"

        invalid_negative = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                    }
                ],
                "annotation_status": "negative",
            },
        )
        assert invalid_negative.status_code == 400


def test_update_class_updates_annotation_read_labels(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = import_dataset(client, zip_path)
        project_id = dataset["project_id"]
        drone_class = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "drone", "color": "#2f80ed"},
        ).json()
        vehicle_class = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "vehicle", "color": "#22c55e"},
        ).json()
        image_id = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"][0][
            "id"
        ]

        replace_response = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": drone_class["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                    }
                ]
            },
        )
        assert replace_response.status_code == 200

        update_response = client.patch(
            f"/api/projects/{project_id}/classes/{drone_class['id']}",
            json={"name": "aircraft", "color": "#111827", "description": None},
        )

        assert update_response.status_code == 200
        assert update_response.json() == {
            **drone_class,
            "name": "aircraft",
            "color": "#111827",
            "description": None,
            "annotation_count": 1,
        }

        annotations_response = client.get(f"/api/images/{image_id}/annotations")

        assert annotations_response.status_code == 200
        assert annotations_response.json()["items"][0]["class_name"] == "aircraft"
        assert annotations_response.json()["items"][0]["class_color"] == "#111827"

        duplicate_response = client.patch(
            f"/api/projects/{project_id}/classes/{drone_class['id']}",
            json={"name": vehicle_class["name"]},
        )
        assert duplicate_response.status_code == 400

        blank_response = client.patch(
            f"/api/projects/{project_id}/classes/{drone_class['id']}",
            json={"name": "   "},
        )
        assert blank_response.status_code == 400

        missing_class_response = client.patch(
            f"/api/projects/{project_id}/classes/{vehicle_class['id'] + 999}",
            json={"name": "missing"},
        )
        assert missing_class_response.status_code == 404

        missing_project_response = client.patch(
            f"/api/projects/{project_id + 999}/classes/{vehicle_class['id']}",
            json={"name": "wrong-project"},
        )
        assert missing_project_response.status_code == 404


def test_delete_class_requires_confirmation_and_protects_references(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = import_dataset(client, zip_path)
        project_id = dataset["project_id"]
        used_class = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "used"},
        ).json()
        version_class = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "version-only"},
        ).json()
        unused_class = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "unused"},
        ).json()
        images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]

        for image in images[:2]:
            annotation_response = client.put(
                f"/api/images/{image['id']}/annotations",
                json={
                    "annotations": [
                        {
                            "class_id": used_class["id"],
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width": 0.25,
                            "height": 0.25,
                        }
                    ]
                },
            )
            assert annotation_response.status_code == 200

        version_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={
                "name": "class-delete-protection",
                "class_ids": [used_class["id"], version_class["id"]],
            },
        )
        assert version_response.status_code == 200

        classes = client.get(f"/api/projects/{project_id}/classes").json()["items"]
        by_name = {item["name"]: item for item in classes}
        assert by_name["used"]["annotation_count"] == 2
        assert by_name["used"]["version_count"] == 1
        assert by_name["version-only"]["annotation_count"] == 0
        assert by_name["version-only"]["version_count"] == 1
        assert by_name["unused"]["annotation_count"] == 0
        assert by_name["unused"]["version_count"] == 0

        delete_unused = client.delete(
            f"/api/projects/{project_id}/classes/{unused_class['id']}"
        )
        assert delete_unused.status_code == 200
        assert delete_unused.json()["name"] == "unused"

        delete_used = client.delete(
            f"/api/projects/{project_id}/classes/{used_class['id']}"
        )
        assert delete_used.status_code == 409
        assert "2 条标注" in delete_used.json()["detail"]

        delete_version_class = client.delete(
            f"/api/projects/{project_id}/classes/{version_class['id']}"
        )
        assert delete_version_class.status_code == 409
        assert "1 个数据集版本" in delete_version_class.json()["detail"]

        assert client.delete(f"/api/projects/{project_id}/classes/999999").status_code == 404
        assert client.delete(f"/api/projects/999999/classes/{used_class['id']}").status_code == 404


def test_annotation_replace_rejects_out_of_bounds_bbox(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = import_dataset(client, zip_path)
        class_payload = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "drone"},
        ).json()
        image_id = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"][0][
            "id"
        ]

        response = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.9,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.5,
                    }
                ]
            },
        )

        assert response.status_code in {400, 422}


def test_annotation_replace_accepts_edge_box_rounding_overshoot(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = import_dataset(client, zip_path)
        class_payload = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "person_red"},
        ).json()
        image_id = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"][0][
            "id"
        ]

        # Mirrors frontend 6-digit rounding of a left-edge box.
        response = client.put(
            f"/api/images/{image_id}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.005233,
                        "y_center": 0.606034,
                        "width": 0.010467,
                        "height": 0.02959,
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text
        saved = response.json()["items"][0]
        assert saved["x_center"] - saved["width"] / 2 >= -1e-9
        assert saved["x_center"] + saved["width"] / 2 <= 1 + 1e-9
