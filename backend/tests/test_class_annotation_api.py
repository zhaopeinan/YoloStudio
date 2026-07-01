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
