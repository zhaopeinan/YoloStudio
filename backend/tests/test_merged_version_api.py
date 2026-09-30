from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from test_dataset_import_api import PNG_1X1, isolated_client


def _write_zip(path: Path, names: list[str]) -> None:
    with ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, PNG_1X1)


def _import(client, zip_path: Path, project_name: str, dataset_name: str) -> dict:
    response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": project_name,
            "dataset_name": dataset_name,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _annotate_all(client, dataset_id: int, class_id: int) -> None:
    images = client.get(f"/api/datasets/{dataset_id}/images").json()["items"]
    for image in images:
        response = client.put(
            f"/api/images/{image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_id,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.3,
                        "height": 0.3,
                    }
                ]
            },
        )
        assert response.status_code == 200


def test_create_merged_dataset_version(tmp_path: Path):
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    _write_zip(
        zip_a,
        [
            "images/a/fire_truck_h20_a000.jpg",
            "images/a/fire_truck_h20_a045.jpg",
        ],
    )
    _write_zip(
        zip_b,
        [
            "images/b/person_white_h20_a000.jpg",
            "images/b/person_white_h20_a090.jpg",
        ],
    )

    with isolated_client(tmp_path) as client:
        first = _import(client, zip_a, "Merge Project", "dataset-a")
        second = _import(client, zip_b, "Merge Project", "dataset-b")
        assert first["project_id"] == second["project_id"]

        class_a = client.post(
            f"/api/projects/{first['project_id']}/classes",
            json={"name": "fire_truck", "color": "#ef4444"},
        ).json()
        class_b = client.post(
            f"/api/projects/{first['project_id']}/classes",
            json={"name": "person_white", "color": "#2f80ed"},
        ).json()
        _annotate_all(client, first["dataset_id"], class_a["id"])
        _annotate_all(client, second["dataset_id"], class_b["id"])

        create = client.post(
            f"/api/datasets/{first['dataset_id']}/versions",
            json={
                "name": "round1+round2",
                "class_ids": [class_a["id"], class_b["id"]],
                "image_scope": "annotated",
                "dataset_ids": [first["dataset_id"], second["dataset_id"]],
            },
        )
        assert create.status_code == 200, create.text
        version = create.json()
        assert version["name"] == "round1+round2"
        assert version["merged"] is True
        assert version["source_dataset_ids"] == [first["dataset_id"], second["dataset_id"]]
        assert version["split_counts"]["train"] + version["split_counts"]["val"] == 4

        manifest = Path(version["artifact_path"]) / "manifest.json"
        payload = manifest.read_text()
        assert '"merged": true' in payload or '"merged": true,'.replace(",", "") in payload
        assert str(second["dataset_id"]) in payload
        # Unique export names should include both dataset prefixes.
        assert "ds" in payload
        assert (Path(version["artifact_path"]) / "data.yaml").is_file()


def test_merge_rejects_cross_project_datasets(tmp_path: Path):
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    _write_zip(zip_a, ["images/a/one.jpg", "images/a/two.jpg"])
    _write_zip(zip_b, ["images/b/one.jpg", "images/b/two.jpg"])

    with isolated_client(tmp_path) as client:
        first = _import(client, zip_a, "Project A", "dataset-a")
        second = _import(client, zip_b, "Project B", "dataset-b")
        class_a = client.post(
            f"/api/projects/{first['project_id']}/classes",
            json={"name": "target", "color": "#ef4444"},
        ).json()
        class_b = client.post(
            f"/api/projects/{second['project_id']}/classes",
            json={"name": "target", "color": "#ef4444"},
        ).json()
        _annotate_all(client, first["dataset_id"], class_a["id"])
        _annotate_all(client, second["dataset_id"], class_b["id"])

        response = client.post(
            f"/api/datasets/{first['dataset_id']}/versions",
            json={
                "dataset_ids": [first["dataset_id"], second["dataset_id"]],
                "class_ids": [class_a["id"]],
            },
        )
        assert response.status_code == 400
        assert "different projects" in response.json()["detail"]
