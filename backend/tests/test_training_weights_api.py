from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def _create_project_with_version(client, zip_path: Path) -> dict:
    import_response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Weight Library Project",
            "dataset_name": "sample",
        },
    )
    assert import_response.status_code == 200
    dataset = import_response.json()

    class_response = client.post(
        f"/api/projects/{dataset['project_id']}/classes",
        json={"name": "drone", "color": "#ef4444"},
    )
    assert class_response.status_code == 200
    class_payload = class_response.json()

    images = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"]
    for image in images:
        annotation_response = client.put(
            f"/api/images/{image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                    }
                ]
            },
        )
        assert annotation_response.status_code == 200

    version_response = client.post(
        f"/api/datasets/{dataset['dataset_id']}/versions",
        json={"name": "weight-version"},
    )
    assert version_response.status_code == 200
    return {
        "project_id": dataset["project_id"],
        "version_id": version_response.json()["id"],
    }


def test_training_weight_library_upload_list_delete_and_model_options(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    weight_bytes = b"fake-pytorch-weights"

    with isolated_client(tmp_path) as client:
        fixture = _create_project_with_version(client, zip_path)
        project_id = fixture["project_id"]

        models = client.get(f"/api/training/models?project_id={project_id}")
        assert models.status_code == 200
        model_refs = [item["model_ref"] for item in models.json()["items"]]
        assert "base:yolov8n.pt" in model_refs
        assert "base:yolov8s.pt" in model_refs
        assert all(not item["model_ref"].startswith("weight:") for item in models.json()["items"])

        upload = client.post(
            "/api/training/weights",
            data={"project_id": str(project_id)},
            files={"weight": ("finetune.pt", weight_bytes, "application/octet-stream")},
        )
        assert upload.status_code == 200
        weight = upload.json()
        assert weight["original_filename"] == "finetune.pt"
        assert weight["size_bytes"] == len(weight_bytes)

        listed = client.get(f"/api/training/weights?project_id={project_id}")
        assert listed.status_code == 200
        assert len(listed.json()["items"]) == 1

        models_after = client.get(f"/api/training/models?project_id={project_id}")
        assert models_after.status_code == 200
        uploaded_option = next(
            item for item in models_after.json()["items"] if item["kind"] == "upload"
        )
        assert uploaded_option["model_ref"] == f"weight:{weight['id']}"
        assert "finetune.pt" in uploaded_option["label"]

        bad_ext = client.post(
            "/api/training/weights",
            data={"project_id": str(project_id)},
            files={"weight": ("notes.txt", b"nope", "text/plain")},
        )
        assert bad_ext.status_code == 400

        delete = client.delete(f"/api/training/weights/{weight['id']}")
        assert delete.status_code == 204
        empty = client.get(f"/api/training/weights?project_id={project_id}")
        assert empty.json()["items"] == []


def test_training_run_accepts_uploaded_weight_ref(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    yolo_models: list[str] = []

    class FakeYOLO:
        def __init__(self, model: str):
            yolo_models.append(model)

        def train(self, **_kwargs):
            class Results:
                results_dict = {"metrics/mAP50(B)": 0.5}

            return Results()

    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        __import__("types").SimpleNamespace(YOLO=FakeYOLO),
    )

    with isolated_client(tmp_path) as client:
        fixture = _create_project_with_version(client, zip_path)
        project_id = fixture["project_id"]
        upload = client.post(
            "/api/training/weights",
            data={"project_id": str(project_id)},
            files={"weight": ("continue.pt", b"ckpt", "application/octet-stream")},
        )
        assert upload.status_code == 200
        weight_id = upload.json()["id"]

        create = client.post(
            "/api/training/runs",
            json={
                "version_id": fixture["version_id"],
                "model": f"weight:{weight_id}",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 2,
            },
        )
        assert create.status_code == 200
        run = create.json()
        assert run["config"]["model"] == f"weight:{weight_id}"
        assert yolo_models
        assert yolo_models[0].endswith("weights.pt")
