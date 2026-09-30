from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client
from test_prediction_api import _create_completed_run


def test_prediction_job_can_evaluate_another_version(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        train_version_id = client.get(f"/api/training/runs/{run['id']}").json()["version_id"]

        alt_version = client.post(
            f"/api/datasets/{run['dataset_id']}/versions",
            json={"name": "eval-only-version", "class_ids": [run["class_id"]]},
        ).json()
        assert alt_version["id"] != train_version_id

        first_image_id = run["image_ids"][0]

        def fake_predict_images(_run, images, _confidence_threshold):
            assert [image.id for image in images] == sorted(run["image_ids"])
            return {
                first_image_id: [
                    {
                        "class_id": 0,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.9,
                    }
                ],
                **{image_id: [] for image_id in run["image_ids"][1:]},
            }

        monkeypatch.setattr("app.prediction.router.predict_images", fake_predict_images)

        create_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-jobs",
            json={
                "image_scope": "all",
                "confidence_threshold": 0.25,
                "version_id": alt_version["id"],
            },
        )
        assert create_response.status_code == 200
        job = create_response.json()
        assert job["status"] == "completed"
        assert job["version_id"] == alt_version["id"]
        assert job["image_count"] == len(run["image_ids"])

        predictions = client.get(f"/api/prediction-jobs/{job['id']}/predictions").json()["items"]
        matched = [item for item in predictions if item["failure_type"] == "matched"]
        assert matched
        assert matched[0]["class_id"] == run["class_id"]

        project_versions = client.get(f"/api/projects/{run['project_id']}/versions").json()["items"]
        assert {item["id"] for item in project_versions} >= {train_version_id, alt_version["id"]}
