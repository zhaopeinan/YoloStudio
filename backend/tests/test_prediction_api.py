from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def _create_completed_run(client, zip_path: Path, monkeypatch) -> dict:
    def fake_execute_training_run(run_id: int, bind, _settings) -> None:
        from app.training.runner import complete_training_run, record_run_metric

        record_run_metric(run_id, epoch=1, name="metrics/mAP50(B)", value=0.5, bind=bind)
        complete_training_run(run_id, bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", fake_execute_training_run)

    import_response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Prediction Test Project",
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
        response = client.put(
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
        assert response.status_code == 200

    version_response = client.post(
        f"/api/datasets/{dataset['dataset_id']}/versions",
        json={"name": "prediction-version"},
    )
    assert version_response.status_code == 200
    version = version_response.json()

    run_response = client.post(
        "/api/training/runs",
        json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
    )
    assert run_response.status_code == 200
    run = client.get(f"/api/training/runs/{run_response.json()['id']}").json()
    assert run["status"] == "completed"
    run["class_id"] = class_payload["id"]
    run["image_ids"] = [image["id"] for image in images]
    return run


def test_prediction_job_persists_matches_and_failures(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        first_image_id, second_image_id = run["image_ids"]

        def fake_predict_images(_run, images, _confidence_threshold):
            return {
                first_image_id: [
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.91,
                    },
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.15,
                        "y_center": 0.15,
                        "width": 0.1,
                        "height": 0.1,
                        "confidence": 0.77,
                    },
                ],
                second_image_id: [],
            }

        monkeypatch.setattr("app.prediction.router.predict_images", fake_predict_images)

        create_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.25},
        )

        assert create_response.status_code == 200
        job = create_response.json()
        assert job["status"] == "completed"
        assert job["image_count"] == 2
        assert job["prediction_count"] == 3
        assert job["matched_count"] == 1
        assert job["false_positive_count"] == 1
        assert job["false_negative_count"] == 1

        predictions_response = client.get(f"/api/prediction-jobs/{job['id']}/predictions")

        assert predictions_response.status_code == 200
        predictions = predictions_response.json()["items"]
        assert [item["failure_type"] for item in predictions] == [
            "matched",
            "false_positive",
            "false_negative",
        ]
        assert predictions[0]["matched_annotation_id"] is not None
        assert predictions[2]["confidence"] == 0

        jobs_response = client.get(f"/api/training/runs/{run['id']}/prediction-jobs")

        assert jobs_response.status_code == 200
        assert jobs_response.json()["items"][0]["id"] == job["id"]

        logs_response = client.get(f"/api/prediction-jobs/{job['id']}/logs")

        assert logs_response.status_code == 200
        assert "prediction completed" in logs_response.json()["text"]

        review_response = client.get(
            f"/api/prediction-jobs/{job['id']}/images/{first_image_id}/review"
        )

        assert review_response.status_code == 200
        review = review_response.json()
        assert review["image"]["id"] == first_image_id
        assert len(review["annotations"]) == 1
        assert len(review["predictions"]) == 2
        assert review["counts"] == {
            "matched": 1,
            "false_positive": 1,
            "false_negative": 0,
        }
