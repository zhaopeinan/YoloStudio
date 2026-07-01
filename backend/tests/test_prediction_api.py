from __future__ import annotations

import sys
from types import SimpleNamespace
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
    run["dataset_id"] = dataset["dataset_id"]
    run["image_ids"] = [image["id"] for image in images]
    run["image_platform_by_id"] = {image["id"]: image["platform"] for image in images}
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

        filtered_response = client.get(
            f"/api/prediction-jobs/{job['id']}/predictions"
            f"?failure_type=false_positive&class_id={run['class_id']}"
            f"&confidence_min=0.7&confidence_max=0.8&platform={run['image_platform_by_id'][first_image_id]}"
        )

        assert filtered_response.status_code == 200
        filtered_predictions = filtered_response.json()["items"]
        assert len(filtered_predictions) == 1
        assert filtered_predictions[0]["failure_type"] == "false_positive"
        assert filtered_predictions[0]["confidence"] == 0.77

        missed_response = client.get(
            f"/api/prediction-jobs/{job['id']}/predictions"
            f"?failure_type=false_negative&confidence_max=0"
            f"&platform={run['image_platform_by_id'][second_image_id]}"
        )

        assert missed_response.status_code == 200
        missed_predictions = missed_response.json()["items"]
        assert len(missed_predictions) == 1
        assert missed_predictions[0]["failure_type"] == "false_negative"

        false_positive_images = client.get(
            f"/api/datasets/{run['dataset_id']}/images?failure_type=false_positive"
        )

        assert false_positive_images.status_code == 200
        assert [image["id"] for image in false_positive_images.json()["items"]] == [first_image_id]

        false_negative_images = client.get(
            f"/api/datasets/{run['dataset_id']}/images?failure_type=false_negative"
        )

        assert false_negative_images.status_code == 200
        assert [image["id"] for image in false_negative_images.json()["items"]] == [
            second_image_id
        ]

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


def test_prediction_threshold_scan_creates_multiple_jobs_and_summary_points(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        first_image_id, second_image_id = run["image_ids"]

        def fake_predict_images(_run, _images, confidence_threshold):
            first_image_predictions = [
                {
                    "class_id": run["class_id"],
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.4,
                    "height": 0.4,
                    "confidence": 0.91,
                }
            ]
            if confidence_threshold <= 0.25:
                first_image_predictions.append(
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.15,
                        "y_center": 0.15,
                        "width": 0.1,
                        "height": 0.1,
                        "confidence": 0.32,
                    }
                )

            second_image_predictions = []
            if confidence_threshold <= 0.25:
                second_image_predictions.append(
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.44,
                    }
                )

            return {
                first_image_id: first_image_predictions,
                second_image_id: second_image_predictions,
            }

        monkeypatch.setattr("app.prediction.router.predict_images", fake_predict_images)

        create_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-threshold-scan",
            json={"image_scope": "all", "thresholds": [0.5, 0.15, 0.25]},
        )

        assert create_response.status_code == 200
        jobs = create_response.json()["items"]
        assert [job["status"] for job in jobs] == ["completed", "completed", "completed"]
        assert [job["confidence_threshold"] for job in jobs] == [0.15, 0.25, 0.5]
        assert [job["matched_count"] for job in jobs] == [2, 2, 1]
        assert [job["false_positive_count"] for job in jobs] == [1, 1, 0]
        assert [job["false_negative_count"] for job in jobs] == [0, 0, 1]

        jobs_response = client.get(f"/api/training/runs/{run['id']}/prediction-jobs")

        assert jobs_response.status_code == 200
        assert [job["id"] for job in jobs_response.json()["items"]] == [
            jobs[2]["id"],
            jobs[1]["id"],
            jobs[0]["id"],
        ]

        summary_response = client.get(f"/api/training/runs/{run['id']}/summary")

        assert summary_response.status_code == 200
        threshold_scan = summary_response.json()["threshold_scan"]
        assert [point["confidence_threshold"] for point in threshold_scan] == [0.15, 0.25, 0.5]
        assert [point["job_id"] for point in threshold_scan] == [job["id"] for job in jobs]
        assert threshold_scan[0]["precision"] == 2 / 3
        assert threshold_scan[0]["recall"] == 1
        assert threshold_scan[2]["precision"] == 1
        assert threshold_scan[2]["recall"] == 0.5


def test_predict_images_passes_tta_to_ultralytics(tmp_path: Path, monkeypatch):
    from app.db.models import Image, TrainingRun
    from app.prediction.runner import predict_images

    calls = []

    class FakeYOLO:
        def __init__(self, weights_path):
            self.weights_path = weights_path

        def predict(self, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    run_root = tmp_path / "run"
    weights = run_root / "ultralytics" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("fake weights")
    run = TrainingRun(
        id=1,
        project_id=1,
        version_id=1,
        status="completed",
        device="cpu",
        config={"tta": True},
        artifact_path=str(run_root),
        log_path=str(run_root / "logs.txt"),
    )
    image = Image(id=10, dataset_id=1, relative_path="/tmp/frame.png")

    results = predict_images(run, [image], confidence_threshold=0.35)

    assert results == {10: []}
    assert calls == [
        {
            "source": "/tmp/frame.png",
            "conf": 0.35,
            "augment": True,
            "verbose": False,
        }
    ]
