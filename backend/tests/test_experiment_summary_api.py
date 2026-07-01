from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client
from test_prediction_api import _create_completed_run


def test_run_experiment_summary_aggregates_metrics_predictions_and_thresholds(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        first_image_id, second_image_id = run["image_ids"]

        def low_threshold_predictions(_run, _images, _confidence_threshold):
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
                second_image_id: [
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.84,
                    }
                ],
            }

        monkeypatch.setattr("app.prediction.router.predict_images", low_threshold_predictions)
        first_job_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.25},
        )
        assert first_job_response.status_code == 200

        def high_threshold_predictions(_run, _images, _confidence_threshold):
            return {
                first_image_id: [
                    {
                        "class_id": run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.92,
                    }
                ],
                second_image_id: [],
            }

        monkeypatch.setattr("app.prediction.router.predict_images", high_threshold_predictions)
        second_job_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.5},
        )
        assert second_job_response.status_code == 200
        latest_job_id = second_job_response.json()["id"]

        response = client.get(f"/api/training/runs/{run['id']}/summary")

        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == run["id"]
        assert payload["latest_prediction_job_id"] == latest_job_id
        map_series = next(item for item in payload["metric_series"] if item["name"] == "metrics/mAP50(B)")
        assert map_series["latest"] == 0.5
        assert [point["epoch"] for point in map_series["points"]] == [1]
        assert payload["class_outcomes"] == [
            {
                "class_id": run["class_id"],
                "class_name": "drone",
                "matched": 1,
                "false_positive": 0,
                "false_negative": 1,
                "class_confusion": 0,
            }
        ]
        assert payload["confusion_matrix"][0]["count"] == 1
        assert [point["confidence_threshold"] for point in payload["threshold_scan"]] == [0.25, 0.5]
        assert payload["threshold_scan"][0]["precision"] == 2 / 3
        assert payload["threshold_scan"][0]["recall"] == 1.0
        assert payload["threshold_scan"][0]["f1"] == 0.8
        assert payload["threshold_scan"][0]["class_confusion"] == 0
        assert payload["threshold_scan"][1]["recall"] == 0.5
        assert payload["threshold_scan"][1]["f1"] == 2 / 3
        assert payload["threshold_recommendation"] == {
            "job_id": first_job_response.json()["id"],
            "confidence_threshold": 0.25,
            "precision": 2 / 3,
            "recall": 1.0,
            "f1": 0.8,
        }


def test_project_experiment_summary_compares_recent_runs(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        first_run = _create_completed_run(client, zip_path, monkeypatch)
        first_image_id, second_image_id = first_run["image_ids"]

        def first_run_predictions(_run, _images, _confidence_threshold):
            return {
                first_image_id: [
                    {
                        "class_id": first_run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.9,
                    },
                    {
                        "class_id": first_run["class_id"],
                        "x_center": 0.12,
                        "y_center": 0.12,
                        "width": 0.1,
                        "height": 0.1,
                        "confidence": 0.7,
                    },
                ],
                second_image_id: [],
            }

        monkeypatch.setattr("app.prediction.router.predict_images", first_run_predictions)
        first_job_response = client.post(
            f"/api/training/runs/{first_run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.25},
        )
        assert first_job_response.status_code == 200

        second_run_response = client.post(
            "/api/training/runs",
            json={
                "version_id": first_run["version_id"],
                "model": "yolov8s.pt",
                "epochs": 3,
                "image_size": 320,
                "batch_size": 1,
                "device": "cpu",
            },
        )
        assert second_run_response.status_code == 200
        second_run = client.get(f"/api/training/runs/{second_run_response.json()['id']}").json()
        assert second_run["status"] == "completed"

        def second_run_predictions(_run, _images, _confidence_threshold):
            return {
                first_image_id: [
                    {
                        "class_id": first_run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.94,
                    }
                ],
                second_image_id: [
                    {
                        "class_id": first_run["class_id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 0.93,
                    }
                ],
            }

        monkeypatch.setattr("app.prediction.router.predict_images", second_run_predictions)
        second_job_response = client.post(
            f"/api/training/runs/{second_run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.5},
        )
        assert second_job_response.status_code == 200

        response = client.get(f"/api/projects/{first_run['project_id']}/training/summary")

        assert response.status_code == 200
        rows = response.json()["runs"]
        assert [row["run_id"] for row in rows] == [second_run["id"], first_run["id"]]
        assert rows[0]["status"] == "completed"
        assert rows[0]["model"] == "yolov8s.pt"
        assert rows[0]["epochs"] == 3
        assert rows[0]["map50"] == 0.5
        assert rows[0]["latest_prediction_job_id"] == second_job_response.json()["id"]
        assert rows[0]["matched"] == 2
        assert rows[0]["false_positive"] == 0
        assert rows[0]["false_negative"] == 0
        assert rows[0]["best_threshold"] == 0.5
        assert rows[0]["best_f1"] == 1.0
        assert rows[1]["model"] == "yolov8n.pt"
        assert rows[1]["map50"] == 0.5
        assert rows[1]["latest_prediction_job_id"] == first_job_response.json()["id"]
        assert rows[1]["matched"] == 1
        assert rows[1]["false_positive"] == 1
        assert rows[1]["false_negative"] == 1
