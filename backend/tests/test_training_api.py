from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def _create_version(client, zip_path: Path) -> dict:
    import_response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Training Test Project",
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

    image_id = client.get(f"/api/datasets/{dataset['dataset_id']}/images").json()["items"][0][
        "id"
    ]
    annotation_response = client.put(
        f"/api/images/{image_id}/annotations",
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
        json={"name": "training-version"},
    )
    assert version_response.status_code == 200
    version = version_response.json()
    version["project_id"] = dataset["project_id"]
    return version


def test_start_training_run_persists_status_artifacts_and_logs(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def fake_execute_training_run(run_id: int, bind, _settings) -> None:
        from app.training.runner import append_run_log, complete_training_run, record_run_metric

        append_run_log(run_id, "fake trainer started", bind=bind)
        record_run_metric(run_id, epoch=1, name="metrics/mAP50(B)", value=0.42, bind=bind)
        complete_training_run(run_id, bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", fake_execute_training_run)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        create_response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "model": "yolov8n.pt",
                "epochs": 2,
                "image_size": 320,
                "batch_size": 1,
                "device": "cpu",
                "augmentation_preset": "small-target",
                "augmentation": {
                    "mosaic": 0.8,
                    "mixup": 0.15,
                    "copy_paste": 0.25,
                    "hsv_h": 0.02,
                    "hsv_s": 0.6,
                    "hsv_v": 0.35,
                    "translate": 0.12,
                    "scale": 0.7,
                    "fliplr": 0.4,
                    "erasing": 0.2,
                    "gridmask": True,
                },
                "tta": True,
                "threshold_scan": True,
            },
        )

        assert create_response.status_code == 200
        queued_run = create_response.json()
        assert queued_run["status"] == "queued"

        run_response = client.get(f"/api/training/runs/{queued_run['id']}")
        assert run_response.status_code == 200
        run = run_response.json()
        assert run["project_id"] == version["project_id"]
        assert run["version_id"] == version["id"]
        assert run["status"] == "completed"
        assert run["device"] == "cpu"
        assert run["config"]["epochs"] == 2
        assert run["config"]["augmentation_preset"] == "small-target"
        assert run["config"]["augmentation"]["mosaic"] == 0.8
        assert run["config"]["augmentation"]["mixup"] == 0.15
        assert run["config"]["augmentation"]["copy_paste"] == 0.25
        assert run["config"]["augmentation"]["gridmask"] is True
        assert run["config"]["tta"] is True
        assert run["error_message"] is None
        assert run["latest_metrics"] == {"metrics/mAP50(B)": 0.42}

        artifact_root = Path(run["artifact_path"])
        assert (artifact_root / "config.json").exists()
        assert '"gridmask": true' in (artifact_root / "config.json").read_text()
        assert (artifact_root / "logs.txt").read_text().splitlines()[-1].endswith(
            "training completed"
        )
        assert (artifact_root / "metrics.jsonl").exists()

        list_response = client.get(f"/api/projects/{version['project_id']}/training/runs")

        assert list_response.status_code == 200
        assert list_response.json()["items"][0]["id"] == run["id"]

        logs_response = client.get(f"/api/training/runs/{run['id']}/logs")

        assert logs_response.status_code == 200
        assert "fake trainer started" in logs_response.json()["text"]


def test_ultralytics_augmentation_kwargs_excludes_local_strategy_flags():
    from app.training.runner import ultralytics_augmentation_kwargs

    kwargs = ultralytics_augmentation_kwargs(
        {
            "augmentation": {
                "mosaic": 0.8,
                "mixup": 0.15,
                "copy_paste": 0.25,
                "hsv_h": 0.02,
                "hsv_s": 0.6,
                "hsv_v": 0.35,
                "translate": 0.12,
                "scale": 0.7,
                "fliplr": 0.4,
                "erasing": 0.2,
                "gridmask": True,
            }
        }
    )

    assert kwargs == {
        "mosaic": 0.8,
        "mixup": 0.15,
        "copy_paste": 0.25,
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.35,
        "translate": 0.12,
        "scale": 0.7,
        "fliplr": 0.4,
        "erasing": 0.2,
    }


def test_post_training_threshold_scan_creates_default_prediction_jobs(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def fake_execute_training_run(run_id: int, bind, _settings) -> None:
        from app.training.runner import complete_training_run

        complete_training_run(run_id, bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", fake_execute_training_run)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)
        create_response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
                "threshold_scan": True,
            },
        )
        assert create_response.status_code == 200
        run = client.get(f"/api/training/runs/{create_response.json()['id']}").json()

        def fake_predict_images(_run, images, confidence_threshold):
            return {
                image.id: [
                    {
                        "class_id": next(iter(version["class_mapping"].keys())),
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "confidence": 1 - confidence_threshold,
                    }
                ]
                for image in images
            }

        from app.core.settings import Settings
        from app.db.session import create_engine_for_settings
        from app.training.runner import (
            DEFAULT_THRESHOLD_SCAN_VALUES,
            run_post_training_threshold_scan,
        )

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)

        run_post_training_threshold_scan(
            run["id"],
            bind=bind,
            settings=settings,
            predictor=fake_predict_images,
        )

        jobs_response = client.get(f"/api/training/runs/{run['id']}/prediction-jobs")

        assert jobs_response.status_code == 200
        jobs = jobs_response.json()["items"]
        assert len(jobs) == len(DEFAULT_THRESHOLD_SCAN_VALUES)
        assert [job["confidence_threshold"] for job in reversed(jobs)] == list(
            DEFAULT_THRESHOLD_SCAN_VALUES
        )
        assert {job["status"] for job in jobs} == {"completed"}

        summary_response = client.get(f"/api/training/runs/{run['id']}/summary")

        assert summary_response.status_code == 200
        assert [point["confidence_threshold"] for point in summary_response.json()["threshold_scan"]] == list(
            DEFAULT_THRESHOLD_SCAN_VALUES
        )

        logs_response = client.get(f"/api/training/runs/{run['id']}/logs")

        assert logs_response.status_code == 200
        assert "threshold scan started" in logs_response.json()["text"]
        assert "threshold scan completed" in logs_response.json()["text"]
        bind.dispose()


def test_training_run_rejects_when_another_run_is_active(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def leave_run_active(run_id: int, bind, _settings) -> None:
        from app.training.runner import append_run_log, mark_training_run_running

        mark_training_run_running(run_id, bind=bind)
        append_run_log(run_id, "holding run open", bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", leave_run_active)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        first = client.post(
            "/api/training/runs",
            json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
        )
        assert first.status_code == 200
        run_response = client.get(f"/api/training/runs/{first.json()['id']}")
        assert run_response.status_code == 200
        assert run_response.json()["status"] == "running"

        second = client.post(
            "/api/training/runs",
            json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
        )

        assert second.status_code == 400
        assert "Another training run is already active" in second.text
