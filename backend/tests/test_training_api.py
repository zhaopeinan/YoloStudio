from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

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
        json={"name": "training-version"},
    )
    assert version_response.status_code == 200
    version = version_response.json()
    assert version["split_counts"] == {"train": 1, "val": 1, "test": 0}
    version["project_id"] = dataset["project_id"]
    return version


def test_training_rejects_version_without_validation_split_before_ultralytics(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    yolo_constructor_calls: list[str] = []

    class FakeYOLO:
        def __init__(self, model: str):
            yolo_constructor_calls.append(model)

        def train(self, **_kwargs):
            raise AssertionError("training must not start for an invalid dataset version")

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        from sqlalchemy.orm import Session

        from app.core.settings import Settings
        from app.db.models import DatasetVersion
        from app.db.session import create_engine_for_settings

        bind = create_engine_for_settings(Settings(workspace_root=tmp_path / "workspace"))
        with Session(bind) as db:
            stored_version = db.get(DatasetVersion, version["id"])
            manifest = dict(stored_version.split_manifest)
            manifest["images"] = [
                item for item in manifest["images"] if item.get("split") != "val"
            ]
            manifest["split_counts"] = {"train": 1, "val": 0, "test": 0}
            stored_version.split_manifest = manifest
            db.commit()
        bind.dispose()

        val_image = next((Path(version["artifact_path"]) / "images" / "val").iterdir())
        val_image.unlink()

        response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "model": "yolov8n.pt",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
                "device": "cpu",
            },
        )

        assert response.status_code == 200
        run = client.get(f"/api/training/runs/{response.json()['id']}").json()
        assert run["status"] == "failed"
        assert run["error_message"] == (
            "Dataset version requires at least one training image and one validation image. "
            "Annotate at least 2 images and create a new version."
        )
        assert yolo_constructor_calls == []

        logs = client.get(f"/api/training/runs/{run['id']}/logs").json()["text"]
        assert "ultralytics training started" not in logs


def test_validate_training_dataset_rejects_missing_exported_validation_artifact(
    tmp_path: Path,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        from sqlalchemy.orm import Session

        from app.core.settings import Settings
        from app.db.models import DatasetVersion
        from app.db.session import create_engine_for_settings
        from app.training.runner import (
            INVALID_TRAINING_SPLIT_MESSAGE,
            validate_training_dataset,
        )

        bind = create_engine_for_settings(Settings(workspace_root=tmp_path / "workspace"))
        with Session(bind) as db:
            stored_version = db.get(DatasetVersion, version["id"])
            val_entry = next(
                item
                for item in stored_version.split_manifest["images"]
                if item.get("split") == "val"
            )
            version_root = Path(stored_version.artifact_path)
            (version_root / val_entry["export_label"]).unlink()

            with pytest.raises(RuntimeError) as exc_info:
                validate_training_dataset(stored_version, version_root)
            assert str(exc_info.value) == INVALID_TRAINING_SPLIT_MESSAGE
        bind.dispose()


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


def test_training_run_artifacts_lists_generated_files(tmp_path: Path, monkeypatch):
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
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
                "device": "cpu",
            },
        )
        assert create_response.status_code == 200
        run = client.get(f"/api/training/runs/{create_response.json()['id']}").json()

        artifact_root = Path(run["artifact_path"])
        weights_dir = artifact_root / "ultralytics" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake weights")
        (artifact_root / "ultralytics" / "results.png").write_bytes(b"png")
        exports_dir = artifact_root / "exports"
        exports_dir.mkdir()
        (exports_dir / "run-1.pt").write_bytes(b"export")

        response = client.get(f"/api/training/runs/{run['id']}/artifacts")

        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == run["id"]
        assert payload["artifact_root"] == run["artifact_path"]
        assert payload["total_count"] >= 6
        rows = {item["relative_path"]: item for item in payload["items"]}
        assert rows["config.json"]["category"] == "config"
        assert rows["logs.txt"]["category"] == "log"
        assert rows["metrics.jsonl"]["category"] == "metrics"
        assert rows["ultralytics/weights/best.pt"]["category"] == "weights"
        assert rows["ultralytics/results.png"]["category"] == "plot"
        assert rows["exports/run-1.pt"]["category"] == "export"
        assert rows["ultralytics/weights/best.pt"]["size_bytes"] == len(b"fake weights")


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


def test_gridmask_dataset_builder_derives_masked_training_artifact(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        from app.training.runner import build_gridmask_dataset

        run_root = tmp_path / "workspace" / "projects" / str(version["project_id"]) / "runs" / "99"
        derived_data_yaml = build_gridmask_dataset(Path(version["artifact_path"]), run_root)
        derived_root = derived_data_yaml.parent

        assert derived_root == run_root / "gridmask_dataset"
        assert derived_data_yaml.exists()
        assert f"path: {derived_root}" in derived_data_yaml.read_text()

        source_image = next((Path(version["artifact_path"]) / "images").rglob("*.png"))
        derived_image = derived_root / "images" / source_image.relative_to(
            Path(version["artifact_path"]) / "images"
        )
        assert derived_image.exists()
        assert derived_image.read_bytes() != source_image.read_bytes()

        source_label = next((Path(version["artifact_path"]) / "labels").rglob("*.txt"))
        derived_label = derived_root / "labels" / source_label.relative_to(
            Path(version["artifact_path"]) / "labels"
        )
        assert derived_label.read_text() == source_label.read_text()

        assert (derived_root / "manifest.json").exists()


def test_execute_training_run_uses_gridmask_dataset_when_enabled(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    train_calls: list[dict] = []

    class FakeResults:
        results_dict = {"metrics/mAP50(B)": 0.55}

    class FakeYOLO:
        def __init__(self, model: str):
            self.model = model

        def train(self, **kwargs):
            train_calls.append(kwargs)
            return FakeResults()

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)
        response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "model": "yolov8n.pt",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
                "augmentation": {"gridmask": True},
            },
        )

        assert response.status_code == 200
        run = client.get(f"/api/training/runs/{response.json()['id']}").json()

        assert run["status"] == "completed"
        assert train_calls
        train_data = Path(train_calls[0]["data"])
        assert train_data.name == "data.yaml"
        assert train_data.parent == Path(run["artifact_path"]) / "gridmask_dataset"
        assert train_data.exists()
        assert (train_data.parent / "images").exists()
        assert train_calls[0]["mosaic"] == 1.0

        logs = client.get(f"/api/training/runs/{run['id']}/logs").json()["text"]
        assert "gridmask dataset prepared" in logs
        assert "ultralytics training started" in logs


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


def test_training_run_can_be_cancelled_and_releases_active_lock(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def leave_run_active(run_id: int, bind, _settings) -> None:
        from app.training.runner import append_run_log, mark_training_run_running

        mark_training_run_running(run_id, bind=bind)
        append_run_log(run_id, "waiting for cancel", bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", leave_run_active)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)

        first = client.post(
            "/api/training/runs",
            json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
        )
        assert first.status_code == 200
        run_id = first.json()["id"]

        cancel_response = client.post(f"/api/training/runs/{run_id}/cancel")

        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

        logs_response = client.get(f"/api/training/runs/{run_id}/logs")

        assert logs_response.status_code == 200
        assert "training cancelled" in logs_response.json()["text"]

        second = client.post(
            "/api/training/runs",
            json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
        )

        assert second.status_code == 200

        completed_cancel = client.post(f"/api/training/runs/{run_id}/cancel")

        assert completed_cancel.status_code == 400
        assert "Training run is not active" in completed_cancel.text
