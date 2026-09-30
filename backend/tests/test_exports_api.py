from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client
from test_prediction_api import _create_completed_run


def _write_best_weights(run: dict) -> Path:
    weights = Path(run["artifact_path"]) / "ultralytics" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"fake weights")
    return weights


def test_pt_export_registers_best_weights(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        weights = _write_best_weights(run)

        capabilities_response = client.get(f"/api/training/runs/{run['id']}/exports/capabilities")

        assert capabilities_response.status_code == 200
        capabilities = capabilities_response.json()
        assert capabilities["pt_available"] is True
        assert capabilities["weights_path"] == str(weights)

        export_response = client.post(
            f"/api/training/runs/{run['id']}/exports",
            json={"format": "pt"},
        )

        assert export_response.status_code == 200
        artifact = export_response.json()
        assert artifact["format"] == "pt"
        assert artifact["status"] == "completed"
        assert Path(artifact["artifact_path"]).exists()
        assert artifact["metadata"]["source"] == str(weights)
        assert artifact["download_url"] == f"/api/training/runs/{run['id']}/exports/{artifact['id']}/file"

        list_response = client.get(f"/api/training/runs/{run['id']}/exports")

        assert list_response.status_code == 200
        listed = list_response.json()["items"][0]
        assert listed["id"] == artifact["id"]
        assert listed["download_url"] == artifact["download_url"]

        download_response = client.get(artifact["download_url"])
        assert download_response.status_code == 200
        assert download_response.content == weights.read_bytes()

        best_response = client.get(f"/api/training/runs/{run['id']}/weights/best")
        assert best_response.status_code == 200
        assert best_response.content == weights.read_bytes()
        assert "run-" in best_response.headers.get("content-disposition", "")


def test_onnx_export_failure_is_persisted_when_ultralytics_missing(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        _write_best_weights(run)

        export_response = client.post(
            f"/api/training/runs/{run['id']}/exports",
            json={"format": "onnx"},
        )

        assert export_response.status_code == 200
        artifact = export_response.json()
        assert artifact["format"] == "onnx"
        assert artifact["status"] in {"completed", "failed"}
        if artifact["status"] == "failed":
            assert artifact["error_message"]


def test_onnx_export_can_complete_through_adapter(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def fake_execute_export(db, artifact, run):
        from app.exports.runner import complete_export, export_root

        destination = export_root(run) / f"run-{run.id}.onnx"
        destination.write_bytes(b"fake onnx")
        return complete_export(
            db,
            artifact,
            destination,
            {"source": "fake-best.pt", "format": "onnx"},
        )

    monkeypatch.setattr("app.exports.router.execute_export", fake_execute_export)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        _write_best_weights(run)

        export_response = client.post(
            f"/api/training/runs/{run['id']}/exports",
            json={"format": "onnx"},
        )

        assert export_response.status_code == 200
        artifact = export_response.json()
        assert artifact["format"] == "onnx"
        assert artifact["status"] == "completed"
        assert artifact["metadata"] == {"source": "fake-best.pt", "format": "onnx"}
        assert Path(artifact["artifact_path"]).exists()


def test_tensorrt_export_reports_unsupported_environment(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    monkeypatch.setattr("app.exports.runner.tensor_rt_available", lambda: False)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)
        _write_best_weights(run)

        capabilities_response = client.get(f"/api/training/runs/{run['id']}/exports/capabilities")

        assert capabilities_response.status_code == 200
        capabilities = capabilities_response.json()
        assert capabilities["tensorrt_available"] is False
        assert "tensorrt" in capabilities["reasons"]

        export_response = client.post(
            f"/api/training/runs/{run['id']}/exports",
            json={"format": "tensorrt"},
        )

        assert export_response.status_code == 200
        artifact = export_response.json()
        assert artifact["format"] == "tensorrt"
        assert artifact["status"] == "failed"
        assert "TensorRT export is not supported" in artifact["error_message"]


def test_export_rejects_non_completed_runs(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    def leave_run_active(run_id: int, bind, _settings) -> None:
        from app.training.runner import mark_training_run_running

        mark_training_run_running(run_id, bind=bind)

    monkeypatch.setattr("app.training.router.execute_training_run", leave_run_active)

    with isolated_client(tmp_path) as client:
        from test_training_api import _create_version

        version = _create_version(client, zip_path)
        run_response = client.post(
            "/api/training/runs",
            json={"version_id": version["id"], "epochs": 1, "image_size": 320, "batch_size": 1},
        )
        assert run_response.status_code == 200
        run = client.get(f"/api/training/runs/{run_response.json()['id']}").json()
        assert run["status"] == "running"

        export_response = client.post(
            f"/api/training/runs/{run['id']}/exports",
            json={"format": "pt"},
        )

        assert export_response.status_code == 400
        assert "Only completed runs can be exported" in export_response.text
