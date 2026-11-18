from __future__ import annotations

from pathlib import Path

import pytest

from app.remote.fingerprint import compute_sync_fingerprint
from app.remote.rsync import parse_rsync_progress_line
from app.training.devices import parse_ultralytics_device
from test_dataset_import_api import create_import_zip, isolated_client
from test_training_api import _create_version


def test_parse_rsync_progress_line():
    progress = parse_rsync_progress_line("  1,234,567  45%  12.34MB/s    0:00:12")
    assert progress is not None
    assert progress.bytes_done == 1234567
    assert progress.percent == 45.0
    assert progress.speed_bps == pytest.approx(12.34 * 1024 * 1024)


def test_parse_ultralytics_device_multi_gpu():
    assert parse_ultralytics_device("0") == 0
    assert parse_ultralytics_device("0,1,2,3") == [0, 1, 2, 3]
    assert parse_ultralytics_device([0, 1]) == [0, 1]
    assert parse_ultralytics_device("cpu") == "cpu"


def test_compute_sync_fingerprint_changes_with_model(tmp_path: Path):
    version_root = tmp_path / "version"
    (version_root / "images").mkdir(parents=True)
    (version_root / "images" / "a.jpg").write_bytes(b"abc")
    (version_root / "data.yaml").write_text("path: .\n")
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"weight-a")
    model_b.write_bytes(b"weight-bb")

    fp_a, files_a, bytes_a = compute_sync_fingerprint(
        project_id=1,
        version_id=2,
        model_ref="base:yolov8n.pt",
        version_root=version_root,
        model_path=model_a,
    )
    fp_b, files_b, bytes_b = compute_sync_fingerprint(
        project_id=1,
        version_id=2,
        model_ref="base:yolov8n.pt",
        version_root=version_root,
        model_path=model_b,
    )
    assert files_a == files_b == 3
    assert bytes_a != bytes_b
    assert fp_a != fp_b


def _stub_bundled_weights(tmp_path: Path, monkeypatch):
    from app.training import weights as weights_module

    model_file = tmp_path / "yolov8n.pt"
    model_file.write_bytes(b"fake-weights")
    monkeypatch.setattr(
        weights_module,
        "_bundled_model_path",
        lambda name: model_file if name == "yolov8n.pt" else None,
    )
    return model_file


def test_remote_training_requires_completed_sync(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    _stub_bundled_weights(tmp_path, monkeypatch)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)
        response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "model": "base:yolov8n.pt",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
                "device": "0",
                "host": "gpu05",
            },
        )
        assert response.status_code == 400
        assert "同步" in response.json()["detail"]


def test_remote_sync_create_and_progress_with_mocked_executor(tmp_path: Path, monkeypatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    _stub_bundled_weights(tmp_path, monkeypatch)

    executed: list[int] = []

    def fake_execute(job_id: int, bind=None, settings=None):
        executed.append(job_id)
        from datetime import UTC, datetime

        from sqlalchemy.orm import Session

        from app.core.settings import Settings
        from app.db.models import RemoteSyncJob
        from app.db.session import create_engine_for_settings

        engine = create_engine_for_settings(Settings(workspace_root=tmp_path / "workspace"))
        with Session(engine) as db:
            job = db.get(RemoteSyncJob, job_id)
            assert job is not None
            job.status = "completed"
            job.bytes_done = job.bytes_total
            job.files_done = job.files_total
            job.remote_fingerprint = job.fingerprint
            job.ended_at = datetime.now(UTC)
            db.commit()
        engine.dispose()

    monkeypatch.setattr("app.remote.router.execute_sync_job", fake_execute)
    monkeypatch.setattr(
        "app.remote.router.probe_remote_host",
        lambda settings: {
            "id": "gpu05",
            "label": "GPU05 (V100)",
            "enabled": True,
            "online": True,
            "ssh_host": "gpu05",
            "remote_root": str(settings.remote_root),
            "python": "python3",
            "ultralytics_ok": True,
            "cuda_available": True,
            "gpu_count": 8,
            "gpus": [{"index": i, "name": "Tesla V100-PCIE-32GB"} for i in range(8)],
            "error": None,
            "hostname": "gpu05",
        },
    )

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)
        hosts = client.get("/api/remote-hosts")
        assert hosts.status_code == 200
        assert hosts.json()["local"]["id"] == "local"
        assert hosts.json()["items"][0]["online"] is True

        create = client.post(
            f"/api/projects/{version['project_id']}/remote-sync",
            json={
                "version_id": version["id"],
                "model_ref": "base:yolov8n.pt",
                "host": "gpu05",
            },
        )
        assert create.status_code == 200, create.text
        job = create.json()
        assert job["status"] == "queued"
        assert job["fingerprint"]
        assert executed == [job["id"]]

        fetched = client.get(f"/api/remote-sync/{job['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "completed"
        assert fetched.json()["percent"] == 100.0
