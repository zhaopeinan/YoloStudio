from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client
from test_prediction_api import _create_completed_run
from test_training_api import _create_version


def _trash_run(client, run_id: int) -> None:
    response = client.post(f"/api/storage/items/training_run/{run_id}/trash")
    assert response.status_code == 200


def test_training_rejects_trashed_dataset_version(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        version = _create_version(client, zip_path)
        trash_response = client.post(
            f"/api/storage/items/dataset_version/{version['id']}/trash"
        )
        assert trash_response.status_code == 200

        response = client.post(
            "/api/training/runs",
            json={
                "version_id": version["id"],
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
            },
        )

        assert response.status_code == 404


def test_training_and_experiment_workflows_hide_trashed_run(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        active_run = _create_completed_run(client, zip_path, monkeypatch)
        second_response = client.post(
            "/api/training/runs",
            json={
                "version_id": active_run["version_id"],
                "epochs": 1,
                "image_size": 320,
                "batch_size": 1,
            },
        )
        assert second_response.status_code == 200
        trashed_run = client.get(
            f"/api/training/runs/{second_response.json()['id']}"
        ).json()
        assert trashed_run["status"] == "completed"

        _trash_run(client, trashed_run["id"])

        list_response = client.get(
            f"/api/projects/{active_run['project_id']}/training/runs"
        )
        assert list_response.status_code == 200
        assert [item["id"] for item in list_response.json()["items"]] == [
            active_run["id"]
        ]

        project_summary = client.get(
            f"/api/projects/{active_run['project_id']}/training/summary"
        )
        assert project_summary.status_code == 200
        assert [item["run_id"] for item in project_summary.json()["runs"]] == [
            active_run["id"]
        ]

        guarded_requests = [
            ("get", f"/api/training/runs/{trashed_run['id']}"),
            ("get", f"/api/training/runs/{trashed_run['id']}/artifacts"),
            ("get", f"/api/training/runs/{trashed_run['id']}/logs"),
            ("get", f"/api/training/runs/{trashed_run['id']}/summary"),
            ("post", f"/api/training/runs/{trashed_run['id']}/cancel"),
        ]
        for method, path in guarded_requests:
            response = getattr(client, method)(path)
            assert response.status_code == 404, path


def test_prediction_and_export_workflows_reject_trashed_run(
    tmp_path: Path,
    monkeypatch,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        run = _create_completed_run(client, zip_path, monkeypatch)

        monkeypatch.setattr(
            "app.prediction.router.predict_images",
            lambda _run, _images, _threshold: {},
        )
        job_response = client.post(
            f"/api/training/runs/{run['id']}/prediction-jobs",
            json={"image_scope": "all", "confidence_threshold": 0.25},
        )
        assert job_response.status_code == 200
        job_id = job_response.json()["id"]

        _trash_run(client, run["id"])

        run_guarded_requests = [
            (
                "post",
                f"/api/training/runs/{run['id']}/prediction-jobs",
                {"image_scope": "all", "confidence_threshold": 0.25},
            ),
            (
                "post",
                f"/api/training/runs/{run['id']}/prediction-threshold-scan",
                {"image_scope": "all", "thresholds": [0.25, 0.5]},
            ),
            ("get", f"/api/training/runs/{run['id']}/prediction-jobs", None),
            ("get", f"/api/training/runs/{run['id']}/exports/capabilities", None),
            ("get", f"/api/training/runs/{run['id']}/exports", None),
            ("post", f"/api/training/runs/{run['id']}/exports", {"format": "pt"}),
        ]
        for method, path, payload in run_guarded_requests:
            response = getattr(client, method)(path, json=payload) if payload else getattr(
                client, method
            )(path)
            assert response.status_code == 404, path

        job_guarded_paths = [
            f"/api/prediction-jobs/{job_id}/predictions",
            f"/api/prediction-jobs/{job_id}/logs",
            f"/api/prediction-jobs/{job_id}/images/{run['image_ids'][0]}/review",
        ]
        for path in job_guarded_paths:
            response = client.get(path)
            assert response.status_code == 404, path
