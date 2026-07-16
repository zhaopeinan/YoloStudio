from __future__ import annotations

from pathlib import Path

from test_dataset_import_api import create_import_zip, isolated_client


def _import_dataset(client, zip_path: Path, dataset_name: str) -> dict:
    response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Visibility Test Project",
            "dataset_name": dataset_name,
        },
    )
    assert response.status_code == 200
    return response.json()


def _annotate_dataset(client, dataset: dict) -> None:
    class_response = client.get(
        f"/api/projects/{dataset['project_id']}/classes"
    )
    assert class_response.status_code == 200
    classes = class_response.json()["items"]
    if classes:
        class_id = classes[0]["id"]
    else:
        create_response = client.post(
            f"/api/projects/{dataset['project_id']}/classes",
            json={"name": "drone", "color": "#ef4444"},
        )
        assert create_response.status_code == 200
        class_id = create_response.json()["id"]

    images_response = client.get(
        f"/api/datasets/{dataset['dataset_id']}/images"
    )
    assert images_response.status_code == 200
    for image in images_response.json()["items"]:
        response = client.put(
            f"/api/images/{image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_id,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ]
            },
        )
        assert response.status_code == 200


def test_trashed_dataset_is_hidden_from_data_annotation_and_quality_workflows(
    tmp_path: Path,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        active = _import_dataset(client, zip_path, "active-dataset")
        trashed = _import_dataset(client, zip_path, "trashed-dataset")
        project_id = trashed["project_id"]

        class_response = client.post(
            f"/api/projects/{project_id}/classes",
            json={"name": "drone", "color": "#ef4444"},
        )
        assert class_response.status_code == 200
        class_id = class_response.json()["id"]

        images_response = client.get(
            f"/api/datasets/{trashed['dataset_id']}/images"
        )
        assert images_response.status_code == 200
        image_id = images_response.json()["items"][0]["id"]

        trash_response = client.post(
            f"/api/storage/items/dataset/{trashed['dataset_id']}/trash"
        )
        assert trash_response.status_code == 200

        projects_response = client.get("/api/projects")
        assert projects_response.status_code == 200
        project = next(
            item
            for item in projects_response.json()["items"]
            if item["id"] == project_id
        )
        assert [item["id"] for item in project["datasets"]] == [
            active["dataset_id"]
        ]

        classes_response = client.get(f"/api/projects/{project_id}/classes")
        assert classes_response.status_code == 200
        assert [item["id"] for item in classes_response.json()["items"]] == [
            class_id
        ]

        dataset_requests = [
            ("get", f"/api/datasets/{trashed['dataset_id']}/images", None),
            ("get", f"/api/datasets/{trashed['dataset_id']}/coverage", None),
            (
                "post",
                f"/api/datasets/{trashed['dataset_id']}/refresh-image-dimensions",
                None,
            ),
            ("get", f"/api/datasets/{trashed['dataset_id']}/quality", None),
            (
                "get",
                f"/api/datasets/{trashed['dataset_id']}/quality/issues",
                None,
            ),
            (
                "post",
                f"/api/datasets/{trashed['dataset_id']}/quality/apply-tags",
                {"issue_type": "all"},
            ),
            (
                "post",
                f"/api/datasets/{trashed['dataset_id']}/versions",
                {"name": "should-not-exist"},
            ),
            ("get", f"/api/datasets/{trashed['dataset_id']}/versions", None),
        ]
        for method, path, body in dataset_requests:
            response = client.request(method, path, json=body)
            assert response.status_code == 404, (method, path, response.text)

        image_requests = [
            ("get", f"/api/images/{image_id}/file", None),
            ("get", f"/api/images/{image_id}/annotations", None),
            (
                "put",
                f"/api/images/{image_id}/annotations",
                {"annotations": []},
            ),
        ]
        for method, path, body in image_requests:
            response = client.request(method, path, json=body)
            assert response.status_code == 404, (method, path, response.text)


def test_trashed_versions_are_hidden_without_hiding_active_versions(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path, "version-dataset")
        _annotate_dataset(client, dataset)

        first_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "first-version"},
        )
        second_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "second-version"},
        )
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        first_version = first_response.json()
        second_version = second_response.json()

        trash_response = client.post(
            f"/api/storage/items/dataset_version/{first_version['id']}/trash"
        )
        assert trash_response.status_code == 200

        list_response = client.get(
            f"/api/datasets/{dataset['dataset_id']}/versions"
        )
        assert list_response.status_code == 200
        assert [item["id"] for item in list_response.json()["items"]] == [
            second_version["id"]
        ]


def test_failure_filter_ignores_predictions_from_trashed_training_runs(
    tmp_path: Path,
):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path, "prediction-dataset")
        _annotate_dataset(client, dataset)
        images = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images"
        ).json()["items"]
        class_id = client.get(
            f"/api/projects/{dataset['project_id']}/classes"
        ).json()["items"][0]["id"]
        version_response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "prediction-version"},
        )
        assert version_response.status_code == 200
        version_id = version_response.json()["id"]

        from app.core.settings import Settings
        from app.db.models import Prediction, PredictionJob, TrainingRun
        from app.db.session import create_engine_for_settings
        from sqlalchemy.orm import Session

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        with Session(bind) as db:
            run = TrainingRun(
                project_id=dataset["project_id"],
                version_id=version_id,
                status="completed",
                device="cpu",
                config={"model": "yolov8n.pt"},
                artifact_path="pending",
                log_path="pending",
            )
            db.add(run)
            db.flush()

            run_root = (
                settings.workspace_root
                / "projects"
                / str(dataset["project_id"])
                / "runs"
                / str(run.id)
            )
            job_root = run_root / "predictions" / "1"
            job_root.mkdir(parents=True)
            run.artifact_path = str(run_root)
            run.log_path = str(run_root / "logs.txt")
            (run_root / "logs.txt").write_text("completed", encoding="utf-8")

            job = PredictionJob(
                run_id=run.id,
                project_id=dataset["project_id"],
                status="completed",
                artifact_path=str(job_root),
                log_path=str(job_root / "logs.txt"),
                image_count=1,
                prediction_count=1,
                false_positive_count=1,
            )
            db.add(job)
            db.flush()
            db.add(
                Prediction(
                    run_id=run.id,
                    job_id=job.id,
                    image_id=images[0]["id"],
                    class_id=class_id,
                    x_center=0.5,
                    y_center=0.5,
                    width=0.25,
                    height=0.25,
                    confidence=0.8,
                    failure_type="false_positive",
                )
            )
            db.commit()
            run_id = run.id
        bind.dispose()

        before_trash = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images"
            "?failure_type=false_positive"
        )
        assert before_trash.status_code == 200
        assert [item["id"] for item in before_trash.json()["items"]] == [
            images[0]["id"]
        ]

        trash_response = client.post(
            f"/api/storage/items/training_run/{run_id}/trash"
        )
        assert trash_response.status_code == 200

        after_trash = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images"
            "?failure_type=false_positive"
        )
        assert after_trash.status_code == 200
        assert after_trash.json()["items"] == []
        assert after_trash.json()["total"] == 0
