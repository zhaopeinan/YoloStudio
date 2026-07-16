from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from test_dataset_import_api import create_import_zip, isolated_client


def _create_storage_fixture(client, tmp_path: Path) -> dict[str, int]:
    zip_path = tmp_path / "storage-sample.zip"
    create_import_zip(zip_path)

    import_response = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Storage Test Project",
            "dataset_name": "storage-sample",
        },
    )
    assert import_response.status_code == 200
    dataset = import_response.json()

    class_response = client.post(
        f"/api/projects/{dataset['project_id']}/classes",
        json={"name": "无人机", "color": "#ef4444"},
    )
    assert class_response.status_code == 200
    class_id = class_response.json()["id"]

    images_response = client.get(f"/api/datasets/{dataset['dataset_id']}/images")
    assert images_response.status_code == 200
    for image in images_response.json()["items"]:
        annotation_response = client.put(
            f"/api/images/{image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_id,
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
        json={"name": "storage-version"},
    )
    assert version_response.status_code == 200
    version = version_response.json()

    from app.core.settings import Settings
    from app.db.models import ExportArtifact, PredictionJob, TrainingRun
    from app.db.session import create_engine_for_settings

    settings = Settings(workspace_root=tmp_path / "workspace")
    bind = create_engine_for_settings(settings)
    with Session(bind) as db:
        runs_root = settings.workspace_root / "projects" / str(dataset["project_id"]) / "runs"
        completed_root = runs_root / "1"
        running_root = runs_root / "2"
        completed_root.mkdir(parents=True)
        running_root.mkdir(parents=True)
        (completed_root / "best.pt").write_bytes(b"completed-model")
        (running_root / "logs.txt").write_text("training", encoding="utf-8")

        completed_run = TrainingRun(
            project_id=dataset["project_id"],
            version_id=version["id"],
            status="completed",
            device="cpu",
            config={"model": "yolov8n.pt"},
            artifact_path=str(completed_root),
            log_path=str(completed_root / "logs.txt"),
        )
        running_run = TrainingRun(
            project_id=dataset["project_id"],
            version_id=version["id"],
            status="running",
            device="mps",
            config={"model": "yolov10n.pt"},
            artifact_path=str(running_root),
            log_path=str(running_root / "logs.txt"),
        )
        db.add_all([completed_run, running_run])
        db.flush()

        prediction_root = completed_root / "predictions" / "1"
        prediction_root.mkdir(parents=True)
        db.add(
            PredictionJob(
                run_id=completed_run.id,
                project_id=dataset["project_id"],
                status="completed",
                artifact_path=str(prediction_root),
                log_path=str(prediction_root / "logs.txt"),
            )
        )
        db.add(
            ExportArtifact(
                run_id=completed_run.id,
                project_id=dataset["project_id"],
                format="onnx",
                status="completed",
                artifact_path=str(completed_root / "exports" / "model.onnx"),
            )
        )
        db.commit()
        completed_run_id = completed_run.id
        running_run_id = running_run.id
    bind.dispose()

    return {
        "project_id": dataset["project_id"],
        "dataset_id": dataset["dataset_id"],
        "version_id": version["id"],
        "completed_run_id": completed_run_id,
        "running_run_id": running_run_id,
    }


def test_storage_catalog_lists_active_datasets_and_versions(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        response = client.get("/api/storage/items")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_size_bytes"] > 0
        assert len(payload["items"]) == 2

        dataset = next(
            item for item in payload["items"] if item["entity_type"] == "dataset"
        )
        version = next(
            item
            for item in payload["items"]
            if item["entity_type"] == "dataset_version"
        )

        assert dataset["entity_id"] == fixture["dataset_id"]
        assert dataset["project_name"] == "Storage Test Project"
        assert dataset["image_count"] == 2
        assert dataset["annotation_count"] == 2
        assert dataset["size_bytes"] > 0
        assert dataset["protected"] is True
        assert dataset["blockers"] == [
            {
                "entity_type": "dataset_version",
                "entity_id": fixture["version_id"],
                "display_name": "storage-version",
                "status": None,
            }
        ]
        assert Path(dataset["artifact_path"]) == (
            tmp_path
            / "workspace"
            / "projects"
            / str(fixture["project_id"])
            / "datasets"
            / str(fixture["dataset_id"])
        ).resolve()

        assert version["entity_id"] == fixture["version_id"]
        assert version["image_count"] == 2
        assert version["annotation_count"] == 2
        assert version["split_counts"] == {"train": 1, "val": 1, "test": 0}
        assert version["size_bytes"] > 0
        assert version["protected"] is True
        assert {blocker["status"] for blocker in version["blockers"]} == {
            "completed",
            "running",
        }


def test_storage_detail_lists_classes_and_non_trashed_runs(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        response = client.get(
            f"/api/storage/items/dataset_version/{fixture['version_id']}"
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["class_names"] == ["无人机"]
        assert {run["status"] for run in payload["related_runs"]} == {
            "completed",
            "running",
        }
        completed_run = next(
            run for run in payload["related_runs"] if run["status"] == "completed"
        )
        assert completed_run["id"] == fixture["completed_run_id"]
        assert completed_run["model"] == "yolov8n.pt"
        assert completed_run["size_bytes"] > 0
        assert completed_run["prediction_job_count"] == 1
        assert completed_run["export_count"] == 1


def test_storage_detail_rejects_unknown_entity_type_and_missing_item(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        invalid_type = client.get("/api/storage/items/training_run/1")
        missing_item = client.get("/api/storage/items/dataset/999")

        assert invalid_type.status_code == 404
        assert invalid_type.json()["detail"] == "不支持的存储对象类型"
        assert missing_item.status_code == 404
        assert missing_item.json()["detail"] == "存储对象不存在或已移入回收站"


def test_storage_detail_excludes_trashed_related_runs(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        from app.core.settings import Settings
        from app.db.models import TrashItem
        from app.db.session import create_engine_for_settings

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        now = datetime.now(UTC)
        with Session(bind) as db:
            db.add(
                TrashItem(
                    entity_type="training_run",
                    entity_id=fixture["completed_run_id"],
                    project_id=fixture["project_id"],
                    dataset_id=fixture["dataset_id"],
                    version_id=fixture["version_id"],
                    display_name=f"训练任务 #{fixture['completed_run_id']}",
                    original_path="projects/1/runs/1",
                    trash_path=".trash/1",
                    size_bytes=0,
                    summary={},
                    status="active",
                    deleted_at=now,
                    purge_after=now + timedelta(days=30),
                )
            )
            db.commit()
        bind.dispose()

        response = client.get(
            f"/api/storage/items/dataset_version/{fixture['version_id']}"
        )

        assert response.status_code == 200
        payload = response.json()
        assert [run["id"] for run in payload["related_runs"]] == [
            fixture["running_run_id"]
        ]
        assert [blocker["entity_id"] for blocker in payload["blockers"]] == [
            fixture["running_run_id"]
        ]


def test_storage_catalog_rejects_version_path_outside_workspace(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        from app.core.settings import Settings
        from app.db.models import DatasetVersion
        from app.db.session import create_engine_for_settings

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        with Session(bind) as db:
            version = db.get(DatasetVersion, fixture["version_id"])
            assert version is not None
            version.artifact_path = str(tmp_path.parent)
            db.commit()
        bind.dispose()

        response = client.get("/api/storage/items")

        assert response.status_code == 400
        assert response.json()["detail"] == "数据路径不在当前工作空间内。"


def _set_run_status(tmp_path: Path, run_id: int, status: str) -> None:
    from app.core.settings import Settings
    from app.db.models import TrainingRun
    from app.db.session import create_engine_for_settings

    settings = Settings(workspace_root=tmp_path / "workspace")
    bind = create_engine_for_settings(settings)
    with Session(bind) as db:
        run = db.get(TrainingRun, run_id)
        assert run is not None
        run.status = status
        db.commit()
    bind.dispose()


@pytest.mark.parametrize(
    ("child_type", "status", "expected_entity_type", "expected_name"),
    [
        ("prediction", "running", "prediction_job", "预测任务"),
        ("export", "queued", "export_artifact", "模型导出"),
    ],
)
def test_completed_run_trash_is_blocked_by_active_child_job(
    tmp_path: Path,
    child_type: str,
    status: str,
    expected_entity_type: str,
    expected_name: str,
):
    from app.core.settings import Settings
    from app.db.models import ExportArtifact, PredictionJob
    from app.db.session import create_engine_for_settings

    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        bind = create_engine_for_settings(
            Settings(workspace_root=tmp_path / "workspace")
        )
        with Session(bind) as db:
            model = PredictionJob if child_type == "prediction" else ExportArtifact
            child = db.scalar(
                select(model).where(model.run_id == fixture["completed_run_id"])
            )
            assert child is not None
            child.status = status
            db.commit()
            child_id = child.id
        bind.dispose()

        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["message"] == "训练任务仍有关联任务正在处理中，无法移入回收站。"
        assert detail["blockers"] == [
            {
                "entity_type": expected_entity_type,
                "entity_id": child_id,
                "display_name": f"{expected_name} #{child_id}",
                "status": status,
            }
        ]


def test_storage_trash_enforces_dependencies_and_moves_child_to_parent(
    tmp_path: Path,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        workspace = (tmp_path / "workspace").resolve()

        dataset_response = client.post(
            f"/api/storage/items/dataset/{fixture['dataset_id']}/trash"
        )
        assert dataset_response.status_code == 409
        assert dataset_response.json()["detail"] == {
            "message": "该数据集仍有关联的标注版本，请先将关联版本移入回收站。",
            "blockers": [
                {
                    "entity_type": "dataset_version",
                    "entity_id": fixture["version_id"],
                    "display_name": "storage-version",
                    "status": None,
                }
            ],
        }

        version_response = client.post(
            f"/api/storage/items/dataset_version/{fixture['version_id']}/trash"
        )
        assert version_response.status_code == 409
        assert version_response.json()["detail"]["message"] == (
            "该标注版本仍有关联的训练任务，请先将关联训练任务移入回收站。"
        )
        assert {
            (blocker["entity_id"], blocker["status"])
            for blocker in version_response.json()["detail"]["blockers"]
        } == {
            (fixture["completed_run_id"], "completed"),
            (fixture["running_run_id"], "running"),
        }

        active_response = client.post(
            f"/api/storage/items/training_run/{fixture['running_run_id']}/trash"
        )
        assert active_response.status_code == 409
        assert active_response.json()["detail"] == {
            "message": "训练任务仍在运行中，无法移入回收站。",
            "blockers": [
                {
                    "entity_type": "training_run",
                    "entity_id": fixture["running_run_id"],
                    "display_name": f"训练任务 #{fixture['running_run_id']}",
                    "status": "running",
                }
            ],
        }

        completed_path = (
            workspace
            / "projects"
            / str(fixture["project_id"])
            / "runs"
            / str(fixture["completed_run_id"])
        )
        completed_response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )
        assert completed_response.status_code == 200
        completed_trash = completed_response.json()
        assert completed_trash["entity_type"] == "training_run"
        assert completed_trash["status"] == "active"
        assert completed_trash["size_bytes"] > 0
        assert Path(completed_trash["trash_path"]).is_dir()
        assert Path(completed_trash["trash_path"]).is_relative_to(workspace / ".trash")
        assert not completed_path.exists()

        duplicate_response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )
        assert duplicate_response.status_code == 409
        assert duplicate_response.json()["detail"] == {
            "message": "该对象已在回收站中。",
            "blockers": [],
        }

        _set_run_status(tmp_path, fixture["running_run_id"], "cancelled")
        running_response = client.post(
            f"/api/storage/items/training_run/{fixture['running_run_id']}/trash"
        )
        assert running_response.status_code == 200

        version_path = (
            workspace
            / "projects"
            / str(fixture["project_id"])
            / "versions"
            / str(fixture["version_id"])
        )
        moved_version = client.post(
            f"/api/storage/items/dataset_version/{fixture['version_id']}/trash"
        )
        assert moved_version.status_code == 200
        assert Path(moved_version.json()["trash_path"]).is_dir()
        assert not version_path.exists()

        dataset_path = (
            workspace
            / "projects"
            / str(fixture["project_id"])
            / "datasets"
            / str(fixture["dataset_id"])
        )
        moved_dataset = client.post(
            f"/api/storage/items/dataset/{fixture['dataset_id']}/trash"
        )
        assert moved_dataset.status_code == 200
        assert Path(moved_dataset.json()["trash_path"]).is_dir()
        assert not dataset_path.exists()


def test_storage_trash_rejects_aliased_run_artifact_path(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        from app.core.settings import Settings
        from app.db.models import TrainingRun
        from app.db.session import create_engine_for_settings

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        with Session(bind) as db:
            run = db.get(TrainingRun, fixture["completed_run_id"])
            assert run is not None
            run.artifact_path = str(
                settings.workspace_root
                / "projects"
                / str(fixture["project_id"])
                / "runs"
                / str(fixture["running_run_id"])
            )
            db.commit()
        bind.dispose()

        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "存储记录路径与规范目录不一致"


def test_storage_trash_rejects_traversal_in_stored_path(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)

        from app.core.settings import Settings
        from app.db.models import TrainingRun
        from app.db.session import create_engine_for_settings

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        with Session(bind) as db:
            run = db.get(TrainingRun, fixture["completed_run_id"])
            assert run is not None
            run.artifact_path = (
                f"projects/{fixture['project_id']}/runs/../runs/{run.id}"
            )
            db.commit()
        bind.dispose()

        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "存储路径不能包含目录穿越"


def test_storage_trash_rejects_symlink_escape(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        workspace = (tmp_path / "workspace").resolve()
        run_path = (
            workspace
            / "projects"
            / str(fixture["project_id"])
            / "runs"
            / str(fixture["completed_run_id"])
        )
        outside = tmp_path / "outside-run"
        outside.mkdir()
        shutil.rmtree(run_path)
        run_path.symlink_to(outside, target_is_directory=True)

        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "数据路径不在当前工作空间内。"


@pytest.mark.parametrize("status", ["queued", "preparing", "running"])
def test_storage_trash_rejects_every_active_run_status(
    tmp_path: Path,
    status: str,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        _set_run_status(tmp_path, fixture["completed_run_id"], status)

        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 409
        assert response.json()["detail"]["message"] == (
            "训练任务仍在运行中，无法移入回收站。"
        )
        assert response.json()["detail"]["blockers"][0]["status"] == status


def test_storage_trash_removes_pending_record_when_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        original_rename = Path.rename

        def fail_run_rename(path: Path, target: Path) -> Path:
            if path.name == str(fixture["completed_run_id"]):
                raise OSError("simulated rename failure")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_run_rename)
        response = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        )

        assert response.status_code == 500
        assert "移动数据到回收站失败" in response.json()["detail"]

        from app.core.settings import Settings
        from app.db.models import TrashItem
        from app.db.session import create_engine_for_settings

        settings = Settings(workspace_root=tmp_path / "workspace")
        bind = create_engine_for_settings(settings)
        with Session(bind) as db:
            assert db.query(TrashItem).count() == 0
        bind.dispose()

        run_path = (
            settings.workspace_root
            / "projects"
            / str(fixture["project_id"])
            / "runs"
            / str(fixture["completed_run_id"])
        )
        assert run_path.is_dir()
