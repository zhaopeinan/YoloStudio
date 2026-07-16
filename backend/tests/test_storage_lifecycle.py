from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import shutil

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import (
    Annotation,
    Base,
    ClassDef,
    Dataset,
    DatasetVersion,
    ExportArtifact,
    Image,
    Prediction,
    PredictionJob,
    Project,
    RunMetric,
    TrainingRun,
    TrashItem,
)
from app.db.session import create_engine_for_settings, get_db
from app.main import app
from app.storage.service import reconcile_trash
from test_dataset_import_api import isolated_client
from test_storage_api import _create_storage_fixture, _set_run_status


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path / "workspace")


@contextmanager
def _db(tmp_path: Path):
    engine = create_engine_for_settings(_settings(tmp_path))
    try:
        with Session(engine) as db:
            yield db
    finally:
        engine.dispose()


def _trash_family(client: TestClient, tmp_path: Path) -> tuple[dict[str, int], list[dict]]:
    fixture = _create_storage_fixture(client, tmp_path)
    _set_run_status(tmp_path, fixture["running_run_id"], "cancelled")
    items = []
    for entity_type, entity_id in (
        ("training_run", fixture["completed_run_id"]),
        ("training_run", fixture["running_run_id"]),
        ("dataset_version", fixture["version_id"]),
        ("dataset", fixture["dataset_id"]),
    ):
        response = client.post(f"/api/storage/items/{entity_type}/{entity_id}/trash")
        assert response.status_code == 200
        items.append(response.json())
    return fixture, items


def test_trash_list_reconciles_and_restore_enforces_parent_order(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture, items = _trash_family(client, tmp_path)
        by_type = {item["entity_type"]: item for item in items}
        run_item = items[0]

        listed = client.get("/api/storage/trash")
        assert listed.status_code == 200
        assert len(listed.json()["items"]) == 4

        blocked_run = client.post(f"/api/storage/trash/{run_item['id']}/restore")
        assert blocked_run.status_code == 409
        assert blocked_run.json()["detail"]["message"].startswith("请先恢复标注版本")

        blocked_version = client.post(
            f"/api/storage/trash/{by_type['dataset_version']['id']}/restore"
        )
        assert blocked_version.status_code == 409
        assert blocked_version.json()["detail"]["message"].startswith("请先恢复原始数据集")

        for item in (
            by_type["dataset"],
            by_type["dataset_version"],
            run_item,
        ):
            restored = client.post(f"/api/storage/trash/{item['id']}/restore")
            assert restored.status_code == 200
            assert Path(item["original_path"]).is_dir()
            assert not Path(item["trash_path"]).exists()

        with _db(tmp_path) as db:
            assert db.get(Dataset, fixture["dataset_id"]) is not None
            assert db.get(DatasetVersion, fixture["version_id"]) is not None
            assert db.get(TrainingRun, fixture["completed_run_id"]) is not None


def test_restore_rejects_occupied_original_path_and_keeps_active(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        Path(trashed["original_path"]).mkdir(parents=True)

        response = client.post(f"/api/storage/trash/{trashed['id']}/restore")

        assert response.status_code == 409
        assert "原目录已存在" in response.json()["detail"]["message"]
        assert Path(trashed["trash_path"]).is_dir()
        with _db(tmp_path) as db:
            assert db.get(TrashItem, trashed["id"]).status == "active"


def test_permanent_purge_requires_exact_name_and_deletes_run_children(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        with _db(tmp_path) as db:
            job = db.scalar(
                select(PredictionJob).where(
                    PredictionJob.run_id == fixture["completed_run_id"]
                )
            )
            image = db.scalar(select(Image).where(Image.dataset_id == fixture["dataset_id"]))
            class_def = db.scalar(
                select(ClassDef).where(ClassDef.project_id == fixture["project_id"])
            )
            db.add(
                Prediction(
                    run_id=fixture["completed_run_id"],
                    job_id=job.id,
                    image_id=image.id,
                    class_id=class_def.id,
                    x_center=0.5,
                    y_center=0.5,
                    width=0.2,
                    height=0.2,
                    confidence=0.9,
                    failure_type="matched",
                )
            )
            db.add(
                RunMetric(
                    run_id=fixture["completed_run_id"],
                    epoch=1,
                    name="metrics/mAP50",
                    value=0.5,
                )
            )
            db.commit()

        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        wrong = client.request(
            "DELETE",
            f"/api/storage/trash/{trashed['id']}",
            json={"confirm_name": "wrong"},
        )
        assert wrong.status_code == 400
        assert Path(trashed["trash_path"]).is_dir()

        purged = client.request(
            "DELETE",
            f"/api/storage/trash/{trashed['id']}",
            json={"confirm_name": trashed["display_name"]},
        )
        assert purged.status_code == 200
        assert not Path(trashed["trash_path"]).exists()

        with _db(tmp_path) as db:
            run_id = fixture["completed_run_id"]
            assert db.get(TrainingRun, run_id) is None
            assert db.scalar(select(func.count(Prediction.id)).where(Prediction.run_id == run_id)) == 0
            assert db.scalar(select(func.count(PredictionJob.id)).where(PredictionJob.run_id == run_id)) == 0
            assert db.scalar(select(func.count(ExportArtifact.id)).where(ExportArtifact.run_id == run_id)) == 0
            assert db.scalar(select(func.count(RunMetric.id)).where(RunMetric.run_id == run_id)) == 0
            assert db.get(Project, fixture["project_id"]) is not None
            assert db.scalar(select(func.count(ClassDef.id))) == 1


def test_expired_purge_runs_child_first_and_preserves_project_classes(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture, _ = _trash_family(client, tmp_path)
        with _db(tmp_path) as db:
            for item in db.scalars(select(TrashItem)).all():
                item.purge_after = datetime.now(UTC) - timedelta(seconds=1)
            db.commit()

        response = client.post("/api/storage/trash/purge-expired")
        assert response.status_code == 200
        assert response.json() == {"purged_count": 4, "failed_count": 0}

        with _db(tmp_path) as db:
            assert db.scalar(select(func.count(TrashItem.id))) == 0
            assert db.scalar(select(func.count(TrainingRun.id))) == 0
            assert db.get(DatasetVersion, fixture["version_id"]) is None
            assert db.get(Dataset, fixture["dataset_id"]) is None
            assert db.get(Project, fixture["project_id"]) is not None
            assert db.scalar(select(func.count(ClassDef.id))) == 1
            assert db.scalar(select(func.count(Annotation.id))) == 0
            assert db.scalar(select(func.count(Image.id))) == 0


@pytest.mark.parametrize(
    ("status", "source_exists", "trash_exists", "expected_status", "record_exists"),
    [
        ("pending_move", True, False, None, False),
        ("pending_move", False, True, "active", True),
        ("pending_restore", False, True, "active", True),
        ("pending_restore", True, False, None, False),
        ("pending_move", True, True, "error", True),
        ("pending_restore", False, False, "error", True),
    ],
)
def test_reconcile_pending_path_states(
    tmp_path: Path,
    status: str,
    source_exists: bool,
    trash_exists: bool,
    expected_status: str | None,
    record_exists: bool,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        source = Path(trashed["original_path"])
        trash = Path(trashed["trash_path"])
        if source_exists:
            source.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(source, ignore_errors=True)
        if trash_exists:
            trash.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(trash, ignore_errors=True)

        with _db(tmp_path) as db:
            item = db.get(TrashItem, trashed["id"])
            item.status = status
            db.commit()
            reconcile_trash(db, _settings(tmp_path))

        with _db(tmp_path) as db:
            item = db.get(TrashItem, trashed["id"])
            assert (item is not None) is record_exists
            if item is not None:
                assert item.status == expected_status


def test_purge_db_failure_restores_staged_directory(tmp_path: Path, monkeypatch):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        original_commit = Session.commit

        def fail_commit(_session):
            raise SQLAlchemyError("simulated commit failure")

        monkeypatch.setattr(Session, "commit", fail_commit)
        response = client.request(
            "DELETE",
            f"/api/storage/trash/{trashed['id']}",
            json={"confirm_name": trashed["display_name"]},
        )
        monkeypatch.setattr(Session, "commit", original_commit)

        assert response.status_code == 500
        assert Path(trashed["trash_path"]).is_dir()
        assert not (_settings(tmp_path).workspace_root / ".trash" / ".purging" / str(trashed["id"])).exists()
        with _db(tmp_path) as db:
            assert db.get(TrashItem, trashed["id"]) is not None
            assert db.get(TrainingRun, fixture["completed_run_id"]) is not None


def test_purge_writes_marker_before_staging_and_cleans_it_on_rename_failure(
    tmp_path: Path,
    monkeypatch,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        trash = Path(trashed["trash_path"])
        original_rename = Path.rename

        def fail_staging_rename(path: Path, target: Path) -> Path:
            if path == trash and target.parent.name == ".purging":
                assert (path / ".purge.json").is_file()
                raise OSError("simulated staging rename failure")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_staging_rename)
        response = client.request(
            "DELETE",
            f"/api/storage/trash/{trashed['id']}",
            json={"confirm_name": trashed["display_name"]},
        )

        assert response.status_code == 500
        assert trash.is_dir()
        assert not (trash / ".purge.json").exists()
        assert not (
            _settings(tmp_path).workspace_root
            / ".trash"
            / ".purging"
            / str(trashed["id"])
            / ".purge.json"
        ).exists()


def test_reconcile_cleans_abandoned_staged_directory_after_db_purge(
    tmp_path: Path,
    monkeypatch,
):
    import app.storage.service as storage_service

    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        staged = (
            _settings(tmp_path).workspace_root
            / ".trash"
            / ".purging"
            / str(trashed["id"])
        )
        original_rmtree = storage_service.shutil.rmtree

        monkeypatch.setattr(storage_service.shutil, "rmtree", lambda *_args, **_kwargs: None)
        response = client.request(
            "DELETE",
            f"/api/storage/trash/{trashed['id']}",
            json={"confirm_name": trashed["display_name"]},
        )
        monkeypatch.setattr(storage_service.shutil, "rmtree", original_rmtree)

        assert response.status_code == 200
        assert staged.is_dir()
        assert (staged / ".purge.json").is_file()
        with _db(tmp_path) as db:
            assert db.get(TrashItem, trashed["id"]) is None
            assert db.get(TrainingRun, fixture["completed_run_id"]) is None
            reconcile_trash(db, _settings(tmp_path))
        assert not staged.exists()


def test_reconcile_restores_interrupted_purge_to_canonical_trash_path(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        trash = Path(trashed["trash_path"])
        staged = (
            _settings(tmp_path).workspace_root
            / ".trash"
            / ".purging"
            / str(trashed["id"])
        )
        staged.parent.mkdir(parents=True, exist_ok=True)
        (trash / ".purge.json").write_text(
            json.dumps(
                {
                    "entity_type": "training_run",
                    "entity_id": fixture["completed_run_id"],
                }
            ),
            encoding="utf-8",
        )
        trash.rename(staged)

        with _db(tmp_path) as db:
            summary = reconcile_trash(db, _settings(tmp_path))

        assert summary.reconciled_count == 1
        assert summary.error_count == 0
        assert trash.is_dir()
        assert not staged.exists()
        assert not (trash / ".purge.json").exists()
        with _db(tmp_path) as db:
            item = db.get(TrashItem, trashed["id"])
            assert item is not None
            assert item.status == "active"
            assert item.error_message is None
            assert db.get(TrainingRun, fixture["completed_run_id"]) is not None


@pytest.mark.parametrize("conflict", ["trash", "source"])
def test_reconcile_interrupted_purge_conflict_marks_error_without_deleting_paths(
    tmp_path: Path,
    conflict: str,
):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        source = Path(trashed["original_path"])
        trash = Path(trashed["trash_path"])
        staged = (
            _settings(tmp_path).workspace_root
            / ".trash"
            / ".purging"
            / str(trashed["id"])
        )
        staged.parent.mkdir(parents=True, exist_ok=True)
        (trash / ".purge.json").write_text(
            json.dumps(
                {
                    "entity_type": "training_run",
                    "entity_id": fixture["completed_run_id"],
                }
            ),
            encoding="utf-8",
        )
        trash.rename(staged)
        conflict_path = trash if conflict == "trash" else source
        conflict_path.mkdir(parents=True)
        (conflict_path / "keep.txt").write_text("keep", encoding="utf-8")

        with _db(tmp_path) as db:
            summary = reconcile_trash(db, _settings(tmp_path))

        assert summary.reconciled_count == 0
        assert summary.error_count == 1
        assert staged.is_dir()
        assert (staged / ".purge.json").is_file()
        assert conflict_path.is_dir()
        assert (conflict_path / "keep.txt").read_text(encoding="utf-8") == "keep"
        with _db(tmp_path) as db:
            item = db.get(TrashItem, trashed["id"])
            assert item is not None
            assert item.status == "error"
            assert "冲突" in (item.error_message or "")
            assert db.get(TrainingRun, fixture["completed_run_id"]) is not None


def test_restore_db_failure_leaves_recoverable_pending_state(tmp_path: Path, monkeypatch):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        trashed = client.post(
            f"/api/storage/items/training_run/{fixture['completed_run_id']}/trash"
        ).json()
        original_commit = Session.commit
        commit_count = 0

        def fail_second_commit(session):
            nonlocal commit_count
            commit_count += 1
            if commit_count == 2:
                raise SQLAlchemyError("simulated final commit failure")
            return original_commit(session)

        monkeypatch.setattr(Session, "commit", fail_second_commit)
        response = client.post(f"/api/storage/trash/{trashed['id']}/restore")
        monkeypatch.setattr(Session, "commit", original_commit)

        assert response.status_code == 500
        assert Path(trashed["original_path"]).is_dir()
        assert not Path(trashed["trash_path"]).exists()
        with _db(tmp_path) as db:
            item = db.get(TrashItem, trashed["id"])
            assert item is not None
            assert item.status == "pending_restore"

        reconciled = client.get("/api/storage/trash")
        assert reconciled.status_code == 200
        assert all(item["id"] != trashed["id"] for item in reconciled.json()["items"])


def test_expired_parent_with_remaining_child_is_retained_as_error(tmp_path: Path):
    with isolated_client(tmp_path) as client:
        fixture = _create_storage_fixture(client, tmp_path)
        settings = _settings(tmp_path)
        source = (
            settings.workspace_root
            / "projects"
            / str(fixture["project_id"])
            / "datasets"
            / str(fixture["dataset_id"])
        )
        trash_root = settings.workspace_root / ".trash"
        trash_root.mkdir(parents=True, exist_ok=True)

        with _db(tmp_path) as db:
            item = TrashItem(
                entity_type="dataset",
                entity_id=fixture["dataset_id"],
                project_id=fixture["project_id"],
                dataset_id=fixture["dataset_id"],
                version_id=None,
                display_name="storage-sample",
                original_path=str(source),
                trash_path="pending",
                size_bytes=1,
                summary={},
                status="active",
                deleted_at=datetime.now(UTC) - timedelta(days=31),
                purge_after=datetime.now(UTC) - timedelta(days=1),
            )
            db.add(item)
            db.flush()
            trash = trash_root / f"{item.id}-dataset-{fixture['dataset_id']}"
            item.trash_path = str(trash)
            db.commit()
            trash_id = item.id
        source.rename(trash)

        response = client.post("/api/storage/trash/purge-expired")
        assert response.status_code == 200
        assert response.json() == {"purged_count": 0, "failed_count": 1}

        with _db(tmp_path) as db:
            item = db.get(TrashItem, trash_id)
            assert item is not None
            assert item.status == "error"
            assert "关联标注版本" in item.error_message
            assert db.get(Dataset, fixture["dataset_id"]) is not None
            assert db.get(DatasetVersion, fixture["version_id"]) is not None
        assert trash.is_dir()


def test_startup_maintenance_uses_overridden_workspace_only(tmp_path: Path, monkeypatch):
    import app.main as main_module

    settings = _settings(tmp_path)
    engine = create_engine_for_settings(settings)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    source = settings.workspace_root / "projects" / "1" / "datasets" / "1"
    source.mkdir(parents=True)
    with Session(engine) as db:
        project = Project(name="startup-project", workspace_path=str(settings.workspace_root))
        db.add(project)
        db.flush()
        dataset = Dataset(
            project_id=project.id,
            name="startup-dataset",
            source_type="folder",
        )
        db.add(dataset)
        db.flush()
        item = TrashItem(
            entity_type="dataset",
            entity_id=dataset.id,
            project_id=project.id,
            dataset_id=dataset.id,
            version_id=None,
            display_name=dataset.name,
            original_path=str(source),
            trash_path=str(settings.workspace_root / ".trash" / "1-dataset-1"),
            size_bytes=0,
            summary={},
            status="pending_move",
            deleted_at=datetime.now(UTC),
            purge_after=datetime.now(UTC) + timedelta(days=30),
        )
        db.add(item)
        db.commit()

    def override_get_db():
        with session_factory() as db:
            yield db

    seen: list[Path] = []
    original = main_module.run_storage_maintenance

    def guarded(settings_arg, engine_arg=None):
        assert settings_arg.workspace_root.resolve() == settings.workspace_root.resolve()
        seen.append(settings_arg.workspace_root.resolve())
        return original(settings_arg, engine_arg)

    monkeypatch.setattr(main_module, "run_storage_maintenance", guarded)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app):
            pass
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert seen == [settings.workspace_root.resolve()]
    check_engine = create_engine_for_settings(settings)
    try:
        with Session(check_engine) as db:
            assert db.scalar(select(func.count(TrashItem.id))) == 0
    finally:
        check_engine.dispose()
