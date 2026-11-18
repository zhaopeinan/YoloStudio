from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import DatasetVersion, RunMetric, TrainingRun
from app.db.session import SessionLocal
from app.remote.fingerprint import compute_sync_fingerprint
from app.remote.paths import remote_run_root, remote_scripts_root, remote_version_root, remote_weight_path
from app.remote.rsync import rsync_path, rsync_pull
from app.remote.ssh import ensure_remote_dirs, quote_remote, run_ssh
from app.remote.sync import find_matching_completed_sync
from app.training.runner import (
    append_run_log,
    complete_training_run,
    fail_training_run,
    mark_training_run_running,
    run_post_training_threshold_scan,
    ultralytics_augmentation_kwargs,
)
from app.training.weights import resolve_training_model


WORKER_FILENAME = "train_worker.py"
LOCAL_WORKER = Path(__file__).with_name(WORKER_FILENAME)


def _now() -> datetime:
    return datetime.now(UTC)


def _session_factory(bind=None):
    return sessionmaker(
        bind=bind or SessionLocal.kw["bind"],
        autoflush=False,
        autocommit=False,
        future=True,
    )


def _read_remote_status(settings: Settings, remote_status_path: Path) -> dict | None:
    result = run_ssh(
        settings,
        f"if [ -f {quote_remote(remote_status_path)} ]; then cat {quote_remote(remote_status_path)}; fi",
        timeout=30,
        check=False,
    )
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _pull_remote_run_artifacts(
    settings: Settings,
    *,
    project_id: int,
    run_id: int,
    local_artifact_root: Path,
) -> None:
    remote_root = remote_run_root(settings, project_id, run_id)
    rsync_pull(settings, str(remote_root), local_artifact_root, is_directory=True)


def _ingest_local_metrics(run_id: int, artifact_root: Path, bind=None) -> None:
    metrics_path = artifact_root / "metrics.jsonl"
    if not metrics_path.is_file():
        return
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        existing = {
            (metric.epoch, metric.name, float(metric.value))
            for metric in db.scalars(
                select(RunMetric).where(RunMetric.run_id == run_id)
            ).all()
        }
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(payload.get("name") or "")
            value = payload.get("value")
            if not name or not isinstance(value, (int, float)):
                continue
            epoch = payload.get("epoch") if isinstance(payload.get("epoch"), int) else None
            key = (epoch, name, float(value))
            if key in existing:
                continue
            existing.add(key)
            db.add(RunMetric(run_id=run_id, epoch=epoch, name=name, value=float(value)))
        db.commit()


def cancel_remote_training_run(run_id: int, settings: Settings | None = None, bind=None) -> None:
    settings = settings or get_settings()
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        project_id = run.project_id
        remote_root = remote_run_root(settings, project_id, run_id)
        pid_file = remote_root / "remote.pid"
    run_ssh(
        settings,
        (
            f"if [ -f {quote_remote(pid_file)} ]; then "
            f"kill $(cat {quote_remote(pid_file)}) 2>/dev/null || true; "
            f"fi; "
            f"echo cancelled > {quote_remote(remote_root / 'cancel.flag')}"
        ),
        timeout=30,
        check=False,
    )


def execute_remote_training_run(
    run_id: int,
    bind=None,
    settings: Settings | None = None,
) -> None:
    settings = settings or get_settings()
    session_factory = _session_factory(bind)

    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "preparing"
        run.started_at = _now()
        db.commit()
        db.refresh(run)

        version = db.get(DatasetVersion, run.version_id)
        if version is None:
            fail_training_run(run_id, "Dataset version was not found", bind=bind)
            return

        config = dict(run.config)
        host = str(config.get("host") or settings.remote_host_id)
        model_ref = str(config.get("model") or "")
        project_id = run.project_id
        version_id = run.version_id
        device = run.device
        local_artifact_root = Path(run.artifact_path)
        try:
            model_path = Path(resolve_training_model(db, settings, project_id, model_ref))
        except ValueError as exc:
            fail_training_run(run_id, str(exc), bind=bind)
            return

        fingerprint, _, _ = compute_sync_fingerprint(
            project_id=project_id,
            version_id=version_id,
            model_ref=model_ref,
            version_root=Path(version.artifact_path),
            model_path=model_path,
        )
        sync_job = find_matching_completed_sync(
            db,
            project_id=project_id,
            version_id=version_id,
            model_ref=model_ref,
            fingerprint=fingerprint,
            host=host,
        )
        if sync_job is None:
            fail_training_run(
                run_id,
                "请先将当前版本与起始模型同步到 GPU05，完成后再启动远程训练",
                bind=bind,
            )
            return

    append_run_log(run_id, "remote training preparing", bind=bind)

    remote_version = remote_version_root(settings, project_id, version_id)
    remote_weight = remote_weight_path(settings, project_id, fingerprint)
    remote_run = remote_run_root(settings, project_id, run_id)
    remote_scripts = remote_scripts_root(settings)
    remote_worker = remote_scripts / WORKER_FILENAME
    remote_config = remote_run / "remote_train_config.json"
    remote_log = remote_run / "logs.txt"
    remote_status = remote_run / "remote_status.json"
    remote_pid = remote_run / "remote.pid"

    try:
        ensure_remote_dirs(settings, remote_run, remote_scripts)
        rsync_path(
            settings,
            LOCAL_WORKER,
            str(remote_worker),
            is_directory=False,
        )

        remote_payload = {
            "artifact_root": str(remote_run),
            "log_path": str(remote_log),
            "version_root": str(remote_version),
            "model_path": str(remote_weight),
            "model_ref": model_ref,
            "epochs": config["epochs"],
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "device": device,
            "augmentation_kwargs": ultralytics_augmentation_kwargs(config),
        }
        local_config = local_artifact_root / "remote_train_config.json"
        local_config.write_text(
            json.dumps(remote_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rsync_path(settings, local_config, str(remote_config), is_directory=False)

        launch = (
            f"cd {quote_remote(remote_run)} && "
            f"nohup {quote_remote(settings.remote_python)} {quote_remote(remote_worker)} "
            f"--config {quote_remote(remote_config)} "
            f"> {quote_remote(remote_log)} 2>&1 & echo $! > {quote_remote(remote_pid)}"
        )
        run_ssh(settings, launch, check=True, timeout=60)
        mark_training_run_running(run_id, bind=bind)
        append_run_log(run_id, "remote ultralytics training started", bind=bind)

        while True:
            with session_factory() as db:
                run = db.get(TrainingRun, run_id)
                if run is None:
                    return
                if run.status == "cancelled":
                    cancel_remote_training_run(run_id, settings=settings, bind=bind)
                    append_run_log(run_id, "remote training cancelled", bind=bind)
                    return

            try:
                _pull_remote_run_artifacts(
                    settings,
                    project_id=project_id,
                    run_id=run_id,
                    local_artifact_root=local_artifact_root,
                )
            except Exception as exc:
                append_run_log(run_id, f"remote artifact pull warning: {exc}", bind=bind)

            status = _read_remote_status(settings, remote_status)
            if status and status.get("status") in {"completed", "failed"}:
                break
            time.sleep(5)

        _pull_remote_run_artifacts(
            settings,
            project_id=project_id,
            run_id=run_id,
            local_artifact_root=local_artifact_root,
        )
        status = status or _read_remote_status(settings, remote_status) or {}
        if status.get("status") == "failed":
            fail_training_run(
                run_id,
                str(status.get("error") or "remote training failed"),
                bind=bind,
            )
            return

        _ingest_local_metrics(run_id, local_artifact_root, bind=bind)
        complete_training_run(run_id, bind=bind)
        run_post_training_threshold_scan(run_id, bind=bind, settings=settings)
    except Exception as exc:
        fail_training_run(run_id, str(exc), bind=bind)
        try:
            _pull_remote_run_artifacts(
                settings,
                project_id=project_id,
                run_id=run_id,
                local_artifact_root=local_artifact_root,
            )
        except Exception:
            pass
