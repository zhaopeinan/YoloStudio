from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import DatasetVersion, RemoteSyncJob
from app.db.session import SessionLocal
from app.remote.fingerprint import compute_sync_fingerprint, write_manifest
from app.remote.paths import (
    remote_manifest_path,
    remote_version_root,
    remote_weight_path,
)
from app.remote.rsync import RsyncProgress, cancel_active_rsync, rsync_path
from app.remote.ssh import ensure_remote_dirs, quote_remote, run_ssh
from app.training.weights import resolve_training_model


ACTIVE_SYNC_STATUSES = {"queued", "running"}


def _now() -> datetime:
    return datetime.now(UTC)


def _session_factory(bind=None):
    return sessionmaker(
        bind=bind or SessionLocal.kw["bind"],
        autoflush=False,
        autocommit=False,
        future=True,
    )


def find_matching_completed_sync(
    db: Session,
    *,
    project_id: int,
    version_id: int,
    model_ref: str,
    fingerprint: str,
    host: str,
) -> RemoteSyncJob | None:
    return db.scalar(
        select(RemoteSyncJob)
        .where(
            RemoteSyncJob.project_id == project_id,
            RemoteSyncJob.version_id == version_id,
            RemoteSyncJob.model_ref == model_ref,
            RemoteSyncJob.host == host,
            RemoteSyncJob.status == "completed",
            RemoteSyncJob.fingerprint == fingerprint,
            RemoteSyncJob.remote_fingerprint == fingerprint,
        )
        .order_by(RemoteSyncJob.id.desc())
    )


def create_sync_job(
    db: Session,
    settings: Settings,
    *,
    project_id: int,
    version_id: int,
    model_ref: str,
    host: str | None = None,
) -> RemoteSyncJob:
    host = host or settings.remote_host_id
    if not settings.remote_enabled:
        raise ValueError("远程同步已禁用")
    if host != settings.remote_host_id:
        raise ValueError(f"不支持的远程主机: {host}")

    version = db.get(DatasetVersion, version_id)
    if version is None or version.project_id != project_id:
        raise LookupError("数据集版本不存在")

    active = db.scalar(
        select(RemoteSyncJob)
        .where(
            RemoteSyncJob.project_id == project_id,
            RemoteSyncJob.host == host,
            RemoteSyncJob.status.in_(ACTIVE_SYNC_STATUSES),
        )
        .order_by(RemoteSyncJob.id.desc())
    )
    if active is not None:
        raise ValueError(f"已有同步任务 #{active.id} 正在进行")

    model_path = Path(resolve_training_model(db, settings, project_id, model_ref))
    version_root = Path(version.artifact_path)
    fingerprint, files_total, bytes_total = compute_sync_fingerprint(
        project_id=project_id,
        version_id=version_id,
        model_ref=model_ref,
        version_root=version_root,
        model_path=model_path,
    )

    existing = find_matching_completed_sync(
        db,
        project_id=project_id,
        version_id=version_id,
        model_ref=model_ref,
        fingerprint=fingerprint,
        host=host,
    )
    if existing is not None:
        # Return a fresh completed snapshot job so the UI can show "already synced".
        job = RemoteSyncJob(
            project_id=project_id,
            version_id=version_id,
            model_ref=model_ref,
            host=host,
            status="completed",
            bytes_total=bytes_total,
            bytes_done=bytes_total,
            files_total=files_total,
            files_done=files_total,
            speed_bps=0.0,
            fingerprint=fingerprint,
            remote_fingerprint=fingerprint,
            started_at=_now(),
            ended_at=_now(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    job = RemoteSyncJob(
        project_id=project_id,
        version_id=version_id,
        model_ref=model_ref,
        host=host,
        status="queued",
        bytes_total=bytes_total,
        bytes_done=0,
        files_total=files_total,
        files_done=0,
        speed_bps=0.0,
        fingerprint=fingerprint,
        remote_fingerprint=None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_sync_progress(
    job_id: int,
    *,
    bytes_done: int | None = None,
    files_done: int | None = None,
    speed_bps: float | None = None,
    status: str | None = None,
    error_message: str | None = None,
    remote_fingerprint: str | None = None,
    local_pid: int | None = None,
    bind=None,
) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        job = db.get(RemoteSyncJob, job_id)
        if job is None:
            return
        if bytes_done is not None:
            job.bytes_done = min(max(bytes_done, 0), max(job.bytes_total, bytes_done))
        if files_done is not None:
            job.files_done = files_done
        if speed_bps is not None:
            job.speed_bps = max(speed_bps, 0.0)
        if status is not None:
            job.status = status
            if status == "running" and job.started_at is None:
                job.started_at = _now()
            if status in {"completed", "failed", "cancelled"}:
                job.ended_at = _now()
        if error_message is not None:
            job.error_message = error_message
        if remote_fingerprint is not None:
            job.remote_fingerprint = remote_fingerprint
        if local_pid is not None:
            job.local_pid = local_pid
        db.commit()


def execute_sync_job(job_id: int, bind=None, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    session_factory = _session_factory(bind)
    with session_factory() as db:
        job = db.get(RemoteSyncJob, job_id)
        if job is None or job.status not in ACTIVE_SYNC_STATUSES:
            return
        version = db.get(DatasetVersion, job.version_id)
        if version is None:
            update_sync_progress(
                job_id,
                status="failed",
                error_message="数据集版本不存在",
                bind=bind,
            )
            return
        try:
            model_path = Path(
                resolve_training_model(db, settings, job.project_id, job.model_ref)
            )
        except ValueError as exc:
            update_sync_progress(job_id, status="failed", error_message=str(exc), bind=bind)
            return
        version_root = Path(version.artifact_path)
        fingerprint = job.fingerprint
        project_id = job.project_id
        version_id = job.version_id
        model_ref = job.model_ref
        bytes_total = job.bytes_total

    update_sync_progress(job_id, status="running", bind=bind)

    remote_version = remote_version_root(settings, project_id, version_id)
    remote_weight = remote_weight_path(settings, project_id, fingerprint)
    try:
        ensure_remote_dirs(
            settings,
            remote_version,
            remote_weight.parent,
            settings.remote_root / "scripts",
        )

        version_bytes = sum(
            path.stat().st_size for path in version_root.rglob("*") if path.is_file()
        )
        model_bytes = model_path.stat().st_size if model_path.is_file() else 0
        transferred_base = 0

        def on_version_progress(progress: RsyncProgress) -> None:
            done = min(progress.bytes_done, version_bytes)
            update_sync_progress(
                job_id,
                bytes_done=done,
                speed_bps=progress.speed_bps,
                bind=bind,
            )

        rsync_path(
            settings,
            version_root,
            str(remote_version),
            is_directory=True,
            job_id=job_id,
            on_progress=on_version_progress,
        )
        transferred_base = version_bytes
        update_sync_progress(
            job_id,
            bytes_done=transferred_base,
            files_done=max(1, (bytes_total and 1) or 1),
            bind=bind,
        )

        def on_model_progress(progress: RsyncProgress) -> None:
            done = transferred_base + min(progress.bytes_done, model_bytes)
            update_sync_progress(
                job_id,
                bytes_done=done,
                speed_bps=progress.speed_bps,
                bind=bind,
            )

        rsync_path(
            settings,
            model_path,
            str(remote_weight),
            is_directory=False,
            job_id=job_id,
            on_progress=on_model_progress,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_local = Path(tmp) / "remote_manifest.json"
            write_manifest(
                manifest_local,
                project_id=project_id,
                version_id=version_id,
                model_ref=model_ref,
                fingerprint=fingerprint,
                model_remote_path=str(remote_weight),
            )
            remote_manifest = remote_manifest_path(settings, project_id, version_id)
            rsync_path(
                settings,
                manifest_local,
                str(remote_manifest),
                is_directory=False,
                job_id=job_id,
            )

        # Rewrite remote data.yaml path to the remote version root.
        remote_data_yaml = remote_version / "data.yaml"
        rewrite = (
            f"{quote_remote(settings.remote_python)} - <<'PY'\n"
            "from pathlib import Path\n"
            f"path = Path({str(remote_data_yaml)!r})\n"
            f"root = Path({str(remote_version)!r})\n"
            "if path.is_file():\n"
            "    lines = path.read_text().splitlines()\n"
            "    rewritten = [f'path: {root}']\n"
            "    rewritten.extend(line for line in lines if not line.startswith('path:'))\n"
            "    path.write_text('\\n'.join(rewritten) + '\\n')\n"
            "PY"
        )
        run_ssh(settings, rewrite, check=True, timeout=60)

        update_sync_progress(
            job_id,
            status="completed",
            bytes_done=bytes_total or (version_bytes + model_bytes),
            files_done=max(1, version_count_hint(version_root) + (1 if model_bytes else 0)),
            remote_fingerprint=fingerprint,
            bind=bind,
        )
    except Exception as exc:
        # If cancelled mid-flight, keep cancelled status.
        session_factory = _session_factory(bind)
        with session_factory() as db:
            job = db.get(RemoteSyncJob, job_id)
            if job is not None and job.status == "cancelled":
                return
        update_sync_progress(
            job_id,
            status="failed",
            error_message=str(exc),
            bind=bind,
        )


def version_count_hint(version_root: Path) -> int:
    return sum(1 for path in version_root.rglob("*") if path.is_file())


def cancel_sync_job(job_id: int, bind=None) -> bool:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        job = db.get(RemoteSyncJob, job_id)
        if job is None or job.status not in ACTIVE_SYNC_STATUSES:
            return False
        job.status = "cancelled"
        job.ended_at = _now()
        db.commit()
    cancel_active_rsync(job_id)
    return True


def require_completed_sync_for_training(
    db: Session,
    settings: Settings,
    *,
    project_id: int,
    version_id: int,
    model_ref: str,
    host: str,
) -> RemoteSyncJob:
    version = db.get(DatasetVersion, version_id)
    if version is None or version.project_id != project_id:
        raise ValueError("数据集版本不存在")
    model_path = Path(resolve_training_model(db, settings, project_id, model_ref))
    fingerprint, _, _ = compute_sync_fingerprint(
        project_id=project_id,
        version_id=version_id,
        model_ref=model_ref,
        version_root=Path(version.artifact_path),
        model_path=model_path,
    )
    job = find_matching_completed_sync(
        db,
        project_id=project_id,
        version_id=version_id,
        model_ref=model_ref,
        fingerprint=fingerprint,
        host=host,
    )
    if job is None:
        raise ValueError("请先将当前版本与起始模型同步到 GPU05，完成后再启动远程训练")
    return job


def sync_job_to_dict(job: RemoteSyncJob) -> dict:
    percent = None
    if job.bytes_total > 0:
        percent = round(100.0 * job.bytes_done / job.bytes_total, 2)
    elif job.status == "completed":
        percent = 100.0
    return {
        "id": job.id,
        "project_id": job.project_id,
        "version_id": job.version_id,
        "model_ref": job.model_ref,
        "host": job.host,
        "status": job.status,
        "bytes_total": job.bytes_total,
        "bytes_done": job.bytes_done,
        "files_total": job.files_total,
        "files_done": job.files_done,
        "speed_bps": job.speed_bps,
        "percent": percent,
        "fingerprint": job.fingerprint,
        "remote_fingerprint": job.remote_fingerprint,
        "error_message": job.error_message,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
