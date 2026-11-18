from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user, require_admin
from app.core.settings import Settings, get_settings
from app.db.models import Project, RemoteSyncJob
from app.db.session import get_db
from app.remote.probe import probe_remote_host
from app.remote.schemas import (
    RemoteHostList,
    RemoteHostStatus,
    RemoteSyncCreate,
    RemoteSyncJobList,
    RemoteSyncJobRead,
)
from app.remote.sync import (
    cancel_sync_job,
    create_sync_job,
    execute_sync_job,
    sync_job_to_dict,
)
from app.storage.visibility import require_active_entity, StorageEntityNotFoundError


router = APIRouter(
    prefix="/api",
    tags=["remote"],
    dependencies=[Depends(get_current_user)],
)


def _read_job(job: RemoteSyncJob) -> RemoteSyncJobRead:
    return RemoteSyncJobRead.model_validate(sync_job_to_dict(job))


@router.get("/remote-hosts", response_model=RemoteHostList)
def list_remote_hosts(
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> RemoteHostList:
    remote = RemoteHostStatus.model_validate(probe_remote_host(settings))
    return RemoteHostList(items=[remote])


@router.post("/projects/{project_id}/remote-sync", response_model=RemoteSyncJobRead)
def start_remote_sync(
    project_id: int,
    request: RemoteSyncCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> RemoteSyncJobRead:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")
    try:
        require_active_entity(db, "dataset_version", request.version_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        job = create_sync_job(
            db,
            settings,
            project_id=project_id,
            version_id=request.version_id,
            model_ref=request.model_ref,
            host=request.host,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if job.status == "queued":
        background_tasks.add_task(execute_sync_job, job.id, db.get_bind(), settings)
    return _read_job(job)


@router.get("/remote-sync/{job_id}", response_model=RemoteSyncJobRead)
def get_remote_sync_job(
    job_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> RemoteSyncJobRead:
    job = db.get(RemoteSyncJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="同步任务不存在")
    return _read_job(job)


@router.get("/projects/{project_id}/remote-sync", response_model=RemoteSyncJobList)
def list_project_remote_sync_jobs(
    project_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> RemoteSyncJobList:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")
    jobs = db.scalars(
        select(RemoteSyncJob)
        .where(RemoteSyncJob.project_id == project_id)
        .order_by(RemoteSyncJob.id.desc())
        .limit(50)
    ).all()
    return RemoteSyncJobList(items=[_read_job(job) for job in jobs])


@router.post("/remote-sync/{job_id}/cancel", response_model=RemoteSyncJobRead)
def cancel_remote_sync_job(
    job_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> RemoteSyncJobRead:
    job = db.get(RemoteSyncJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="同步任务不存在")
    if not cancel_sync_job(job_id, bind=db.get_bind()):
        raise HTTPException(status_code=400, detail="同步任务未在进行中")
    db.refresh(job)
    return _read_job(job)
