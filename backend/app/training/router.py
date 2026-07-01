from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.models import DatasetVersion, Project, TrainingRun
from app.db.session import get_db
from app.experiments.schemas import RunExperimentSummary
from app.experiments.summary import build_run_experiment_summary
from app.training.runner import (
    cancel_training_run,
    create_queued_run,
    execute_training_run,
    latest_metrics_for_run,
    read_run_logs,
)
from app.training.schemas import TrainingRunCreate, TrainingRunList, TrainingRunLogs, TrainingRunRead


router = APIRouter(prefix="/api", tags=["training"])


def _read_run(db: Session, run: TrainingRun) -> TrainingRunRead:
    return TrainingRunRead(
        id=run.id,
        project_id=run.project_id,
        version_id=run.version_id,
        status=run.status,
        device=run.device,
        config=run.config,
        artifact_path=run.artifact_path,
        log_path=run.log_path,
        error_message=run.error_message,
        latest_metrics=latest_metrics_for_run(db, run.id),
        started_at=run.started_at,
        ended_at=run.ended_at,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post("/training/runs", response_model=TrainingRunRead)
def create_training_run(
    request: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TrainingRunRead:
    version = db.get(DatasetVersion, request.version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Dataset version was not found")

    try:
        run = create_queued_run(db, settings, version, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(execute_training_run, run.id, db.get_bind(), settings)
    return _read_run(db, run)


@router.get("/projects/{project_id}/training/runs", response_model=TrainingRunList)
def list_project_training_runs(project_id: int, db: Session = Depends(get_db)) -> TrainingRunList:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")
    runs = db.scalars(
        select(TrainingRun).where(TrainingRun.project_id == project_id).order_by(TrainingRun.id.desc())
    ).all()
    return TrainingRunList(items=[_read_run(db, run) for run in runs])


@router.get("/training/runs/{run_id}", response_model=TrainingRunRead)
def get_training_run(run_id: int, db: Session = Depends(get_db)) -> TrainingRunRead:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return _read_run(db, run)


@router.post("/training/runs/{run_id}/cancel", response_model=TrainingRunRead)
def cancel_run(run_id: int, db: Session = Depends(get_db)) -> TrainingRunRead:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    if not cancel_training_run(run_id, bind=db.get_bind()):
        raise HTTPException(status_code=400, detail="Training run is not active")
    db.refresh(run)
    return _read_run(db, run)


@router.get("/training/runs/{run_id}/logs", response_model=TrainingRunLogs)
def get_training_run_logs(run_id: int, db: Session = Depends(get_db)) -> TrainingRunLogs:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return TrainingRunLogs(run_id=run.id, text=read_run_logs(run))


@router.get("/training/runs/{run_id}/summary", response_model=RunExperimentSummary)
def get_training_run_summary(
    run_id: int,
    db: Session = Depends(get_db),
) -> RunExperimentSummary:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return build_run_experiment_summary(db, run)
