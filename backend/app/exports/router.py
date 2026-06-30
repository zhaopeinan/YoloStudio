from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ExportArtifact, TrainingRun
from app.db.session import get_db
from app.exports.runner import (
    create_export_artifact,
    execute_export,
    export_capabilities,
)
from app.exports.schemas import (
    ExportArtifactList,
    ExportArtifactRead,
    ExportCapabilities,
    ExportCreate,
)


router = APIRouter(prefix="/api/training/runs", tags=["exports"])


def _read_export(artifact: ExportArtifact) -> ExportArtifactRead:
    return ExportArtifactRead(
        id=artifact.id,
        run_id=artifact.run_id,
        project_id=artifact.project_id,
        format=artifact.format,
        status=artifact.status,
        artifact_path=artifact.artifact_path,
        error_message=artifact.error_message,
        metadata=artifact.metadata_ or {},
        started_at=artifact.started_at,
        ended_at=artifact.ended_at,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


def _get_run(db: Session, run_id: int) -> TrainingRun:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return run


@router.get("/{run_id}/exports/capabilities", response_model=ExportCapabilities)
def get_run_export_capabilities(
    run_id: int,
    db: Session = Depends(get_db),
) -> ExportCapabilities:
    run = _get_run(db, run_id)
    return ExportCapabilities(**export_capabilities(run))


@router.get("/{run_id}/exports", response_model=ExportArtifactList)
def list_run_exports(run_id: int, db: Session = Depends(get_db)) -> ExportArtifactList:
    _get_run(db, run_id)
    artifacts = db.scalars(
        select(ExportArtifact)
        .where(ExportArtifact.run_id == run_id)
        .order_by(ExportArtifact.id.desc())
    ).all()
    return ExportArtifactList(items=[_read_export(artifact) for artifact in artifacts])


@router.post("/{run_id}/exports", response_model=ExportArtifactRead)
def create_run_export(
    run_id: int,
    request: ExportCreate,
    db: Session = Depends(get_db),
) -> ExportArtifactRead:
    run = _get_run(db, run_id)
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Only completed runs can be exported")

    try:
        artifact = create_export_artifact(db, run, request.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    artifact = execute_export(db, artifact, run)
    return _read_export(artifact)
