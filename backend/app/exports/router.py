from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.auth.deps import get_current_user, require_admin
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ExportArtifact, TrainingRun
from app.db.session import get_db
from app.exports.runner import (
    best_weights_path,
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
from app.storage.visibility import StorageEntityNotFoundError, require_active_entity


router = APIRouter(
    prefix="/api/training/runs",
    tags=["exports"],
    dependencies=[Depends(get_current_user)],
)


def _export_download_url(artifact: ExportArtifact) -> str | None:
    if artifact.status != "completed" or not artifact.artifact_path:
        return None
    path = Path(artifact.artifact_path)
    if not path.is_file():
        return None
    return f"/api/training/runs/{artifact.run_id}/exports/{artifact.id}/file"


def _read_export(artifact: ExportArtifact) -> ExportArtifactRead:
    return ExportArtifactRead(
        id=artifact.id,
        run_id=artifact.run_id,
        project_id=artifact.project_id,
        format=artifact.format,
        status=artifact.status,
        artifact_path=artifact.artifact_path,
        download_url=_export_download_url(artifact),
        error_message=artifact.error_message,
        metadata=artifact.metadata_ or {},
        started_at=artifact.started_at,
        ended_at=artifact.ended_at,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
    )


def _get_run(db: Session, run_id: int) -> TrainingRun:
    try:
        require_active_entity(db, "training_run", run_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@router.get("/{run_id}/weights/best")
def download_run_best_weights(run_id: int, db: Session = Depends(get_db)) -> FileResponse:
    run = _get_run(db, run_id)
    weights = best_weights_path(run)
    if not weights.is_file():
        raise HTTPException(status_code=404, detail="Best weights were not found")
    return FileResponse(
        weights,
        media_type="application/octet-stream",
        filename=f"run-{run.id}-best.pt",
    )


@router.get("/{run_id}/exports", response_model=ExportArtifactList)
def list_run_exports(run_id: int, db: Session = Depends(get_db)) -> ExportArtifactList:
    _get_run(db, run_id)
    artifacts = db.scalars(
        select(ExportArtifact)
        .where(ExportArtifact.run_id == run_id)
        .order_by(ExportArtifact.id.desc())
    ).all()
    return ExportArtifactList(items=[_read_export(artifact) for artifact in artifacts])


@router.get("/{run_id}/exports/{export_id}/file")
def download_run_export(
    run_id: int,
    export_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    _get_run(db, run_id)
    artifact = db.get(ExportArtifact, export_id)
    if artifact is None or artifact.run_id != run_id:
        raise HTTPException(status_code=404, detail="Export artifact was not found")
    if artifact.status != "completed" or not artifact.artifact_path:
        raise HTTPException(status_code=404, detail="Export file is not available")

    path = Path(artifact.artifact_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export file was not found")

    suffix = path.suffix.lstrip(".") or artifact.format
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"run-{run_id}.{suffix}",
    )


@router.post("/{run_id}/exports", response_model=ExportArtifactRead)
def create_run_export(
    run_id: int,
    request: ExportCreate,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
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
