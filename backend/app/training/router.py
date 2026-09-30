from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile

from app.auth.deps import get_current_user, require_admin
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.models import DatasetVersion, ModelWeight, Project, TrainingRun
from app.db.session import get_db
from app.experiments.schemas import ProjectExperimentSummary, RunExperimentSummary
from app.experiments.summary import build_project_experiment_summary, build_run_experiment_summary
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    require_active_entity,
)
from app.training.live import build_live_snapshot, query_nvidia_smi
from app.training.runner import (
    cancel_training_run,
    create_queued_run,
    execute_training_run,
    latest_metrics_for_run,
    read_run_logs,
)
from app.training.schemas import (
    GpuStatus,
    ModelWeightList,
    ModelWeightRead,
    TrainingLiveSnapshot,
    TrainingModelList,
    TrainingModelOption,
    TrainingRunArtifact,
    TrainingRunArtifactSummary,
    TrainingRunCreate,
    TrainingRunList,
    TrainingRunLogs,
    TrainingRunRead,
)
from app.training.weights import (
    MAX_WEIGHT_BYTES,
    create_model_weight,
    delete_model_weight,
    list_model_weights,
    list_training_model_options,
)


router = APIRouter(
    prefix="/api",
    tags=["training"],
    dependencies=[Depends(get_current_user)],
)

MAX_ARTIFACT_ITEMS = 200
ARTIFACT_CATEGORY_ORDER = {
    "config": 0,
    "log": 1,
    "metrics": 2,
    "weights": 3,
    "plot": 4,
    "prediction": 5,
    "export": 6,
    "dataset": 7,
    "artifact": 8,
}


def _artifact_category(relative_path: str) -> str:
    path = relative_path.lower()
    name = Path(path).name
    suffix = Path(path).suffix

    if name == "config.json" or path.endswith("/config.json"):
        return "config"
    if name == "logs.txt" or suffix == ".log":
        return "log"
    if name == "metrics.jsonl" or "metrics" in path:
        return "metrics"
    if path.startswith("ultralytics/weights/") or path.startswith("weights/"):
        return "weights"
    if path.startswith("exports/") or suffix in {".onnx", ".engine"}:
        return "export"
    if path.startswith("predictions/"):
        return "prediction"
    if path.startswith("gridmask_dataset/"):
        return "dataset"
    if path.startswith("ultralytics/") and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "plot"
    return "artifact"


def _list_run_artifacts(run: TrainingRun) -> TrainingRunArtifactSummary:
    artifact_root = Path(run.artifact_path)
    resolved_root = artifact_root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        return TrainingRunArtifactSummary(
            run_id=run.id,
            artifact_root=run.artifact_path,
            total_count=0,
            items=[],
        )

    items: list[TrainingRunArtifact] = []
    for path in resolved_root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(resolved_root).as_posix()
        items.append(
            TrainingRunArtifact(
                relative_path=relative_path,
                category=_artifact_category(relative_path),
                size_bytes=path.stat().st_size,
            )
        )

    items.sort(
        key=lambda item: (
            ARTIFACT_CATEGORY_ORDER.get(item.category, ARTIFACT_CATEGORY_ORDER["artifact"]),
            item.relative_path,
        )
    )
    return TrainingRunArtifactSummary(
        run_id=run.id,
        artifact_root=run.artifact_path,
        total_count=len(items),
        items=items[:MAX_ARTIFACT_ITEMS],
    )


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


def _require_active(db: Session, entity_type: str, entity_id: int) -> None:
    try:
        require_active_entity(db, entity_type, entity_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _get_active_run(db: Session, run_id: int) -> TrainingRun:
    _require_active(db, "training_run", run_id)
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return run


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project was not found")
    return project


def _read_weight(weight: ModelWeight) -> ModelWeightRead:
    return ModelWeightRead(
        id=weight.id,
        project_id=weight.project_id,
        original_filename=weight.original_filename,
        size_bytes=weight.size_bytes,
        created_at=weight.created_at,
        updated_at=weight.updated_at,
    )


@router.get("/training/models", response_model=TrainingModelList)
def list_training_models(
    project_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> TrainingModelList:
    _require_project(db, project_id)
    return TrainingModelList(
        items=[TrainingModelOption(**item) for item in list_training_model_options(db, project_id)]
    )


@router.get("/training/weights", response_model=ModelWeightList)
def list_training_weights(
    project_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
) -> ModelWeightList:
    _require_project(db, project_id)
    return ModelWeightList(items=[_read_weight(item) for item in list_model_weights(db, project_id)])


@router.post("/training/weights", response_model=ModelWeightRead)
async def upload_training_weight(
    project_id: int = Form(..., ge=1),
    weight: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> ModelWeightRead:
    _require_project(db, project_id)
    content = await weight.read(MAX_WEIGHT_BYTES + 1)
    try:
        saved = create_model_weight(
            db,
            settings,
            project_id,
            weight.filename or "uploaded.pt",
            content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _read_weight(saved)


@router.delete("/training/weights/{weight_id}", status_code=204)
def remove_training_weight(
    weight_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> None:
    try:
        delete_model_weight(db, settings, weight_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/training/runs", response_model=TrainingRunRead)
def create_training_run(
    request: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> TrainingRunRead:
    _require_active(db, "dataset_version", request.version_id)
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
        select(TrainingRun)
        .where(
            TrainingRun.project_id == project_id,
            active_entity_predicate("training_run", TrainingRun.id),
        )
        .order_by(TrainingRun.id.desc())
    ).all()
    return TrainingRunList(items=[_read_run(db, run) for run in runs])


@router.get("/projects/{project_id}/training/summary", response_model=ProjectExperimentSummary)
def get_project_training_summary(
    project_id: int,
    db: Session = Depends(get_db),
) -> ProjectExperimentSummary:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project was not found")
    return build_project_experiment_summary(db, project_id)


@router.get("/training/runs/{run_id}", response_model=TrainingRunRead)
def get_training_run(run_id: int, db: Session = Depends(get_db)) -> TrainingRunRead:
    run = _get_active_run(db, run_id)
    return _read_run(db, run)


@router.get("/training/runs/{run_id}/artifacts", response_model=TrainingRunArtifactSummary)
def get_training_run_artifacts(
    run_id: int,
    db: Session = Depends(get_db),
) -> TrainingRunArtifactSummary:
    run = _get_active_run(db, run_id)
    return _list_run_artifacts(run)


@router.post("/training/runs/{run_id}/cancel", response_model=TrainingRunRead)
def cancel_run(
    run_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> TrainingRunRead:
    run = _get_active_run(db, run_id)
    if not cancel_training_run(run_id, bind=db.get_bind()):
        raise HTTPException(status_code=400, detail="Training run is not active")
    db.refresh(run)
    return _read_run(db, run)


@router.get("/training/runs/{run_id}/logs", response_model=TrainingRunLogs)
def get_training_run_logs(run_id: int, db: Session = Depends(get_db)) -> TrainingRunLogs:
    run = _get_active_run(db, run_id)
    return TrainingRunLogs(run_id=run.id, text=read_run_logs(run))


@router.get("/training/runs/{run_id}/live", response_model=TrainingLiveSnapshot)
def get_training_run_live(
    run_id: int,
    include_gpu: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> TrainingLiveSnapshot:
    """Lightweight live snapshot for the training monitor modal (1s polling)."""
    run = _get_active_run(db, run_id)
    return TrainingLiveSnapshot.model_validate(build_live_snapshot(run, include_gpu=include_gpu))


@router.get("/system/gpu", response_model=GpuStatus)
def get_system_gpu() -> GpuStatus:
    """Read-only nvidia-smi snapshot; failures never affect training."""
    return GpuStatus.model_validate(query_nvidia_smi())


@router.get("/training/runs/{run_id}/summary", response_model=RunExperimentSummary)
def get_training_run_summary(
    run_id: int,
    db: Session = Depends(get_db),
) -> RunExperimentSummary:
    run = _get_active_run(db, run_id)
    return build_run_experiment_summary(db, run)
