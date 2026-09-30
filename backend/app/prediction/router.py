from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.deps import get_current_user, require_admin
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.models import Annotation, ClassDef, Image, Prediction, PredictionJob, TrainingRun
from app.db.session import get_db
from app.prediction.runner import (
    create_prediction_job,
    execute_prediction_job,
    predict_images,
    read_prediction_job_filters,
    read_prediction_logs,
)
from app.prediction.schemas import (
    PredictionJobCreate,
    PredictionImageReview,
    PredictionJobList,
    PredictionJobLogs,
    PredictionJobRead,
    PredictionThresholdScanCreate,
    PredictionThresholdScanRead,
    PredictionReviewAnnotation,
    PredictionReviewImage,
    PredictionList,
    PredictionRead,
)
from app.storage.visibility import StorageEntityNotFoundError, require_active_entity


router = APIRouter(
    prefix="/api",
    tags=["prediction"],
    dependencies=[Depends(get_current_user)],
)
FAILURE_TYPE_PATTERN = "^(all|matched|false_positive|false_negative|class_confusion)$"


def _class_confusion_count(db: Session, job_id: int) -> int:
    return (
        db.scalar(
            select(func.count(Prediction.id)).where(
                Prediction.job_id == job_id,
                Prediction.failure_type == "class_confusion",
            )
        )
        or 0
    )


def _read_job(job: PredictionJob, db: Session) -> PredictionJobRead:
    return PredictionJobRead(
        id=job.id,
        run_id=job.run_id,
        project_id=job.project_id,
        version_id=job.version_id,
        status=job.status,
        image_scope=job.image_scope,
        confidence_threshold=job.confidence_threshold,
        image_filters=read_prediction_job_filters(job),
        artifact_path=job.artifact_path,
        log_path=job.log_path,
        image_count=job.image_count,
        prediction_count=job.prediction_count,
        matched_count=job.matched_count,
        false_positive_count=job.false_positive_count,
        false_negative_count=job.false_negative_count,
        class_confusion_count=_class_confusion_count(db, job.id),
        error_message=job.error_message,
        started_at=job.started_at,
        ended_at=job.ended_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _read_prediction(prediction: Prediction) -> PredictionRead:
    return PredictionRead(
        id=prediction.id,
        run_id=prediction.run_id,
        job_id=prediction.job_id,
        image_id=prediction.image_id,
        class_id=prediction.class_id,
        x_center=prediction.x_center,
        y_center=prediction.y_center,
        width=prediction.width,
        height=prediction.height,
        confidence=prediction.confidence,
        matched_annotation_id=prediction.matched_annotation_id,
        failure_type=prediction.failure_type,
    )


def _require_active_run(db: Session, run_id: int) -> TrainingRun:
    try:
        require_active_entity(db, "training_run", run_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run was not found")
    return run


def _get_job_for_active_run(db: Session, job_id: int) -> PredictionJob:
    job = db.get(PredictionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Prediction job was not found")
    _require_active_run(db, job.run_id)
    return job


@router.post("/training/runs/{run_id}/prediction-jobs", response_model=PredictionJobRead)
def create_run_prediction_job(
    run_id: int,
    request: PredictionJobCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> PredictionJobRead:
    run = _require_active_run(db, run_id)

    try:
        job = create_prediction_job(
            db,
            settings,
            run,
            image_scope=request.image_scope,
            confidence_threshold=request.confidence_threshold,
            version_id=request.version_id,
            image_filters=request.image_filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = execute_prediction_job(db, job, run, predictor=predict_images)
    return _read_job(job, db)


@router.post(
    "/training/runs/{run_id}/prediction-threshold-scan",
    response_model=PredictionThresholdScanRead,
)
def create_run_prediction_threshold_scan(
    run_id: int,
    request: PredictionThresholdScanCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> PredictionThresholdScanRead:
    run = _require_active_run(db, run_id)

    jobs: list[PredictionJobRead] = []
    for threshold in request.thresholds:
        try:
            job = create_prediction_job(
                db,
                settings,
                run,
                image_scope=request.image_scope,
                confidence_threshold=threshold,
                version_id=request.version_id,
                image_filters=request.image_filters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job = execute_prediction_job(db, job, run, predictor=predict_images)
        jobs.append(_read_job(job, db))
    return PredictionThresholdScanRead(items=jobs)


@router.get("/training/runs/{run_id}/prediction-jobs", response_model=PredictionJobList)
def list_run_prediction_jobs(run_id: int, db: Session = Depends(get_db)) -> PredictionJobList:
    _require_active_run(db, run_id)
    jobs = db.scalars(
        select(PredictionJob).where(PredictionJob.run_id == run_id).order_by(PredictionJob.id.desc())
    ).all()
    return PredictionJobList(items=[_read_job(job, db) for job in jobs])


@router.get("/prediction-jobs/{job_id}/predictions", response_model=PredictionList)
def list_job_predictions(
    job_id: int,
    failure_type: str = Query("all", pattern=FAILURE_TYPE_PATTERN),
    class_id: int | None = None,
    confidence_min: float | None = Query(None, ge=0, le=1),
    confidence_max: float | None = Query(None, ge=0, le=1),
    platform: str | None = Query(None, max_length=80),
    altitude_min: float | None = None,
    altitude_max: float | None = None,
    timestamp_min: float | None = None,
    timestamp_max: float | None = None,
    db: Session = Depends(get_db),
) -> PredictionList:
    _get_job_for_active_run(db, job_id)
    query = select(Prediction).where(Prediction.job_id == job_id)
    needs_image_join = any(
        value is not None
        for value in [
            platform,
            altitude_min,
            altitude_max,
            timestamp_min,
            timestamp_max,
        ]
    )
    if needs_image_join:
        query = query.join(Image, Prediction.image_id == Image.id)
    if failure_type != "all":
        query = query.where(Prediction.failure_type == failure_type)
    if class_id is not None:
        query = query.where(Prediction.class_id == class_id)
    if confidence_min is not None:
        query = query.where(Prediction.confidence >= confidence_min)
    if confidence_max is not None:
        query = query.where(Prediction.confidence <= confidence_max)
    if platform:
        query = query.where(Image.platform == platform)
    if altitude_min is not None:
        query = query.where(Image.altitude >= altitude_min)
    if altitude_max is not None:
        query = query.where(Image.altitude <= altitude_max)
    if timestamp_min is not None:
        query = query.where(Image.timestamp >= timestamp_min)
    if timestamp_max is not None:
        query = query.where(Image.timestamp <= timestamp_max)

    predictions = db.scalars(query.order_by(Prediction.id)).all()
    return PredictionList(items=[_read_prediction(prediction) for prediction in predictions])


@router.get("/prediction-jobs/{job_id}/logs", response_model=PredictionJobLogs)
def get_prediction_job_logs(job_id: int, db: Session = Depends(get_db)) -> PredictionJobLogs:
    job = _get_job_for_active_run(db, job_id)
    return PredictionJobLogs(job_id=job.id, text=read_prediction_logs(job))


@router.get("/prediction-jobs/{job_id}/images/{image_id}/review", response_model=PredictionImageReview)
def get_prediction_image_review(
    job_id: int,
    image_id: int,
    db: Session = Depends(get_db),
) -> PredictionImageReview:
    _get_job_for_active_run(db, job_id)
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image was not found")

    annotation_rows = db.execute(
        select(Annotation, ClassDef)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.id)
    ).all()
    predictions = db.scalars(
        select(Prediction)
        .where(Prediction.job_id == job_id, Prediction.image_id == image_id)
        .order_by(Prediction.id)
    ).all()
    counts = {
        "matched": sum(1 for prediction in predictions if prediction.failure_type == "matched"),
        "false_positive": sum(
            1 for prediction in predictions if prediction.failure_type == "false_positive"
        ),
        "false_negative": sum(
            1 for prediction in predictions if prediction.failure_type == "false_negative"
        ),
        "class_confusion": sum(
            1 for prediction in predictions if prediction.failure_type == "class_confusion"
        ),
    }

    return PredictionImageReview(
        image=PredictionReviewImage(
            id=image.id,
            relative_path=image.relative_path,
            image_url=f"/api/images/{image.id}/file",
            platform=image.platform,
            altitude=image.altitude,
            timestamp=image.timestamp,
        ),
        annotations=[
            PredictionReviewAnnotation(
                id=annotation.id,
                image_id=annotation.image_id,
                class_id=annotation.class_id,
                class_name=class_def.name,
                class_color=class_def.color,
                x_center=annotation.x_center,
                y_center=annotation.y_center,
                width=annotation.width,
                height=annotation.height,
                track_id=annotation.track_id,
                edge_tags=annotation.edge_tags or [],
            )
            for annotation, class_def in annotation_rows
        ],
        predictions=[_read_prediction(prediction) for prediction in predictions],
        counts=counts,
    )
