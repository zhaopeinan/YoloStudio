from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Annotation, ClassDef, Prediction, PredictionJob, RunMetric, TrainingRun
from app.experiments.schemas import (
    ClassOutcome,
    ConfusionCell,
    MetricPoint,
    MetricSeries,
    RunExperimentSummary,
    ThresholdPoint,
)


def build_run_experiment_summary(db: Session, run: TrainingRun) -> RunExperimentSummary:
    latest_job = _latest_completed_prediction_job(db, run.id)
    return RunExperimentSummary(
        run_id=run.id,
        metric_series=_metric_series(db, run.id),
        class_outcomes=_class_outcomes(db, latest_job.id) if latest_job else [],
        confusion_matrix=_confusion_matrix(db, latest_job.id) if latest_job else [],
        threshold_scan=_threshold_scan(db, run.id),
        latest_prediction_job_id=latest_job.id if latest_job else None,
    )


def _metric_series(db: Session, run_id: int) -> list[MetricSeries]:
    rows = db.scalars(
        select(RunMetric).where(RunMetric.run_id == run_id).order_by(RunMetric.name, RunMetric.id)
    ).all()
    grouped: dict[str, list[RunMetric]] = defaultdict(list)
    for metric in rows:
        grouped[metric.name].append(metric)

    return [
        MetricSeries(
            name=name,
            points=[
                MetricPoint(epoch=metric.epoch, step=metric.step, value=metric.value)
                for metric in metrics
            ],
            latest=metrics[-1].value if metrics else None,
        )
        for name, metrics in sorted(grouped.items())
    ]


def _latest_completed_prediction_job(db: Session, run_id: int) -> PredictionJob | None:
    return db.scalar(
        select(PredictionJob)
        .where(PredictionJob.run_id == run_id, PredictionJob.status == "completed")
        .order_by(PredictionJob.id.desc())
    )


def _threshold_scan(db: Session, run_id: int) -> list[ThresholdPoint]:
    jobs = db.scalars(
        select(PredictionJob)
        .where(PredictionJob.run_id == run_id, PredictionJob.status == "completed")
        .order_by(PredictionJob.confidence_threshold, PredictionJob.id)
    ).all()
    points: list[ThresholdPoint] = []
    for job in jobs:
        precision_denominator = job.matched_count + job.false_positive_count
        recall_denominator = job.matched_count + job.false_negative_count
        precision = job.matched_count / precision_denominator if precision_denominator else 0
        recall = job.matched_count / recall_denominator if recall_denominator else 0
        points.append(
            ThresholdPoint(
                job_id=job.id,
                confidence_threshold=job.confidence_threshold,
                matched=job.matched_count,
                false_positive=job.false_positive_count,
                false_negative=job.false_negative_count,
                precision=precision,
                recall=recall,
            )
        )
    return points


def _class_outcomes(db: Session, job_id: int) -> list[ClassOutcome]:
    classes = db.scalars(select(ClassDef).order_by(ClassDef.id)).all()
    names = {class_def.id: class_def.name for class_def in classes}
    counts: dict[int, dict[str, int]] = {
        class_def.id: {"matched": 0, "false_positive": 0, "false_negative": 0} for class_def in classes
    }
    predictions = db.scalars(
        select(Prediction).where(Prediction.job_id == job_id).order_by(Prediction.id)
    ).all()
    for prediction in predictions:
        bucket = counts.setdefault(
            prediction.class_id,
            {"matched": 0, "false_positive": 0, "false_negative": 0},
        )
        if prediction.failure_type in bucket:
            bucket[prediction.failure_type] += 1

    return [
        ClassOutcome(
            class_id=class_id,
            class_name=names.get(class_id, f"Class {class_id}"),
            matched=values["matched"],
            false_positive=values["false_positive"],
            false_negative=values["false_negative"],
        )
        for class_id, values in sorted(counts.items())
        if values["matched"] or values["false_positive"] or values["false_negative"]
    ]


def _confusion_matrix(db: Session, job_id: int) -> list[ConfusionCell]:
    rows = db.execute(
        select(Prediction, Annotation, ClassDef)
        .join(Annotation, Prediction.matched_annotation_id == Annotation.id)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Prediction.job_id == job_id, Prediction.failure_type == "matched")
        .order_by(Prediction.id)
    ).all()
    class_defs = db.scalars(select(ClassDef).order_by(ClassDef.id)).all()
    names = {class_def.id: class_def.name for class_def in class_defs}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    actual_names: dict[int, str] = {}
    for prediction, annotation, actual_class in rows:
        counts[(annotation.class_id, prediction.class_id)] += 1
        actual_names[annotation.class_id] = actual_class.name

    return [
        ConfusionCell(
            actual_class_id=actual_id,
            actual_class_name=actual_names.get(actual_id, names.get(actual_id, f"Class {actual_id}")),
            predicted_class_id=predicted_id,
            predicted_class_name=names.get(predicted_id, f"Class {predicted_id}"),
            count=count,
        )
        for (actual_id, predicted_id), count in sorted(counts.items())
    ]
