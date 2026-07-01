from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Annotation, ClassDef, Prediction, PredictionJob, RunMetric, TrainingRun
from app.experiments.schemas import (
    ClassOutcome,
    ConfusionCell,
    MetricPoint,
    MetricSeries,
    ProjectExperimentSummary,
    RunExperimentSummary,
    RunComparisonRow,
    ThresholdPoint,
    ThresholdRecommendation,
)


def build_run_experiment_summary(db: Session, run: TrainingRun) -> RunExperimentSummary:
    latest_job = _latest_completed_prediction_job(db, run.id)
    threshold_scan = _threshold_scan(db, run.id)
    return RunExperimentSummary(
        run_id=run.id,
        metric_series=_metric_series(db, run.id),
        class_outcomes=_class_outcomes(db, latest_job.id) if latest_job else [],
        confusion_matrix=_confusion_matrix(db, latest_job.id) if latest_job else [],
        threshold_scan=threshold_scan,
        threshold_recommendation=_threshold_recommendation(threshold_scan),
        latest_prediction_job_id=latest_job.id if latest_job else None,
    )


def build_project_experiment_summary(db: Session, project_id: int) -> ProjectExperimentSummary:
    runs = db.scalars(
        select(TrainingRun)
        .where(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.id.desc())
        .limit(50)
    ).all()
    return ProjectExperimentSummary(
        project_id=project_id,
        runs=[_comparison_row(db, run) for run in runs],
    )


def _comparison_row(db: Session, run: TrainingRun) -> RunComparisonRow:
    latest_job = _latest_completed_prediction_job(db, run.id)
    threshold_scan = _threshold_scan(db, run.id)
    recommendation = _threshold_recommendation(threshold_scan)
    return RunComparisonRow(
        run_id=run.id,
        status=run.status,
        model=str((run.config or {}).get("model") or "model"),
        epochs=_optional_int((run.config or {}).get("epochs")),
        device=run.device,
        artifact_path=run.artifact_path,
        map50=_latest_metric_value(db, run.id, "metrics/mAP50(B)"),
        box_loss=_latest_metric_value(db, run.id, "train/box_loss"),
        latest_prediction_job_id=latest_job.id if latest_job else None,
        matched=latest_job.matched_count if latest_job else 0,
        false_positive=latest_job.false_positive_count if latest_job else 0,
        false_negative=latest_job.false_negative_count if latest_job else 0,
        class_confusion=_prediction_count(db, latest_job.id, "class_confusion")
        if latest_job
        else 0,
        best_threshold=recommendation.confidence_threshold if recommendation else None,
        best_f1=recommendation.f1 if recommendation else None,
    )


def _optional_int(value) -> int | None:
    return value if isinstance(value, int) else None


def _latest_metric_value(db: Session, run_id: int, name: str) -> float | None:
    metric = db.scalar(
        select(RunMetric)
        .where(RunMetric.run_id == run_id, RunMetric.name == name)
        .order_by(RunMetric.id.desc())
    )
    return metric.value if metric else None


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
        class_confusion_count = _prediction_count(db, job.id, "class_confusion")
        precision_denominator = job.matched_count + job.false_positive_count + class_confusion_count
        recall_denominator = job.matched_count + job.false_negative_count + class_confusion_count
        precision = job.matched_count / precision_denominator if precision_denominator else 0
        recall = job.matched_count / recall_denominator if recall_denominator else 0
        f1 = _f1(precision, recall)
        points.append(
            ThresholdPoint(
                job_id=job.id,
                confidence_threshold=job.confidence_threshold,
                matched=job.matched_count,
                false_positive=job.false_positive_count,
                false_negative=job.false_negative_count,
                class_confusion=class_confusion_count,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return points


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator else 0


def _prediction_count(db: Session, job_id: int, failure_type: str) -> int:
    return (
        db.scalar(
            select(func.count(Prediction.id)).where(
                Prediction.job_id == job_id,
                Prediction.failure_type == failure_type,
            )
        )
        or 0
    )


def _threshold_recommendation(points: list[ThresholdPoint]) -> ThresholdRecommendation | None:
    if not points:
        return None
    best = max(points, key=lambda point: (point.f1, point.recall, point.confidence_threshold))
    return ThresholdRecommendation(
        job_id=best.job_id,
        confidence_threshold=best.confidence_threshold,
        precision=best.precision,
        recall=best.recall,
        f1=best.f1,
    )


def _class_outcomes(db: Session, job_id: int) -> list[ClassOutcome]:
    classes = db.scalars(select(ClassDef).order_by(ClassDef.id)).all()
    names = {class_def.id: class_def.name for class_def in classes}
    counts: dict[int, dict[str, int]] = {
        class_def.id: {
            "matched": 0,
            "false_positive": 0,
            "false_negative": 0,
            "class_confusion": 0,
        }
        for class_def in classes
    }
    predictions = db.scalars(
        select(Prediction).where(Prediction.job_id == job_id).order_by(Prediction.id)
    ).all()
    for prediction in predictions:
        bucket = counts.setdefault(
            prediction.class_id,
            {"matched": 0, "false_positive": 0, "false_negative": 0, "class_confusion": 0},
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
            class_confusion=values["class_confusion"],
        )
        for class_id, values in sorted(counts.items())
        if values["matched"]
        or values["false_positive"]
        or values["false_negative"]
        or values["class_confusion"]
    ]


def _confusion_matrix(db: Session, job_id: int) -> list[ConfusionCell]:
    rows = db.execute(
        select(Prediction, Annotation, ClassDef)
        .join(Annotation, Prediction.matched_annotation_id == Annotation.id)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Prediction.job_id == job_id, Prediction.failure_type.in_(["matched", "class_confusion"]))
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
