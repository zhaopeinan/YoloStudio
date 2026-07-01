from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.db.models import (
    Annotation,
    DatasetVersion,
    Image,
    Prediction,
    PredictionJob,
    TrainingRun,
)
from app.prediction.matching import Box, annotation_box, iou


def _now() -> datetime:
    return datetime.now(UTC)


def _log_line(message: str) -> str:
    return f"{_now().isoformat()} {message}\n"


def append_prediction_log(job: PredictionJob, message: str) -> None:
    path = Path(job.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(_log_line(message))


def prediction_artifact_root(settings: Settings, run: TrainingRun, job_id: int) -> Path:
    return Path(run.artifact_path) / "predictions" / str(job_id)


def create_prediction_job(
    db: Session,
    settings: Settings,
    run: TrainingRun,
    image_scope: str,
    confidence_threshold: float,
) -> PredictionJob:
    job = PredictionJob(
        run_id=run.id,
        project_id=run.project_id,
        status="queued",
        image_scope=image_scope,
        confidence_threshold=confidence_threshold,
        artifact_path="",
        log_path="",
    )
    db.add(job)
    db.flush()
    artifact_root = prediction_artifact_root(settings, run, job.id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    job.artifact_path = str(artifact_root)
    job.log_path = str(artifact_root / "logs.txt")
    Path(job.log_path).write_text(_log_line("prediction queued"))
    db.commit()
    db.refresh(job)
    return job


def _manifest_image_ids(version: DatasetVersion, image_scope: str) -> list[int]:
    images = version.split_manifest.get("images", [])
    ids = []
    for item in images:
        if image_scope != "all" and item.get("split") != image_scope:
            continue
        image_id = item.get("image_id")
        if isinstance(image_id, int):
            ids.append(image_id)
    return ids


def list_job_images(db: Session, run: TrainingRun, image_scope: str) -> list[Image]:
    version = db.get(DatasetVersion, run.version_id)
    if version is None:
        return []
    image_ids = _manifest_image_ids(version, image_scope)
    if not image_ids:
        return []
    return db.scalars(select(Image).where(Image.id.in_(image_ids)).order_by(Image.id)).all()


def run_uses_tta(run: TrainingRun) -> bool:
    return bool((run.config or {}).get("tta"))


def predict_images(
    run: TrainingRun,
    images: list[Image],
    confidence_threshold: float,
) -> dict[int, list[dict]]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not available: {exc.__class__.__name__}") from exc

    weights = Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"Model weights were not found: {weights}")

    model = YOLO(str(weights))
    use_tta = run_uses_tta(run)
    results_by_image: dict[int, list[dict]] = {}
    for image in images:
        result = model.predict(
            source=image.relative_path,
            conf=confidence_threshold,
            augment=use_tta,
            verbose=False,
        )
        results_by_image[image.id] = []
        for item in result:
            boxes = getattr(item, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xywhn = box.xywhn[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                results_by_image[image.id].append(
                    {
                        "class_id": class_id,
                        "x_center": float(xywhn[0]),
                        "y_center": float(xywhn[1]),
                        "width": float(xywhn[2]),
                        "height": float(xywhn[3]),
                        "confidence": confidence,
                    }
                )
    return results_by_image


def _prediction_box(row: dict) -> Box:
    return Box(
        x_center=float(row["x_center"]),
        y_center=float(row["y_center"]),
        width=float(row["width"]),
        height=float(row["height"]),
    )


def _merge_tags(existing_tags: list[str] | None, new_tags: list[str]) -> list[str]:
    merged = list(existing_tags or [])
    for tag in new_tags:
        if tag not in merged:
            merged.append(tag)
    return merged


def _class_id_for_prediction(raw_class_id: int, version: DatasetVersion | None) -> int:
    if version is None:
        return raw_class_id

    class_mapping = version.class_mapping or {}
    reverse_mapping = {int(yolo_index): int(class_id) for class_id, yolo_index in class_mapping.items()}
    if raw_class_id in reverse_mapping:
        return reverse_mapping[raw_class_id]
    return raw_class_id


def _best_annotation_match(
    prediction_box: Box,
    image_annotations: list[Annotation],
    matched_annotations: set[int],
    class_id: int,
    same_class_only: bool,
) -> tuple[Annotation | None, float]:
    best_annotation = None
    best_iou = 0.0
    for annotation in image_annotations:
        if annotation.id in matched_annotations:
            continue
        if same_class_only and annotation.class_id != class_id:
            continue
        if not same_class_only and annotation.class_id == class_id:
            continue
        score = iou(prediction_box, annotation_box(annotation))
        if score > best_iou:
            best_iou = score
            best_annotation = annotation
    return best_annotation, best_iou


def persist_predictions(
    db: Session,
    job: PredictionJob,
    run: TrainingRun,
    images: list[Image],
    predictions_by_image: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> None:
    image_ids = [image.id for image in images]
    annotations = db.scalars(
        select(Annotation).where(Annotation.image_id.in_(image_ids)).order_by(Annotation.id)
    ).all()
    annotations_by_image: dict[int, list[Annotation]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    version = db.get(DatasetVersion, run.version_id)
    rows: list[Prediction] = []
    matched_annotations: set[int] = set()
    for image in images:
        image_annotations = annotations_by_image.get(image.id, [])
        for prediction in predictions_by_image.get(image.id, []):
            raw_class_id = int(prediction["class_id"])
            class_id = _class_id_for_prediction(raw_class_id, version)
            prediction_box = _prediction_box(prediction)
            best_annotation, best_iou = _best_annotation_match(
                prediction_box,
                image_annotations,
                matched_annotations,
                class_id,
                same_class_only=True,
            )

            failure_type = "false_positive"
            matched_annotation_id = None
            if best_annotation is not None and best_iou >= iou_threshold:
                failure_type = "matched"
                matched_annotation_id = best_annotation.id
                matched_annotations.add(best_annotation.id)
            else:
                confused_annotation, confused_iou = _best_annotation_match(
                    prediction_box,
                    image_annotations,
                    matched_annotations,
                    class_id,
                    same_class_only=False,
                )
                if confused_annotation is not None and confused_iou >= iou_threshold:
                    failure_type = "class_confusion"
                    matched_annotation_id = confused_annotation.id
                    matched_annotations.add(confused_annotation.id)
                    confused_annotation.edge_tags = _merge_tags(
                        confused_annotation.edge_tags,
                        ["class_confusion"],
                    )

            rows.append(
                Prediction(
                    run_id=run.id,
                    job_id=job.id,
                    image_id=image.id,
                    class_id=class_id,
                    x_center=float(prediction["x_center"]),
                    y_center=float(prediction["y_center"]),
                    width=float(prediction["width"]),
                    height=float(prediction["height"]),
                    confidence=float(prediction["confidence"]),
                    matched_annotation_id=matched_annotation_id,
                    failure_type=failure_type,
                )
            )

        for annotation in image_annotations:
            if annotation.id in matched_annotations:
                continue
            annotation.edge_tags = _merge_tags(annotation.edge_tags, ["false_negative"])
            rows.append(
                Prediction(
                    run_id=run.id,
                    job_id=job.id,
                    image_id=image.id,
                    class_id=annotation.class_id,
                    x_center=annotation.x_center,
                    y_center=annotation.y_center,
                    width=annotation.width,
                    height=annotation.height,
                    confidence=0,
                    matched_annotation_id=annotation.id,
                    failure_type="false_negative",
                )
            )

    for row in rows:
        db.add(row)

    job.image_count = len(images)
    job.prediction_count = len(rows)
    job.matched_count = sum(1 for row in rows if row.failure_type == "matched")
    job.false_positive_count = sum(1 for row in rows if row.failure_type == "false_positive")
    job.false_negative_count = sum(1 for row in rows if row.failure_type == "false_negative")
    (Path(job.artifact_path) / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "image_id": row.image_id,
                    "class_id": row.class_id,
                    "x_center": row.x_center,
                    "y_center": row.y_center,
                    "width": row.width,
                    "height": row.height,
                    "confidence": row.confidence,
                    "matched_annotation_id": row.matched_annotation_id,
                    "failure_type": row.failure_type,
                }
                for row in rows
            ],
            indent=2,
        )
        + "\n"
    )
    db.commit()


def execute_prediction_job(db: Session, job: PredictionJob, run: TrainingRun, predictor=None) -> PredictionJob:
    job.status = "running"
    job.started_at = _now()
    db.commit()
    db.refresh(job)
    append_prediction_log(job, "prediction running")

    images = list_job_images(db, run, job.image_scope)
    try:
        prediction_fn = predictor or predict_images
        predictions = prediction_fn(run, images, job.confidence_threshold)
        persist_predictions(db, job, run, images, predictions)
        job.status = "completed"
        job.ended_at = _now()
        db.commit()
        db.refresh(job)
        append_prediction_log(job, "prediction completed")
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.ended_at = _now()
        db.commit()
        db.refresh(job)
        append_prediction_log(job, f"prediction failed: {exc}")
    return job


def read_prediction_logs(job: PredictionJob) -> str:
    path = Path(job.log_path)
    if not path.exists():
        return ""
    return path.read_text()
