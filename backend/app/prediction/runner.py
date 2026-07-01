from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import exists, select, text
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
from app.prediction.schemas import PredictionImageFilters


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
    image_filters: PredictionImageFilters | None = None,
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
    _write_prediction_job_metadata(job, image_filters)
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


def _normalize_image_filters(
    image_filters: PredictionImageFilters | dict | None,
) -> dict[str, str | int | float | None]:
    if image_filters is None:
        return {
            "platform": None,
            "label_status": "all",
            "class_id": None,
            "edge_tag": None,
            "failure_type": "all",
            "altitude_min": None,
            "altitude_max": None,
        }
    if isinstance(image_filters, PredictionImageFilters):
        return image_filters.model_dump()
    return PredictionImageFilters.model_validate(image_filters).model_dump()


def _has_active_image_filters(image_filters: PredictionImageFilters | dict | None) -> bool:
    filters = _normalize_image_filters(image_filters)
    return any(
        [
            filters["platform"],
            filters["label_status"] != "all",
            filters["class_id"] is not None,
            filters["edge_tag"],
            filters["failure_type"] != "all",
            filters["altitude_min"] is not None,
            filters["altitude_max"] is not None,
        ]
    )


def _prediction_job_metadata_path(job: PredictionJob) -> Path:
    return Path(job.artifact_path) / "job_config.json"


def _write_prediction_job_metadata(
    job: PredictionJob,
    image_filters: PredictionImageFilters | dict | None,
) -> None:
    path = _prediction_job_metadata_path(job)
    path.write_text(
        json.dumps(
            {
                "image_filters": _normalize_image_filters(image_filters),
                "uses_image_filters": _has_active_image_filters(image_filters),
            },
            indent=2,
        )
        + "\n"
    )


def read_prediction_job_filters(job: PredictionJob) -> PredictionImageFilters | None:
    path = _prediction_job_metadata_path(job)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    filters = payload.get("image_filters")
    if not isinstance(filters, dict) or not payload.get("uses_image_filters"):
        return None
    return PredictionImageFilters.model_validate(filters)


def _apply_image_filters(
    query,
    image_filters: PredictionImageFilters | dict | None,
):
    filters = _normalize_image_filters(image_filters)
    if filters["platform"]:
        query = query.where(Image.platform == filters["platform"])
    if filters["altitude_min"] is not None:
        query = query.where(Image.altitude >= filters["altitude_min"])
    if filters["altitude_max"] is not None:
        query = query.where(Image.altitude <= filters["altitude_max"])
    if filters["label_status"] == "annotated":
        query = query.where(exists().where(Annotation.image_id == Image.id))
    elif filters["label_status"] == "unannotated":
        query = query.where(~exists().where(Annotation.image_id == Image.id))
    if filters["class_id"] is not None:
        query = query.where(
            exists().where(
                Annotation.image_id == Image.id,
                Annotation.class_id == filters["class_id"],
            )
        )
    if filters["edge_tag"]:
        query = query.where(
            exists().where(
                Annotation.image_id == Image.id,
                text(
                    "EXISTS (SELECT 1 FROM json_each(annotations.edge_tags) "
                    "WHERE json_each.value = :edge_tag)"
                ).bindparams(edge_tag=filters["edge_tag"]),
            )
        )
    if filters["failure_type"] != "all":
        query = query.where(
            exists().where(
                Prediction.image_id == Image.id,
                Prediction.failure_type == filters["failure_type"],
            )
        )
    return query


def list_job_images(
    db: Session,
    run: TrainingRun,
    image_scope: str,
    image_filters: PredictionImageFilters | dict | None = None,
) -> list[Image]:
    version = db.get(DatasetVersion, run.version_id)
    if version is None:
        return []
    image_ids = _manifest_image_ids(version, image_scope)
    if not image_ids:
        return []
    query = select(Image).where(Image.id.in_(image_ids))
    if image_filters is not None:
        query = _apply_image_filters(query, image_filters)
    return db.scalars(query.order_by(Image.id)).all()


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
    image_filters: PredictionImageFilters | dict | None = None,
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
            {
                "image_scope": job.image_scope,
                "image_filters": _normalize_image_filters(image_filters),
                "uses_image_filters": _has_active_image_filters(image_filters),
                "image_ids": image_ids,
                "predictions": [
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
            },
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

    image_filters = read_prediction_job_filters(job)
    images = list_job_images(db, run, job.image_scope, image_filters)
    if image_filters is not None:
        append_prediction_log(
            job,
            f"prediction image filters: {json.dumps(image_filters.model_dump(), sort_keys=True)}",
        )
    try:
        prediction_fn = predictor or predict_images
        predictions = prediction_fn(run, images, job.confidence_threshold)
        persist_predictions(db, job, run, images, predictions, image_filters)
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
