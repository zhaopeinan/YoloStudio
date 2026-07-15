from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.core.devices import detect_devices
from app.core.settings import Settings, get_settings
from app.db.models import DatasetVersion, RunMetric, TrainingRun
from app.db.session import SessionLocal


ACTIVE_STATUSES = {"queued", "preparing", "running"}
DEFAULT_THRESHOLD_SCAN_VALUES = (0.15, 0.25, 0.35, 0.5, 0.65)
GRIDMASK_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_TRAINING_SPLITS = ("train", "val")
INVALID_TRAINING_SPLIT_MESSAGE = (
    "Dataset version requires at least one training image and one validation image. "
    "Annotate at least 2 images and create a new version."
)


def _now() -> datetime:
    return datetime.now(UTC)


def _session_factory(bind=None):
    return sessionmaker(
        bind=bind or SessionLocal.kw["bind"],
        autoflush=False,
        autocommit=False,
        future=True,
    )


def run_artifact_root(settings: Settings, project_id: int, run_id: int) -> Path:
    return settings.workspace_root / "projects" / str(project_id) / "runs" / str(run_id)


def build_run_config(request) -> dict:
    augmentation = request.augmentation.model_dump()
    return {
        "model": request.model,
        "epochs": request.epochs,
        "image_size": request.image_size,
        "batch_size": request.batch_size,
        "augmentation_preset": request.augmentation_preset,
        "augmentation": augmentation,
        "tta": request.tta,
        "threshold_scan": request.threshold_scan,
    }


def ultralytics_augmentation_kwargs(config: dict) -> dict:
    augmentation = dict(config.get("augmentation") or {})
    supported_keys = {
        "mosaic",
        "mixup",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "fliplr",
        "erasing",
    }
    return {key: value for key, value in augmentation.items() if key in supported_keys}


def _rewrite_data_yaml(source: Path, destination: Path, dataset_root: Path) -> None:
    lines = source.read_text().splitlines()
    rewritten = [f"path: {dataset_root}"]
    rewritten.extend(line for line in lines if not line.startswith("path:"))
    destination.write_text("\n".join(rewritten) + "\n")


def _gridmask_image(source: Path, destination: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required to apply GridMask augmentation") from exc

    with Image.open(source) as image:
        output = image.convert("RGB") if image.mode not in {"RGB", "RGBA"} else image.copy()
        width, height = output.size
        spacing = max(16, min(width, height) // 6)
        cutout = max(4, spacing // 3)
        draw = ImageDraw.Draw(output)
        for x in range(0, width, spacing):
            draw.rectangle((x, 0, min(width, x + cutout), height), fill=0)
        for y in range(0, height, spacing):
            draw.rectangle((0, y, width, min(height, y + cutout)), fill=0)
        output.save(destination)


def build_gridmask_dataset(version_root: Path, run_artifact_root: Path) -> Path:
    if not version_root.exists():
        raise RuntimeError("Dataset version artifact path was not found")

    target_root = run_artifact_root / "gridmask_dataset"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for child in version_root.iterdir():
        if child.name in {"images", "data.yaml"}:
            continue
        destination = target_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)

    source_images = version_root / "images"
    target_images = target_root / "images"
    if not source_images.exists():
        raise RuntimeError("Dataset version images directory was not found")

    for source in source_images.rglob("*"):
        relative = source.relative_to(source_images)
        destination = target_images / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in GRIDMASK_IMAGE_SUFFIXES:
            _gridmask_image(source, destination)
        else:
            shutil.copy2(source, destination)

    data_yaml = version_root / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError("Dataset version data.yaml was not found")
    _rewrite_data_yaml(data_yaml, target_root / "data.yaml", target_root)
    return target_root / "data.yaml"


def resolve_device(requested_device: str | None) -> str:
    if requested_device:
        return requested_device
    return detect_devices().selected


def validate_training_dataset(version: DatasetVersion, version_root: Path) -> None:
    resolved_version_root = version_root.resolve()
    manifest_images = version.split_manifest.get("images", [])
    for split in REQUIRED_TRAINING_SPLITS:
        split_entries = [item for item in manifest_images if item.get("split") == split]
        if not split_entries:
            raise RuntimeError(INVALID_TRAINING_SPLIT_MESSAGE)

        expected_image_root = (resolved_version_root / "images" / split).resolve()
        expected_label_root = (resolved_version_root / "labels" / split).resolve()
        for item in split_entries:
            image_relative_path = Path(str(item.get("export_image", "")))
            label_relative_path = Path(str(item.get("export_label", "")))
            if image_relative_path.is_absolute() or label_relative_path.is_absolute():
                raise RuntimeError(INVALID_TRAINING_SPLIT_MESSAGE)

            image_path = (resolved_version_root / image_relative_path).resolve()
            label_path = (resolved_version_root / label_relative_path).resolve()
            if (
                not image_path.is_relative_to(expected_image_root)
                or not label_path.is_relative_to(expected_label_root)
                or not image_path.is_file()
                or not label_path.is_file()
            ):
                raise RuntimeError(INVALID_TRAINING_SPLIT_MESSAGE)


def create_queued_run(db, settings: Settings, version: DatasetVersion, request) -> TrainingRun:
    active = db.scalar(
        select(TrainingRun)
        .where(TrainingRun.project_id == version.project_id, TrainingRun.status.in_(ACTIVE_STATUSES))
        .order_by(TrainingRun.id.desc())
    )
    if active is not None:
        raise ValueError("Another training run is already active")

    run = TrainingRun(
        project_id=version.project_id,
        version_id=version.id,
        status="queued",
        device=resolve_device(request.device),
        config=build_run_config(request),
        artifact_path="",
        log_path="",
    )
    db.add(run)
    db.flush()

    artifact_root = run_artifact_root(settings, version.project_id, run.id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run.artifact_path = str(artifact_root)
    run.log_path = str(artifact_root / "logs.txt")
    (artifact_root / "config.json").write_text(json.dumps(run.config, indent=2) + "\n")
    (artifact_root / "metrics.jsonl").write_text("")
    Path(run.log_path).write_text(_log_line("training queued"))
    db.commit()
    db.refresh(run)
    return run


def _log_line(message: str) -> str:
    return f"{_now().isoformat()} {message}\n"


def append_run_log(run_id: int, message: str, bind=None) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        Path(run.log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(run.log_path).open("a") as handle:
            handle.write(_log_line(message))


def mark_training_run_running(run_id: int, bind=None) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "running"
        run.started_at = run.started_at or _now()
        db.commit()


def record_run_metric(
    run_id: int,
    epoch: int | None,
    name: str,
    value: float,
    bind=None,
) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        metric = RunMetric(run_id=run.id, epoch=epoch, name=name, value=value)
        db.add(metric)
        metrics_path = Path(run.artifact_path) / "metrics.jsonl"
        with metrics_path.open("a") as handle:
            handle.write(
                json.dumps({"epoch": epoch, "name": name, "value": value}, sort_keys=True) + "\n"
            )
        db.commit()


def complete_training_run(run_id: int, bind=None) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "completed"
        run.ended_at = _now()
        db.commit()
    append_run_log(run_id, "training completed", bind=bind)


def cancel_training_run(run_id: int, bind=None) -> bool:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None or run.status not in ACTIVE_STATUSES:
            return False
        run.status = "cancelled"
        run.ended_at = _now()
        db.commit()
    append_run_log(run_id, "training cancelled", bind=bind)
    return True


def run_post_training_threshold_scan(
    run_id: int,
    bind=None,
    settings: Settings | None = None,
    predictor=None,
) -> None:
    from app.prediction.runner import create_prediction_job, execute_prediction_job, predict_images

    settings = settings or get_settings()
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None or not run.config.get("threshold_scan"):
            return
        append_run_log(run_id, "threshold scan started", bind=bind)
        for threshold in DEFAULT_THRESHOLD_SCAN_VALUES:
            job = create_prediction_job(
                db,
                settings,
                run,
                image_scope="all",
                confidence_threshold=threshold,
            )
            execute_prediction_job(db, job, run, predictor=predictor or predict_images)
        append_run_log(run_id, "threshold scan completed", bind=bind)


def fail_training_run(run_id: int, message: str, bind=None) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "failed"
        run.error_message = message
        run.ended_at = _now()
        db.commit()
    append_run_log(run_id, f"training failed: {message}", bind=bind)


def execute_training_run(run_id: int, bind=None, settings: Settings | None = None) -> None:
    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "preparing"
        run.started_at = _now()
        db.commit()
        db.refresh(run)
        version = db.get(DatasetVersion, run.version_id)
        data_yaml = Path(version.artifact_path) / "data.yaml" if version else None
        version_root = Path(version.artifact_path) if version else None

    append_run_log(run_id, "training preparing", bind=bind)
    if data_yaml is None or not data_yaml.exists():
        fail_training_run(run_id, "Dataset version data.yaml was not found", bind=bind)
        return
    if version_root is None or not version_root.is_dir():
        fail_training_run(run_id, "Dataset version artifact path was not found", bind=bind)
        return
    try:
        validate_training_dataset(version, version_root)
    except RuntimeError as exc:
        fail_training_run(run_id, str(exc), bind=bind)
        return

    try:
        from ultralytics import YOLO
    except Exception as exc:
        fail_training_run(
            run_id,
            f"Ultralytics is not available: {exc.__class__.__name__}",
            bind=bind,
        )
        return

    mark_training_run_running(run_id, bind=bind)

    session_factory = _session_factory(bind)
    with session_factory() as db:
        run = db.get(TrainingRun, run_id)
        config = dict(run.config)
        artifact_root = Path(run.artifact_path)
        device = run.device

    try:
        training_data_yaml = data_yaml
        if (config.get("augmentation") or {}).get("gridmask"):
            training_data_yaml = build_gridmask_dataset(version_root, artifact_root)
            append_run_log(
                run_id,
                f"gridmask dataset prepared: {training_data_yaml.parent}",
                bind=bind,
            )
        model = YOLO(config["model"])
        append_run_log(run_id, "ultralytics training started", bind=bind)
        results = model.train(
            data=str(training_data_yaml),
            epochs=config["epochs"],
            imgsz=config["image_size"],
            batch=config["batch_size"],
            device=device,
            project=str(artifact_root),
            name="ultralytics",
            exist_ok=True,
            **ultralytics_augmentation_kwargs(config),
        )
        metrics = getattr(results, "results_dict", {}) or {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                record_run_metric(
                    run_id,
                    epoch=config["epochs"],
                    name=name,
                    value=float(value),
                    bind=bind,
                )
        complete_training_run(run_id, bind=bind)
        run_post_training_threshold_scan(
            run_id,
            bind=bind,
            settings=settings or get_settings(),
        )
    except Exception as exc:
        fail_training_run(run_id, str(exc), bind=bind)


def latest_metrics_for_run(db, run_id: int) -> dict[str, float]:
    rows = db.scalars(
        select(RunMetric).where(RunMetric.run_id == run_id).order_by(RunMetric.id)
    ).all()
    latest: dict[str, float] = {}
    for metric in rows:
        latest[metric.name] = metric.value
    return latest


def read_run_logs(run: TrainingRun) -> str:
    path = Path(run.log_path)
    if not path.exists():
        return ""
    return path.read_text()
