from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings
from app.db.models import Annotation, ClassDef, Image, PreviewJob, PreviewVideo, TrainingRun
from app.preview.schemas import PreviewBox
from app.storage.visibility import active_entity_predicate


ACTIVE_PREVIEW_STATUSES = {"queued", "preparing", "running"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024


def _now() -> datetime:
    return datetime.now(UTC)


def _base_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "yolov8n.pt"


def _safe_workspace_path(settings: Settings, value: str | Path) -> Path:
    root = settings.workspace_root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("路径必须位于工作空间内")
    return resolved


def _validate_video_filename(source_filename: str) -> str:
    suffix = Path(source_filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError("仅支持 MP4、MOV、AVI、MKV 和 WEBM 视频")
    return suffix


def _ensure_no_active_preview(db: Session, project_id: int) -> None:
    active_job = db.scalar(
        select(PreviewJob.id).where(
            PreviewJob.project_id == project_id,
            PreviewJob.status.in_(ACTIVE_PREVIEW_STATUSES),
        )
    )
    if active_job is not None:
        raise ValueError("当前项目已有视频预览正在处理")


def preview_video_root(settings: Settings, project_id: int, video_id: int) -> Path:
    return (
        settings.workspace_root
        / "projects"
        / str(project_id)
        / "preview-videos"
        / str(video_id)
    )


def preview_root(settings: Settings, project_id: int, preview_id: int) -> Path:
    return settings.workspace_root / "projects" / str(project_id) / "previews" / str(preview_id)


def create_preview_video(
    db: Session,
    settings: Settings,
    project_id: int,
    source_filename: str,
    source_bytes: bytes,
) -> PreviewVideo:
    if len(source_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("视频文件不能超过 500 MB")
    suffix = _validate_video_filename(source_filename)
    video = PreviewVideo(
        project_id=project_id,
        original_filename=Path(source_filename).name,
        stored_path="",
        size_bytes=len(source_bytes),
    )
    db.add(video)
    db.flush()
    root = preview_video_root(settings, project_id, video.id)
    root.mkdir(parents=True, exist_ok=True)
    stored_path = root / f"source{suffix}"
    stored_path.write_bytes(source_bytes)
    video.stored_path = str(stored_path)
    db.commit()
    db.refresh(video)
    return video


def list_preview_videos(db: Session, project_id: int) -> list[PreviewVideo]:
    return list(
        db.scalars(
            select(PreviewVideo)
            .where(PreviewVideo.project_id == project_id)
            .order_by(PreviewVideo.id.desc())
        ).all()
    )


def delete_preview_video(db: Session, settings: Settings, video_id: int) -> None:
    video = db.get(PreviewVideo, video_id)
    if video is None:
        raise LookupError("视频不存在")
    active = db.scalar(
        select(PreviewJob.id).where(
            PreviewJob.video_id == video_id,
            PreviewJob.status.in_(ACTIVE_PREVIEW_STATUSES),
        )
    )
    if active is not None:
        raise ValueError("该视频仍有预览任务正在处理，完成后才能删除")
    root = preview_video_root(settings, video.project_id, video.id)
    # Detach completed jobs from this video so FK does not block delete.
    for job in db.scalars(select(PreviewJob).where(PreviewJob.video_id == video_id)).all():
        job.video_id = None
    db.delete(video)
    db.commit()
    shutil.rmtree(root, ignore_errors=True)


def create_video_job_from_video(
    db: Session,
    settings: Settings,
    project_id: int,
    video_id: int,
    model_ref: str,
    confidence_threshold: float,
    frame_step: int,
) -> PreviewJob:
    video = db.get(PreviewVideo, video_id)
    if video is None or video.project_id != project_id:
        raise LookupError("视频不存在")
    source_path = Path(video.stored_path)
    if not source_path.is_file():
        raise ValueError("视频文件不存在")
    resolve_model_path(db, settings, model_ref)
    _ensure_no_active_preview(db, project_id)

    job = PreviewJob(
        project_id=project_id,
        video_id=video.id,
        model_ref=model_ref,
        kind="video",
        source_filename=video.original_filename,
        source_path=str(source_path),
        frame_directory="",
        status="queued",
        confidence_threshold=confidence_threshold,
        frame_step=frame_step,
    )
    db.add(job)
    db.flush()
    root = preview_root(settings, project_id, job.id)
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    job.frame_directory = str(frames)
    db.commit()
    db.refresh(job)
    return job


def create_video_job(
    db: Session,
    settings: Settings,
    project_id: int,
    model_ref: str,
    confidence_threshold: float,
    frame_step: int,
    source_filename: str,
    source_bytes: bytes,
) -> PreviewJob:
    """Upload into the video library and immediately queue a preview job."""
    resolve_model_path(db, settings, model_ref)
    _ensure_no_active_preview(db, project_id)
    video = create_preview_video(db, settings, project_id, source_filename, source_bytes)
    return create_video_job_from_video(
        db,
        settings,
        project_id,
        video.id,
        model_ref,
        confidence_threshold,
        frame_step,
    )

def resolve_model_path(
    db: Session, settings: Settings, model_ref: str
) -> tuple[Path, int | None]:
    if model_ref == "base:yolov8n.pt":
        path = _base_model_path().resolve()
        if not path.is_file():
            raise ValueError("内置模型 yolov8n.pt 不存在")
        return path, None

    if not model_ref.startswith("run:") or not model_ref[4:].isdigit():
        raise ValueError("模型引用无效，请选择内置模型或已完成训练任务")
    run_id = int(model_ref[4:])
    run = db.scalar(
        select(TrainingRun).where(
            TrainingRun.id == run_id,
            TrainingRun.status == "completed",
            active_entity_predicate("training_run", TrainingRun.id),
        )
    )
    if run is None:
        raise ValueError("训练任务不存在或尚未完成")
    artifact_root = _safe_workspace_path(settings, run.artifact_path)
    weights = (artifact_root / "ultralytics" / "weights" / "best.pt").resolve()
    if not weights.is_relative_to(artifact_root) or not weights.is_file():
        raise ValueError("训练任务缺少 best.pt 权重文件")
    return weights, run.id


@lru_cache(maxsize=16)
def load_preview_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics 未安装，无法进行模型预览") from exc
    return YOLO(model_path)


def _model_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, f"类别 {class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"类别 {class_id}"


def _prediction_boxes(model: Any, source: Any, confidence_threshold: float) -> list[PreviewBox]:
    results = model(source, conf=confidence_threshold, verbose=False)
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    try:
        xywhn = boxes.xywhn.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
    except AttributeError:
        xywhn = boxes.xywhn.tolist()
        classes = boxes.cls.tolist()
        confidences = boxes.conf.tolist()

    output: list[PreviewBox] = []
    for values, class_value, confidence in zip(xywhn, classes, confidences):
        class_id = int(class_value)
        x_center, y_center, width, height = (float(value) for value in values[:4])
        output.append(
            PreviewBox(
                class_id=class_id,
                class_name=_model_name(model, class_id),
                color="#f59e0b",
                x_center=max(0.0, min(1.0, x_center)),
                y_center=max(0.0, min(1.0, y_center)),
                width=max(0.0001, min(1.0, width)),
                height=max(0.0001, min(1.0, height)),
                confidence=max(0.0, min(1.0, float(confidence))),
            )
        )
    return output


def predict_image(model_path: Path, image_path: Path, confidence_threshold: float) -> list[PreviewBox]:
    return _prediction_boxes(load_preview_model(str(model_path)), str(image_path), confidence_threshold)


def _annotation_boxes(db: Session, image_id: int) -> list[PreviewBox]:
    rows = db.execute(
        select(Annotation, ClassDef)
        .join(ClassDef, Annotation.class_id == ClassDef.id)
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.id)
    ).all()
    return [
        PreviewBox(
            class_id=annotation.class_id,
            class_name=class_def.name,
            color=class_def.color,
            x_center=annotation.x_center,
            y_center=annotation.y_center,
            width=annotation.width,
            height=annotation.height,
        )
        for annotation, class_def in rows
    ]


def preview_image(
    db: Session,
    settings: Settings,
    image: Image,
    model_ref: str,
    confidence_threshold: float,
) -> tuple[list[PreviewBox], list[PreviewBox]]:
    model_path, _ = resolve_model_path(db, settings, model_ref)
    image_path = _safe_workspace_path(settings, image.relative_path)
    if not image_path.is_file():
        raise ValueError("图像文件不存在")
    return _annotation_boxes(db, image.id), _prediction_boxes(
        load_preview_model(str(model_path)), str(image_path), confidence_threshold
    )


def _draw_predictions(frame: Any, predictions: list[PreviewBox]) -> Any:
    import cv2

    height, width = frame.shape[:2]
    for prediction in predictions:
        left = int((prediction.x_center - prediction.width / 2) * width)
        top = int((prediction.y_center - prediction.height / 2) * height)
        right = int((prediction.x_center + prediction.width / 2) * width)
        bottom = int((prediction.y_center + prediction.height / 2) * height)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 180, 255), 2)
        label = f"{prediction.class_name} {prediction.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (max(left, 0), max(top - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )
    return frame


def _mux_audio(source: Path, silent: Path, result: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-i",
        str(silent),
        "-map",
        "1:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(result),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        # ffmpeg is optional: keep the silent OpenCV output when it is unavailable.
        shutil.copy2(silent, result)
        return
    if completed.returncode != 0:
        shutil.copy2(silent, result)


def process_video_preview(preview_id: int, bind, settings: Settings) -> None:
    import cv2

    factory = sessionmaker(bind=bind, autoflush=False, autocommit=False, future=True)
    with factory() as db:
        job = db.get(PreviewJob, preview_id)
        if job is None or job.status == "deleted":
            return
        try:
            model_path, run_id = resolve_model_path(db, settings, job.model_ref)
            job.run_id = run_id
            job.status = "preparing"
            job.started_at = _now()
            db.commit()

            capture = cv2.VideoCapture(job.source_path)
            if not capture.isOpened():
                raise RuntimeError("无法读取视频文件")
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                capture.release()
                raise RuntimeError("无法读取视频尺寸")
            job.fps = fps
            job.total_frames = total_frames
            job.status = "running"
            db.commit()

            silent_path = Path(job.frame_directory).parent / "silent.mp4"
            result_path = Path(job.frame_directory).parent / "result.mp4"
            writer = cv2.VideoWriter(
                str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            model = load_preview_model(str(model_path))
            frame_index = 0
            processed = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % job.frame_step == 0:
                    predictions = _prediction_boxes(model, frame, job.confidence_threshold)
                    frame = _draw_predictions(frame, predictions)
                    processed += 1
                    job.processed_frames = processed
                    db.commit()
                writer.write(frame)
                cv2.imwrite(str(Path(job.frame_directory) / f"{frame_index:08d}.jpg"), frame)
                frame_index += 1
            capture.release()
            writer.release()
            _mux_audio(Path(job.source_path), silent_path, result_path)
            job.result_path = str(result_path)
            job.status = "completed"
            job.ended_at = _now()
            db.commit()
            silent_path.unlink(missing_ok=True)
        except Exception as exc:
            db.rollback()
            failed_job = db.get(PreviewJob, preview_id)
            if failed_job is not None:
                failed_job.status = "failed"
                failed_job.error_message = str(exc)
                failed_job.ended_at = _now()
                db.commit()
                root = preview_root(settings, failed_job.project_id, preview_id)
                (root / "result.mp4").unlink(missing_ok=True)
