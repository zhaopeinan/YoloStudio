from __future__ import annotations

import asyncio
import mimetypes
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.auth.deps import get_current_user, require_admin
from app.core.settings import Settings, get_settings
from app.db.models import Image, PreviewJob, PreviewVideo, Project, TrainingRun
from app.db.session import get_db
from app.preview.schemas import (
    ImagePreviewRequest,
    ImagePreviewResponse,
    PreviewJobList,
    PreviewJobRead,
    PreviewModelList,
    PreviewModelOption,
    PreviewVideoList,
    PreviewVideoRead,
)
from app.preview.service import (
    ACTIVE_PREVIEW_STATUSES,
    MAX_VIDEO_BYTES,
    create_preview_video,
    create_video_job,
    create_video_job_from_video,
    delete_preview_video,
    list_preview_videos,
    preview_image,
    preview_root,
    process_video_preview,
)
from app.storage.visibility import (
    StorageEntityNotFoundError,
    active_entity_predicate,
    require_active_entity,
)


router = APIRouter(
    prefix="/api/preview",
    tags=["preview"],
    dependencies=[Depends(get_current_user)],
)


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _read_video(video: PreviewVideo) -> PreviewVideoRead:
    return PreviewVideoRead(
        id=video.id,
        project_id=video.project_id,
        original_filename=video.original_filename,
        size_bytes=video.size_bytes,
        created_at=video.created_at,
    )


def _read_job(job: PreviewJob) -> PreviewJobRead:
    return PreviewJobRead(
        id=job.id,
        project_id=job.project_id,
        dataset_id=job.dataset_id,
        run_id=job.run_id,
        video_id=job.video_id,
        model_ref=job.model_ref,
        kind=job.kind,
        source_filename=job.source_filename,
        status=job.status,
        confidence_threshold=job.confidence_threshold,
        frame_step=job.frame_step,
        fps=job.fps,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        error_message=job.error_message,
        result_url=f"/api/preview/video-jobs/{job.id}/result" if job.result_path else None,
        stream_url=(
            f"/api/preview/video-jobs/{job.id}/stream"
            if job.status in ACTIVE_PREVIEW_STATUSES
            else None
        ),
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
    )


@router.get("/models", response_model=PreviewModelList)
def list_preview_models(
    project_id: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> PreviewModelList:
    items = [
        PreviewModelOption(
            model_ref="base:yolov8n.pt",
            label="内置模型 · YOLOv8n",
            kind="base",
            status="可用",
        )
    ]
    query = select(TrainingRun).where(
        TrainingRun.status == "completed",
        active_entity_predicate("training_run", TrainingRun.id),
    )
    if project_id is not None:
        query = query.where(TrainingRun.project_id == project_id)
    for run in db.scalars(query.order_by(TrainingRun.id.desc())).all():
        weights = Path(run.artifact_path) / "ultralytics" / "weights" / "best.pt"
        if weights.is_file():
            items.append(
                PreviewModelOption(
                    model_ref=f"run:{run.id}",
                    label=f"训练任务 #{run.id} · best.pt",
                    kind="trained",
                    run_id=run.id,
                    status=run.status,
                )
            )
    return PreviewModelList(items=items)


@router.post("/images/{image_id}", response_model=ImagePreviewResponse)
def preview_image_endpoint(
    image_id: int,
    request: ImagePreviewRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ImagePreviewResponse:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="图像不存在")
    try:
        require_active_entity(db, "dataset", image.dataset_id)
        annotations, predictions = preview_image(
            db, settings, image, request.model_ref, request.confidence_threshold
        )
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if image.width is None or image.height is None:
        raise HTTPException(status_code=400, detail="图像尺寸不可用，请先刷新图像尺寸")
    return ImagePreviewResponse(
        image_id=image.id,
        filename=Path(image.relative_path).name,
        image_url=f"/api/images/{image.id}/file",
        width=image.width,
        height=image.height,
        model_ref=request.model_ref,
        confidence_threshold=request.confidence_threshold,
        annotations=annotations,
        predictions=predictions,
    )


@router.get("/videos", response_model=PreviewVideoList)
def list_preview_video_library(
    project_id: int = Query(..., ge=1),
    db: Session = Depends(get_db),
    _: object = Depends(require_admin),
) -> PreviewVideoList:
    _require_project(db, project_id)
    return PreviewVideoList(items=[_read_video(video) for video in list_preview_videos(db, project_id)])


@router.post("/videos", response_model=PreviewVideoRead)
async def upload_preview_video(
    project_id: int = Form(..., ge=1),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> PreviewVideoRead:
    _require_project(db, project_id)
    content = await video.read(MAX_VIDEO_BYTES + 1)
    try:
        saved = create_preview_video(
            db,
            settings,
            project_id,
            video.filename or "uploaded-video.mp4",
            content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _read_video(saved)


@router.delete("/videos/{video_id}", status_code=204)
def remove_preview_video(
    video_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> None:
    try:
        delete_preview_video(db, settings, video_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/video-jobs", response_model=PreviewJobRead)
async def create_preview_video_job(
    background_tasks: BackgroundTasks,
    project_id: int = Form(..., ge=1),
    model_ref: str = Form(..., min_length=1, max_length=240),
    confidence_threshold: float = Form(0.25, ge=0.01, le=0.99),
    frame_step: int = Form(1, ge=1, le=30),
    video_id: int | None = Form(default=None),
    video: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> PreviewJobRead:
    _require_project(db, project_id)
    has_upload = video is not None and bool(video.filename)
    has_video_id = video_id is not None
    if has_upload == has_video_id:
        raise HTTPException(status_code=400, detail="请上传视频文件或指定已保存的 video_id，二者选一")
    try:
        if has_video_id:
            assert video_id is not None
            job = create_video_job_from_video(
                db,
                settings,
                project_id,
                video_id,
                model_ref,
                confidence_threshold,
                frame_step,
            )
        else:
            assert video is not None
            content = await video.read(MAX_VIDEO_BYTES + 1)
            job = create_video_job(
                db,
                settings,
                project_id,
                model_ref,
                confidence_threshold,
                frame_step,
                video.filename or "uploaded-video.mp4",
                content,
            )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(process_video_preview, job.id, db.get_bind(), settings)
    return _read_job(job)


@router.get("/video-jobs", response_model=PreviewJobList)
def list_preview_jobs(
    project_id: int = Query(..., ge=1), db: Session = Depends(get_db)
) -> PreviewJobList:
    _require_project(db, project_id)
    jobs = db.scalars(
        select(PreviewJob).where(PreviewJob.project_id == project_id).order_by(PreviewJob.id.desc())
    ).all()
    return PreviewJobList(items=[_read_job(job) for job in jobs])


@router.get("/video-jobs/{preview_id}", response_model=PreviewJobRead)
def get_preview_video_job(preview_id: int, db: Session = Depends(get_db)) -> PreviewJobRead:
    job = db.get(PreviewJob, preview_id)
    if job is None:
        raise HTTPException(status_code=404, detail="视频预览任务不存在")
    return _read_job(job)


async def _frame_stream(preview_id: int, bind) -> AsyncIterator[bytes]:
    factory = sessionmaker(bind=bind, autoflush=False, autocommit=False, future=True)
    frame_index = 0
    while True:
        with factory() as db:
            job = db.get(PreviewJob, preview_id)
            if job is None:
                break
            frame_directory = Path(job.frame_directory)
            status = job.status
        frame = frame_directory / f"{frame_index:08d}.jpg"
        if frame.is_file():
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame.read_bytes()
                + b"\r\n"
            )
            frame_index += 1
            continue
        if status in {"completed", "failed", "deleted"}:
            break
        await asyncio.sleep(0.15)


@router.get("/video-jobs/{preview_id}/stream")
def stream_preview_video(preview_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    job = db.get(PreviewJob, preview_id)
    if job is None:
        raise HTTPException(status_code=404, detail="视频预览任务不存在")
    return StreamingResponse(
        _frame_stream(preview_id, db.get_bind()),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video-jobs/{preview_id}/result")
def get_preview_video_result(preview_id: int, db: Session = Depends(get_db)):
    job = db.get(PreviewJob, preview_id)
    if job is None or job.result_path is None or job.status != "completed":
        raise HTTPException(status_code=404, detail="视频结果尚未生成")
    path = Path(job.result_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="视频结果文件不存在")
    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
        filename=f"{Path(job.source_filename).stem}-preview.mp4",
    )


@router.delete("/video-jobs/{preview_id}", status_code=204)
def delete_preview_video_job(
    preview_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _: object = Depends(require_admin),
) -> None:
    job = db.get(PreviewJob, preview_id)
    if job is None:
        raise HTTPException(status_code=404, detail="视频预览任务不存在")
    if job.status in ACTIVE_PREVIEW_STATUSES:
        raise HTTPException(status_code=409, detail="视频仍在处理，完成后才能删除结果")
    root = preview_root(settings, job.project_id, job.id)
    db.delete(job)
    db.commit()
    shutil.rmtree(root, ignore_errors=True)
