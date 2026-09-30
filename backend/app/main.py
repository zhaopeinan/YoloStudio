from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import inspect
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.annotations.router import router as annotations_router
from app.auth.router import router as auth_router
from app.auth.service import ensure_default_admin
from app.classes.router import router as classes_router
from app.core.devices import detect_devices
from app.core.settings import REPO_ROOT, Settings, get_settings
from app.datasets.router import images_router, projects_router, router as datasets_router
from app.db.session import create_engine_for_settings, init_db
from app.exports.router import router as exports_router
from app.prediction.router import router as prediction_router
from app.preview.router import router as preview_router
from app.quality.router import router as quality_router
from app.storage.router import router as storage_router
from app.storage.service import purge_expired_trash, reconcile_trash
from app.training.router import router as training_router
from app.users.router import router as users_router
from app.versions.router import project_versions_router, router as versions_router


logger = logging.getLogger(__name__)
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def run_storage_maintenance(
    settings: Settings,
    db_engine: Engine | None = None,
) -> None:
    owns_engine = db_engine is None
    engine = db_engine or create_engine_for_settings(settings)
    try:
        init_db(engine)
        with Session(engine) as db:
            ensure_default_admin(db, settings)
            reconcile_trash(db, settings)
            purge_expired_trash(db, settings)
    finally:
        if owns_engine:
            engine.dispose()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings_override = application.dependency_overrides.get(get_settings)
    settings = settings_override() if settings_override is not None else get_settings()
    if inspect.isawaitable(settings):
        settings = await settings
    try:
        run_storage_maintenance(settings)
    except Exception:
        logger.exception("Storage startup maintenance failed")
    yield


app = FastAPI(title="YOLO Trainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://172.31.246.73:5173",
        "http://172.31.246.73:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(datasets_router)
app.include_router(images_router)
app.include_router(projects_router)
app.include_router(classes_router)
app.include_router(annotations_router)
app.include_router(quality_router)
app.include_router(versions_router)
app.include_router(project_versions_router)
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(preview_router)
app.include_router(exports_router)
app.include_router(storage_router)


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    device_info = detect_devices()
    return {
        "status": "ok",
        "app": "YOLO Trainer",
        "workspace_root": str(settings.workspace_root),
        "database_path": str(settings.database_path),
        "devices": {
            "selected": device_info.selected,
            "available": device_info.available,
            "details": device_info.details,
        },
    }


if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def frontend_spa(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (FRONTEND_DIST / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
