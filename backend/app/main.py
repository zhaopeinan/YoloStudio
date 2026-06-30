from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.annotations.router import router as annotations_router
from app.classes.router import router as classes_router
from app.core.devices import detect_devices
from app.core.settings import get_settings
from app.datasets.router import images_router, router as datasets_router
from app.db.session import init_db
from app.quality.router import router as quality_router
from app.training.router import router as training_router
from app.versions.router import router as versions_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="YOLO Trainer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(images_router)
app.include_router(classes_router)
app.include_router(annotations_router)
app.include_router(quality_router)
app.include_router(versions_router)
app.include_router(training_router)


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
