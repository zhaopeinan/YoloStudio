# YOLO Trainer Scaffold And Import Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working vertical slice of the local YOLO trainer: repo scaffold, FastAPI backend, SQLite workspace foundation, dataset zip scanner, and React/Vite UI that previews the provided dataset.

**Architecture:** Use a two-service local app: FastAPI owns workspace, database, dataset scanning, and API state; React/Vite owns the browser UI. This plan intentionally stops before the full annotator/training adapter, but it creates the module boundaries and durable schema those features will use.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x, Pydantic Settings, pytest, httpx, React, Vite, TypeScript, Vitest, Testing Library, lucide-react.

---

## Scope Check

The approved spec covers a full local YOLO workbench. This plan implements the first vertical slice only:

- App scaffold and local development commands.
- Backend health/config API.
- SQLite models for the initial project/dataset/image foundation.
- Device detection shell for `cuda`, `mps`, and `cpu`.
- Zip scanner for the current `image_dataset.zip` shape.
- Frontend dashboard and dataset scan preview wired to the backend.

The next plan should start with the lightweight annotator and class library once this slice is verified.

## File Structure

Create or modify these files:

- `.gitignore`: ignore generated Python, Node, database, workspace, model, and training artifacts.
- `README.md`: local setup and verification commands.
- `backend/pyproject.toml`: backend dependencies and pytest configuration.
- `backend/app/__init__.py`: backend package marker.
- `backend/app/main.py`: FastAPI app assembly and health endpoint.
- `backend/app/core/__init__.py`: core package marker.
- `backend/app/core/settings.py`: repo-aware workspace and database settings.
- `backend/app/core/devices.py`: local device capability detection.
- `backend/app/db/__init__.py`: database package marker.
- `backend/app/db/models.py`: SQLAlchemy entities for projects, datasets, and images.
- `backend/app/db/session.py`: engine/session creation and database initialization.
- `backend/app/datasets/__init__.py`: datasets package marker.
- `backend/app/datasets/schemas.py`: Pydantic request/response schemas for scanning.
- `backend/app/datasets/scanner.py`: zip scanner and `meta.jsonl` parser.
- `backend/app/datasets/router.py`: dataset scanning API route.
- `backend/tests/test_health.py`: API health tests.
- `backend/tests/test_settings_and_db.py`: settings and SQLite initialization tests.
- `backend/tests/test_dataset_scanner.py`: zip scanner tests.
- `frontend/package.json`: frontend dependencies and scripts.
- `frontend/index.html`: Vite entry document.
- `frontend/tsconfig.json`: TypeScript config.
- `frontend/tsconfig.node.json`: Vite config TypeScript config.
- `frontend/vite.config.ts`: Vite and Vitest config.
- `frontend/src/main.tsx`: React entry.
- `frontend/src/App.tsx`: dashboard and scan preview UI.
- `frontend/src/api.ts`: typed API client.
- `frontend/src/styles.css`: app layout and component styling.
- `frontend/src/App.test.tsx`: frontend smoke tests.

## Task 1: Repository And Backend Test Harness

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/test_health.py`

- [ ] **Step 1: Write the failing health test**

Create `backend/tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_local_app_state():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "YOLO Trainer"
    assert "workspace_root" in payload
    assert "database_path" in payload
    assert payload["devices"]["selected"] in {"cuda", "mps", "cpu"}
```

- [ ] **Step 2: Run the test to verify it fails before scaffold code exists**

Run:

```bash
cd backend
python -m pytest tests/test_health.py -v
```

Expected: FAIL with an import error for `app.main` or missing FastAPI dependencies.

- [ ] **Step 3: Add backend package and health endpoint**

Create `.gitignore`:

```gitignore
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
node_modules/
dist/
coverage/
.coverage
workspace/
*.sqlite
*.db
*.pt
*.onnx
*.engine
runs/
```

Create `backend/pyproject.toml`:

```toml
[project]
name = "yolo-trainer-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115.0",
  "httpx>=0.27.0",
  "pydantic-settings>=2.4.0",
  "sqlalchemy>=2.0.30",
  "uvicorn[standard]>=0.30.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
  "ruff>=0.5.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Create `backend/app/__init__.py`:

```python
"""YOLO Trainer backend package."""
```

Create `backend/app/main.py`:

```python
from fastapi import FastAPI


app = FastAPI(title="YOLO Trainer API")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": "YOLO Trainer",
        "workspace_root": "",
        "database_path": "",
        "devices": {"selected": "cpu", "available": ["cpu"]},
    }
```

Create `README.md`:

```markdown
# YOLO Trainer

Local single-user YOLO training workbench.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -v
python -m uvicorn app.main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```
```

- [ ] **Step 4: Run the health test**

Run:

```bash
cd backend
python -m pip install -e ".[dev]"
python -m pytest tests/test_health.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add .gitignore README.md backend
git commit -m "feat: scaffold backend health API"
```

## Task 2: Settings, Workspace, Device Detection, And SQLite Foundation

**Files:**
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/settings.py`
- Create: `backend/app/core/devices.py`
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/models.py`
- Create: `backend/app/db/session.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_settings_and_db.py`
- Modify: `backend/tests/test_health.py`

- [ ] **Step 1: Write failing settings and database tests**

Create `backend/tests/test_settings_and_db.py`:

```python
from pathlib import Path

from sqlalchemy import inspect

from app.core.settings import Settings
from app.db.session import create_engine_for_settings, init_db


def test_settings_resolve_workspace_and_database_paths(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path / "workspace")

    assert settings.workspace_root == tmp_path / "workspace"
    assert settings.database_path == tmp_path / "workspace" / "app.db"
    assert settings.database_url.startswith("sqlite:///")


def test_init_db_creates_foundation_tables(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path / "workspace")
    engine = create_engine_for_settings(settings)

    init_db(engine)

    tables = set(inspect(engine).get_table_names())
    assert {"projects", "datasets", "images"}.issubset(tables)
    assert settings.workspace_root.exists()
```

Modify `backend/tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_local_app_state():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "YOLO Trainer"
    assert payload["workspace_root"]
    assert payload["database_path"].endswith("app.db")
    assert payload["devices"]["selected"] in {"cuda", "mps", "cpu"}
    assert "cpu" in payload["devices"]["available"]
```

- [ ] **Step 2: Run tests to verify the new expectations fail**

Run:

```bash
cd backend
python -m pytest tests/test_settings_and_db.py tests/test_health.py -v
```

Expected: FAIL because `app.core` and `app.db` modules do not exist.

- [ ] **Step 3: Add settings, devices, models, and database session**

Create `backend/app/core/__init__.py`:

```python
"""Core backend utilities."""
```

Create `backend/app/core/settings.py`:

```python
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOLO_TRAINER_", arbitrary_types_allowed=True)

    workspace_root: Path = Field(default=REPO_ROOT / "workspace")

    @property
    def database_path(self) -> Path:
        return self.workspace_root / "app.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    def ensure_workspace(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_workspace()
    return settings
```

Create `backend/app/core/devices.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    selected: str
    available: list[str]
    details: dict[str, str]


def detect_devices() -> DeviceInfo:
    available = ["cpu"]
    details: dict[str, str] = {}

    try:
        import torch

        if torch.cuda.is_available():
            available.insert(0, "cuda")
            details["cuda"] = torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            insert_at = 1 if "cuda" in available else 0
            available.insert(insert_at, "mps")
            details["mps"] = "Apple Metal Performance Shaders"
    except Exception as exc:
        details["torch"] = f"unavailable: {exc.__class__.__name__}"

    return DeviceInfo(selected=available[0], available=available, details=details)
```

Create `backend/app/db/__init__.py`:

```python
"""Database package."""
```

Create `backend/app/db/models.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="project")


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    import_status: Mapped[str] = mapped_column(String(40), nullable=False, default="scanned")
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    project: Mapped[Project] = relationship(back_populates="datasets")
    images: Mapped[list[Image]] = relationship(back_populates="dataset")


class Image(TimestampMixin, Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    platform: Mapped[str | None] = mapped_column(String(80), index=True)
    altitude: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[float | None] = mapped_column(Float)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    dataset: Mapped[Dataset] = relationship(back_populates="images")
```

Create `backend/app/db/session.py`:

```python
from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import Base


def create_engine_for_settings(settings: Settings) -> Engine:
    settings.ensure_workspace()
    return create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        future=True,
    )


engine = create_engine_for_settings(get_settings())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(db_engine: Engine = engine) -> None:
    Base.metadata.create_all(bind=db_engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Modify `backend/app/main.py`:

```python
from fastapi import FastAPI

from app.core.devices import detect_devices
from app.core.settings import get_settings
from app.db.session import init_db


app = FastAPI(title="YOLO Trainer API")


@app.on_event("startup")
def startup() -> None:
    init_db()


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
```

- [ ] **Step 4: Run backend tests**

Run:

```bash
cd backend
python -m pytest tests/test_settings_and_db.py tests/test_health.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add backend/app backend/tests
git commit -m "feat: add backend settings and database foundation"
```

## Task 3: Dataset Zip Scanner

**Files:**
- Create: `backend/app/datasets/__init__.py`
- Create: `backend/app/datasets/schemas.py`
- Create: `backend/app/datasets/scanner.py`
- Create: `backend/tests/test_dataset_scanner.py`

- [ ] **Step 1: Write failing scanner tests**

Create `backend/tests/test_dataset_scanner.py`:

```python
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.datasets.scanner import scan_dataset_zip


def create_sample_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z25m_00001.jpg", b"fake")
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z26m_00002.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/vtol/meta.jsonl",
            (
                '{"file":"frame_vtol_z25m_00001.jpg","drone":"vtol","z":25.5,"t":1.0}\\n'
                '{"file":"frame_vtol_z26m_00002.jpg","drone":"vtol","z":26.5,"t":2.0}\\n'
            ),
        )
        archive.writestr("yolo_dataset/iris/images/raw/frame_iris_z12m_00001.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/iris/meta.jsonl",
            '{"file":"frame_iris_z12m_00001.jpg","drone":"iris","z":12.0,"t":3.0}\\n',
        )


def test_scan_dataset_zip_counts_images_and_metadata(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_sample_zip(zip_path)

    result = scan_dataset_zip(zip_path)

    assert result.source_path == zip_path
    assert result.total_images == 3
    assert result.total_yolo_labels == 0
    assert result.total_yaml_files == 0
    assert result.total_metadata_rows == 3
    assert [group.name for group in result.groups] == ["iris", "vtol"]
    assert result.groups[0].image_count == 1
    assert result.groups[0].altitude_min == 12.0
    assert result.groups[1].image_count == 2
    assert result.groups[1].altitude_max == 26.5


def test_scan_dataset_zip_rejects_missing_zip(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        scan_dataset_zip(tmp_path / "missing.zip")
```

- [ ] **Step 2: Run scanner tests to verify failure**

Run:

```bash
cd backend
python -m pytest tests/test_dataset_scanner.py -v
```

Expected: FAIL because `app.datasets.scanner` does not exist.

- [ ] **Step 3: Add scanner schemas and implementation**

Create `backend/app/datasets/__init__.py`:

```python
"""Dataset import and scanning package."""
```

Create `backend/app/datasets/schemas.py`:

```python
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetScanRequest(BaseModel):
    source_path: Path = Field(..., description="Absolute path to a local dataset zip file")


class DatasetGroupSummary(BaseModel):
    name: str
    image_count: int
    metadata_rows: int
    altitude_min: float | None = None
    altitude_max: float | None = None
    first_timestamp: float | None = None
    last_timestamp: float | None = None


class DatasetScanSummary(BaseModel):
    source_path: Path
    archive_name: str
    total_files: int
    total_images: int
    total_yolo_labels: int
    total_yaml_files: int
    total_json_files: int
    total_metadata_rows: int
    has_yolo_labels: bool
    has_data_yaml: bool
    groups: list[DatasetGroupSummary]
    warnings: list[str]
```

Create `backend/app/datasets/scanner.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.datasets.schemas import DatasetGroupSummary, DatasetScanSummary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YAML_NAMES = {"data.yaml", "dataset.yaml"}


def _collection_name(zip_name: str) -> str:
    parts = [part for part in zip_name.split("/") if part]
    if "images" in parts:
        index = parts.index("images")
        if index > 0:
            return parts[index - 1]
    if len(parts) >= 2:
        return parts[1]
    return "ungrouped"


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scan_metadata_rows(archive: ZipFile, name: str) -> list[dict]:
    rows: list[dict] = []
    with archive.open(name) as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": line})
    return rows


def scan_dataset_zip(source_path: Path) -> DatasetScanSummary:
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file, got {source_path.name}")

    try:
        archive = ZipFile(source_path)
    except BadZipFile as exc:
        raise ValueError(f"Invalid zip archive: {source_path}") from exc

    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        image_names = [name for name in names if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
        label_names = [name for name in names if Path(name).suffix.lower() == ".txt"]
        yaml_names = [name for name in names if Path(name).name.lower() in YAML_NAMES]
        json_names = [
            name for name in names if Path(name).suffix.lower() in {".json", ".jsonl"}
        ]

        image_counts: dict[str, int] = defaultdict(int)
        metadata_rows_by_group: dict[str, list[dict]] = defaultdict(list)
        for image_name in image_names:
            image_counts[_collection_name(image_name)] += 1

        warnings: list[str] = []
        for json_name in json_names:
            if Path(json_name).name != "meta.jsonl":
                continue
            group_name = _collection_name(json_name)
            rows = _scan_metadata_rows(archive, json_name)
            metadata_rows_by_group[group_name].extend(rows)
            parse_errors = [row for row in rows if "_parse_error" in row]
            if parse_errors:
                warnings.append(f"{json_name} contains {len(parse_errors)} unparsable rows")

    group_names = sorted(set(image_counts) | set(metadata_rows_by_group))
    groups: list[DatasetGroupSummary] = []
    for group_name in group_names:
        rows = metadata_rows_by_group.get(group_name, [])
        altitudes = [
            altitude
            for altitude in (_safe_float(row.get("z")) for row in rows)
            if altitude is not None
        ]
        timestamps = [
            timestamp
            for timestamp in (_safe_float(row.get("t")) for row in rows)
            if timestamp is not None
        ]
        groups.append(
            DatasetGroupSummary(
                name=group_name,
                image_count=image_counts.get(group_name, 0),
                metadata_rows=len(rows),
                altitude_min=min(altitudes) if altitudes else None,
                altitude_max=max(altitudes) if altitudes else None,
                first_timestamp=min(timestamps) if timestamps else None,
                last_timestamp=max(timestamps) if timestamps else None,
            )
        )

    if not label_names:
        warnings.append("No YOLO label .txt files were found")
    if not yaml_names:
        warnings.append("No data.yaml or dataset.yaml file was found")

    return DatasetScanSummary(
        source_path=source_path,
        archive_name=source_path.name,
        total_files=len(names),
        total_images=len(image_names),
        total_yolo_labels=len(label_names),
        total_yaml_files=len(yaml_names),
        total_json_files=len(json_names),
        total_metadata_rows=sum(group.metadata_rows for group in groups),
        has_yolo_labels=bool(label_names),
        has_data_yaml=bool(yaml_names),
        groups=groups,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run scanner tests**

Run:

```bash
cd backend
python -m pytest tests/test_dataset_scanner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add backend/app/datasets backend/tests/test_dataset_scanner.py
git commit -m "feat: add dataset zip scanner"
```

## Task 4: Dataset Scan API

**Files:**
- Create: `backend/app/datasets/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_dataset_scan_api.py`

- [ ] **Step 1: Write failing API tests**

Create `backend/tests/test_dataset_scan_api.py`:

```python
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from app.main import app


def create_sample_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("yolo_dataset/vtol/images/raw/frame_vtol_z25m_00001.jpg", b"fake")
        archive.writestr(
            "yolo_dataset/vtol/meta.jsonl",
            '{"file":"frame_vtol_z25m_00001.jpg","drone":"vtol","z":25.5,"t":1.0}\\n',
        )


def test_scan_dataset_endpoint_returns_summary(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_sample_zip(zip_path)
    client = TestClient(app)

    response = client.post("/api/datasets/scan", json={"source_path": str(zip_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["archive_name"] == "sample.zip"
    assert payload["total_images"] == 1
    assert payload["groups"][0]["name"] == "vtol"
    assert "No YOLO label .txt files were found" in payload["warnings"]


def test_scan_dataset_endpoint_reports_missing_file(tmp_path: Path):
    client = TestClient(app)

    response = client.post(
        "/api/datasets/scan",
        json={"source_path": str(tmp_path / "missing.zip")},
    )

    assert response.status_code == 404
    assert "Dataset archive was not found" in response.json()["detail"]
```

- [ ] **Step 2: Run API tests to verify failure**

Run:

```bash
cd backend
python -m pytest tests/test_dataset_scan_api.py -v
```

Expected: FAIL with 404 for the missing `/api/datasets/scan` route.

- [ ] **Step 3: Add router and include it in the app**

Create `backend/app/datasets/router.py`:

```python
from fastapi import APIRouter, HTTPException

from app.datasets.scanner import scan_dataset_zip
from app.datasets.schemas import DatasetScanRequest, DatasetScanSummary


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/scan", response_model=DatasetScanSummary)
def scan_dataset(request: DatasetScanRequest) -> DatasetScanSummary:
    try:
        return scan_dataset_zip(request.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset archive was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

Modify `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.devices import detect_devices
from app.core.settings import get_settings
from app.datasets.router import router as datasets_router
from app.db.session import init_db


app = FastAPI(title="YOLO Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)


@app.on_event("startup")
def startup() -> None:
    init_db()


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
```

- [ ] **Step 4: Run backend tests**

Run:

```bash
cd backend
python -m pytest -v
```

Expected: PASS.

- [ ] **Step 5: Manually scan the provided dataset**

Run:

```bash
cd backend
python - <<'PY'
from pathlib import Path
from app.datasets.scanner import scan_dataset_zip

summary = scan_dataset_zip(Path("../image_dataset.zip"))
print(summary.total_images)
print([(group.name, group.image_count, group.metadata_rows) for group in summary.groups])
print(summary.has_yolo_labels, summary.has_data_yaml)
PY
```

Expected:

```text
1099
[('iris', 433, 433), ('vtol', 666, 666)]
False False
```

If the image count differs from 1099, inspect zip contents with `zipinfo -1 ../image_dataset.zip | tail` and update the scanner test only if the archive changed.

- [ ] **Step 6: Commit**

Run:

```bash
git add backend/app/main.py backend/app/datasets/router.py backend/tests/test_dataset_scan_api.py
git commit -m "feat: expose dataset scan API"
```

## Task 5: Frontend Scaffold And API Client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/App.test.tsx`

- [ ] **Step 1: Write the frontend test first**

Create `frontend/src/App.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  getHealth: async () => ({
    status: "ok",
    app: "YOLO Trainer",
    workspace_root: "/tmp/workspace",
    database_path: "/tmp/workspace/app.db",
    devices: { selected: "cpu", available: ["cpu"], details: {} },
  }),
  scanDataset: async () => ({
    source_path: "/tmp/image_dataset.zip",
    archive_name: "image_dataset.zip",
    total_files: 1102,
    total_images: 1099,
    total_yolo_labels: 0,
    total_yaml_files: 0,
    total_json_files: 2,
    total_metadata_rows: 1099,
    has_yolo_labels: false,
    has_data_yaml: false,
    groups: [
      {
        name: "iris",
        image_count: 433,
        metadata_rows: 433,
        altitude_min: 12,
        altitude_max: 13,
        first_timestamp: 1,
        last_timestamp: 2,
      },
    ],
    warnings: ["No YOLO label .txt files were found"],
  }),
}));

describe("App", () => {
  it("renders local app status and dataset scan controls", async () => {
    render(<App />);

    expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
    expect(await screen.findByText("cpu")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset zip path")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Scan Dataset" })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the frontend test to verify failure**

Run:

```bash
cd frontend
npm install
npm test -- --run
```

Expected: FAIL because the frontend package and app files are not complete yet.

- [ ] **Step 3: Add Vite scaffold, API client, and dashboard UI**

Create `frontend/package.json`:

```json
{
  "name": "yolo-trainer-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest",
    "preview": "vite preview"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "lucide-react": "^0.468.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.8",
    "@testing-library/react": "^16.0.0",
    "@testing-library/user-event": "^14.5.2",
    "jsdom": "^24.1.1",
    "typescript": "^5.5.4",
    "vite": "^5.4.0",
    "vitest": "^2.0.5"
  }
}
```

Create `frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>YOLO Trainer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

Create `frontend/vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

Create `frontend/src/main.tsx`:

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

Create `frontend/src/api.ts`:

```ts
export type DeviceInfo = {
  selected: "cuda" | "mps" | "cpu";
  available: string[];
  details: Record<string, string>;
};

export type HealthResponse = {
  status: string;
  app: string;
  workspace_root: string;
  database_path: string;
  devices: DeviceInfo;
};

export type DatasetGroupSummary = {
  name: string;
  image_count: number;
  metadata_rows: number;
  altitude_min: number | null;
  altitude_max: number | null;
  first_timestamp: number | null;
  last_timestamp: number | null;
};

export type DatasetScanSummary = {
  source_path: string;
  archive_name: string;
  total_files: number;
  total_images: number;
  total_yolo_labels: number;
  total_yaml_files: number;
  total_json_files: number;
  total_metadata_rows: number;
  has_yolo_labels: boolean;
  has_data_yaml: boolean;
  groups: DatasetGroupSummary[];
  warnings: string[];
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function scanDataset(sourcePath: string): Promise<DatasetScanSummary> {
  return requestJson<DatasetScanSummary>("/api/datasets/scan", {
    method: "POST",
    body: JSON.stringify({ source_path: sourcePath }),
  });
}
```

Create `frontend/src/App.tsx`:

```tsx
import { Activity, AlertTriangle, Database, FolderSearch, HardDrive } from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { DatasetScanSummary, HealthResponse, getHealth, scanDataset } from "./api";

const defaultDatasetPath = "~/DevProjects/YOLO_Trainer/image_dataset.zip";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  useEffect(() => {
    getHealth().then(setHealth).catch((error: Error) => setHealthError(error.message));
  }, []);

  const totalGroupImages = useMemo(
    () => scan?.groups.reduce((total, group) => total + group.image_count, 0) ?? 0,
    [scan],
  );

  async function handleScan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsScanning(true);
    setScanError(null);
    try {
      setScan(await scanDataset(datasetPath));
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "Dataset scan failed");
    } finally {
      setIsScanning(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Local Detection Workbench</p>
          <h1>YOLO Trainer</h1>
        </div>
        <div className="device-pill">
          <Activity size={16} />
          <span>{health?.devices.selected ?? "connecting"}</span>
        </div>
      </section>

      <section className="status-grid">
        <StatusTile
          icon={<HardDrive size={20} />}
          label="Workspace"
          value={health?.workspace_root ?? "Waiting for backend"}
        />
        <StatusTile
          icon={<Database size={20} />}
          label="Database"
          value={health?.database_path ?? "SQLite will initialize on startup"}
        />
        <StatusTile
          icon={<Activity size={20} />}
          label="Devices"
          value={health ? health.devices.available.join(", ") : "Detecting"}
        />
      </section>

      {healthError ? <div className="error-banner">{healthError}</div> : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Dataset Intake</p>
            <h2>Scan Local Zip</h2>
          </div>
          <FolderSearch size={22} />
        </div>

        <form className="scan-form" onSubmit={handleScan}>
          <label htmlFor="dataset-path">Dataset zip path</label>
          <div className="input-row">
            <input
              id="dataset-path"
              value={datasetPath}
              onChange={(event) => setDatasetPath(event.target.value)}
            />
            <button type="submit" disabled={isScanning || datasetPath.trim().length === 0}>
              {isScanning ? "Scanning" : "Scan Dataset"}
            </button>
          </div>
        </form>

        {scanError ? <div className="error-banner">{scanError}</div> : null}

        {scan ? (
          <div className="scan-results">
            <div className="metrics-row">
              <Metric label="Images" value={scan.total_images.toLocaleString()} />
              <Metric label="Metadata rows" value={scan.total_metadata_rows.toLocaleString()} />
              <Metric label="YOLO labels" value={scan.total_yolo_labels.toLocaleString()} />
              <Metric label="Class config" value={scan.has_data_yaml ? "Found" : "Missing"} />
            </div>

            <div className="group-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Images</th>
                    <th>Metadata</th>
                    <th>Altitude</th>
                  </tr>
                </thead>
                <tbody>
                  {scan.groups.map((group) => (
                    <tr key={group.name}>
                      <td>{group.name}</td>
                      <td>{group.image_count}</td>
                      <td>{group.metadata_rows}</td>
                      <td>
                        {group.altitude_min === null || group.altitude_max === null
                          ? "Not available"
                          : `${group.altitude_min.toFixed(1)}-${group.altitude_max.toFixed(1)}m`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="summary-line">
              Grouped images: {totalGroupImages.toLocaleString()} from {scan.archive_name}
            </div>

            {scan.warnings.length > 0 ? (
              <div className="warnings">
                <AlertTriangle size={18} />
                <div>
                  {scan.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </main>
  );
}

function StatusTile(props: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="status-tile">
      {props.icon}
      <div>
        <span>{props.label}</span>
        <strong>{props.value}</strong>
      </div>
    </div>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}
```

Create `frontend/src/styles.css`:

```css
:root {
  color: #172026;
  background: #eef2f5;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
}

button,
input {
  font: inherit;
}

.app-shell {
  width: min(1180px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.topbar,
.panel,
.status-tile {
  border: 1px solid #d8e0e7;
  background: #ffffff;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 24px;
  border-radius: 8px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #5f6f7a;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2 {
  margin: 0;
}

h1 {
  font-size: 32px;
  line-height: 1.1;
}

h2 {
  font-size: 22px;
}

.device-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 0 12px;
  border-radius: 18px;
  color: #0b5f6a;
  background: #dff6f2;
  font-weight: 700;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}

.status-tile {
  display: flex;
  gap: 12px;
  min-width: 0;
  padding: 16px;
  border-radius: 8px;
}

.status-tile span,
.metric span,
.scan-form label {
  display: block;
  color: #62727e;
  font-size: 13px;
  font-weight: 700;
}

.status-tile strong {
  display: block;
  min-width: 0;
  margin-top: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.panel {
  padding: 22px;
  border-radius: 8px;
}

.panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 20px;
}

.scan-form {
  display: grid;
  gap: 8px;
}

.input-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
}

input {
  width: 100%;
  min-height: 42px;
  border: 1px solid #c7d2da;
  border-radius: 6px;
  padding: 0 12px;
  color: #172026;
  background: #fbfdff;
}

button {
  min-height: 42px;
  border: 0;
  border-radius: 6px;
  padding: 0 16px;
  color: #ffffff;
  background: #1f6f78;
  font-weight: 800;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.error-banner,
.warnings {
  margin-top: 14px;
  border-radius: 8px;
  padding: 12px 14px;
  color: #7c2d12;
  background: #fff1e7;
}

.scan-results {
  margin-top: 20px;
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.metric {
  min-width: 0;
  border: 1px solid #d9e3ea;
  border-radius: 8px;
  padding: 14px;
  background: #f8fafb;
}

.metric strong {
  display: block;
  margin-top: 6px;
  font-size: 24px;
}

.group-table-wrap {
  margin-top: 16px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 560px;
}

th,
td {
  border-bottom: 1px solid #dfe7ec;
  padding: 12px 10px;
  text-align: left;
}

th {
  color: #60707c;
  font-size: 13px;
}

.summary-line {
  margin-top: 12px;
  color: #54636f;
  font-weight: 700;
}

.warnings {
  display: flex;
  gap: 10px;
}

.warnings p {
  margin: 0 0 4px;
}

@media (max-width: 820px) {
  .topbar,
  .input-row {
    grid-template-columns: 1fr;
  }

  .topbar {
    align-items: flex-start;
  }

  .status-grid,
  .metrics-row {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 4: Run frontend tests and build**

Run:

```bash
cd frontend
npm install
npm test -- --run
npm run build
```

Expected: PASS for tests and successful Vite production build.

- [ ] **Step 5: Commit**

Run:

```bash
git add frontend
git commit -m "feat: scaffold frontend dataset scan UI"
```

## Task 6: End-To-End Local Smoke Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with smoke verification steps**

Modify `README.md` so it contains:

```markdown
# YOLO Trainer

Local single-user YOLO training workbench.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -v
python -m uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Dataset scan check:

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/scan \
  -H "Content-Type: application/json" \
  -d '{"source_path":"~/DevProjects/YOLO_Trainer/image_dataset.zip"}'
```

Expected scan facts for the provided archive:

- `total_images` is `1099`.
- `groups` contains `iris` and `vtol`.
- `has_yolo_labels` is `false`.
- `has_data_yaml` is `false`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, keep the backend running, and scan:

```text
~/DevProjects/YOLO_Trainer/image_dataset.zip
```
```

- [ ] **Step 2: Run all automated checks**

Run:

```bash
cd backend
python -m pytest -v
cd ../frontend
npm test -- --run
npm run build
```

Expected: all backend tests pass, frontend tests pass, and frontend build succeeds.

- [ ] **Step 3: Run backend and verify the provided archive through HTTP**

Run:

```bash
cd backend
python -m uvicorn app.main:app --port 8000
```

In another terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/datasets/scan \
  -H "Content-Type: application/json" \
  -d '{"source_path":"~/DevProjects/YOLO_Trainer/image_dataset.zip"}'
```

Expected response includes:

```json
{
  "archive_name": "image_dataset.zip",
  "total_images": 1099,
  "has_yolo_labels": false,
  "has_data_yaml": false
}
```

- [ ] **Step 4: Run frontend and verify the browser page**

Run:

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`, press `Scan Dataset`, and verify:

- Device pill shows `cuda`, `mps`, or `cpu`.
- Image count shows `1,099`.
- Group table shows `iris` and `vtol`.
- Warning panel states that YOLO labels and `data.yaml` are missing.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md
git commit -m "docs: add local smoke verification"
```

## Self-Review Checklist

- Spec coverage for this slice: project shell, backend health, workspace settings, SQLite foundation, device detection, dataset scanning, and frontend preview are covered.
- The plan intentionally leaves annotator, class library, dataset versioning, training, prediction analysis, and export for follow-on plans because each is independently testable and larger than the scaffold slice.
- Type consistency: `DatasetScanSummary`, `DatasetGroupSummary`, `HealthResponse`, and device fields use the same property names in backend schemas, API tests, frontend types, and UI.
- Verification: backend tests, frontend tests, frontend build, HTTP scan, and browser smoke check are included.
