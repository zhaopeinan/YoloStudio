# Model Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add run-level `.pt`, ONNX, and conditional TensorRT export workflow so completed training runs can move toward deployment.

**Architecture:** Persist export attempts as `ExportArtifact` rows linked to `TrainingRun`. The backend exposes capability, create, and list endpoints; `.pt` export registers the best weights path, ONNX calls an adapter that can use Ultralytics when installed, and TensorRT is blocked unless capability detection says it is available. The frontend adds an export panel driven by the latest run.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, optional Ultralytics/TensorRT, pytest, React/Vite/TypeScript/Vitest.

---

## Tasks

### Task 1: Backend Export API

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/app/exports/__init__.py`
- Create: `backend/app/exports/schemas.py`
- Create: `backend/app/exports/runner.py`
- Create: `backend/app/exports/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_exports_api.py`

- [x] Add `ExportArtifact` with run/project IDs, format, status, artifact path, error, and metadata.
- [x] Add capability endpoint returning ONNX availability and TensorRT support status.
- [x] Add create/list endpoints for run exports.
- [x] Register `.pt` export from `ultralytics/weights/best.pt`.
- [x] Implement ONNX adapter with clear failure when weights or Ultralytics are missing.
- [x] Block TensorRT export when TensorRT is unsupported.

### Task 2: Frontend Export Panel

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [x] Add export artifact and capability API types.
- [x] Add run-level export panel below prediction analysis.
- [x] Show `.pt`, ONNX, and TensorRT support states.
- [x] Create exports and list persisted export artifacts.
- [x] Surface export errors without breaking run history.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`

- [x] Document export behavior and optional dependencies.
- [x] Run `cd backend && python -m pytest -v`.
- [x] Run `cd frontend && npm test -- --run && npm run build`.
