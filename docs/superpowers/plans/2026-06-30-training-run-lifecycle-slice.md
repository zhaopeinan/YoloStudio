# Training Run Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first trainable experiment slice: persistent training runs, one-active-run guard, Ultralytics adapter boundary, logs/metrics artifacts, backend APIs, and frontend training monitor.

**Architecture:** The backend owns training run state and writes run artifacts under the managed workspace. A small runner service starts a background task, records status transitions, and delegates real training to an adapter that can fail clearly when Ultralytics is unavailable. The frontend provides a training setup panel and a compact experiment monitor tied to dataset versions.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, background tasks, optional Ultralytics YOLO, pytest, React/Vite/TypeScript/Vitest.

---

## Scope

In scope:

- `TrainingRun` model with project/version IDs, status, device, config JSON, artifact path, started/ended time, error message, and log path.
- `RunMetric` model for epoch/metric/value rows.
- APIs to create a run, list project runs, read one run, and read logs.
- One active run guard for queued/preparing/running states.
- Deterministic run artifact directory: `workspace/projects/<project_id>/runs/<run_id>/`.
- `config.json`, `logs.txt`, and `metrics.jsonl` artifacts.
- Optional Ultralytics adapter boundary. If `ultralytics` is not installed, the run fails with a clear message.
- Frontend panel for model preset, epochs, image size, batch size, device, and run history.

Out of scope for this slice:

- Full curve plotting.
- Prediction/failure mining after training.
- ONNX/TensorRT export.
- Cancellation.
- Hyperparameter sweeps.

## Tasks

### Task 1: Backend training API

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/app/training/__init__.py`
- Create: `backend/app/training/schemas.py`
- Create: `backend/app/training/runner.py`
- Create: `backend/app/training/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_training_api.py`

- [ ] Write tests that create a dataset version, start a training run, assert queued/preparing/running/completed/failed state is persisted, list runs, and read logs.
- [ ] Add `TrainingRun` and `RunMetric` models.
- [ ] Implement request/response schemas.
- [ ] Implement one-active-run guard and artifact creation.
- [ ] Implement the adapter boundary so tests can run a fake trainer without requiring Ultralytics.
- [ ] Include the router in FastAPI.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Commit `feat: add training run lifecycle API`.

### Task 2: Frontend training monitor

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add API types and calls for create/list/read run/logs.
- [ ] Add training setup panel below version export.
- [ ] Disable start until a dataset version exists.
- [ ] Show latest run status, artifact path, config, and logs preview.
- [ ] Refresh run history after creating a run.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
- [ ] Commit `feat: add training setup and run monitor UI`.

### Task 3: Documentation and smoke

**Files:**
- Modify: `README.md`

- [ ] Document training lifecycle behavior and optional Ultralytics dependency.
- [ ] Run backend and frontend verification.
- [ ] API smoke: create a run from an exported version and confirm status/logs.
- [ ] Commit `docs: add training run smoke steps`.

## Self-Review Checklist

- This slice moves the MVP toward real training without claiming dashboard/prediction/export completion.
- Training runs persist after restart because state is in SQLite and artifacts are on disk.
- If Ultralytics is unavailable, users see a run-level failed status with the reason.
- Existing dataset import, annotation, quality, and version export workflows remain intact.
