# Prediction Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first prediction-analysis loop: create prediction jobs for a training run, persist predictions, classify false positives/false negatives/matches, and surface failure samples in the UI.

**Architecture:** Backend prediction jobs run against a training run's frozen dataset version and write artifacts under the run directory. The prediction adapter can call Ultralytics when available, while tests inject a fake predictor. The frontend adds a compact prediction panel to the experiment area and links failure samples back to image IDs for later annotator integration.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, optional Ultralytics YOLO, pytest, React/Vite/TypeScript/Vitest.

---

## Scope

In scope:

- `PredictionJob` model with run ID, status, image scope, confidence threshold, artifact path, log path, counts, and error.
- `Prediction` model with run ID, job ID, image ID, class ID, bbox, confidence, matched annotation ID, and failure type.
- API to start a prediction job, list jobs, list predictions, and read prediction logs.
- Basic IoU matching against ground-truth annotations.
- Failure classifications: `matched`, `false_positive`, `false_negative`.
- Frontend panel for confidence threshold, image scope, job history, counts, failure samples, and logs.

Out of scope for this slice:

- Overlaying predictions directly on the annotation canvas.
- Full PR/F1 curve computation.
- Confusion matrix plots.
- ONNX/TensorRT export.

## Tasks

### Task 1: Backend prediction API

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/app/prediction/__init__.py`
- Create: `backend/app/prediction/schemas.py`
- Create: `backend/app/prediction/matching.py`
- Create: `backend/app/prediction/runner.py`
- Create: `backend/app/prediction/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_prediction_api.py`

- [ ] Write tests that create a version/run, inject fake predictions, start a job, assert job counts, predictions, failures, and logs.
- [ ] Add prediction models.
- [ ] Implement IoU matching and failure classification.
- [ ] Implement prediction job runner with optional Ultralytics adapter.
- [ ] Include router in FastAPI.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Commit `feat: add prediction analysis API`.

### Task 2: Frontend prediction panel

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add API types and calls for prediction jobs/results/logs.
- [ ] Add panel near Run History.
- [ ] Enable prediction once a run exists.
- [ ] Display counts and failure samples.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
- [ ] Commit `feat: add prediction analysis UI`.

### Task 3: Documentation and smoke

**Files:**
- Modify: `README.md`

- [ ] Document prediction job behavior and optional Ultralytics dependency.
- [ ] Run backend/frontend verification.
- [ ] API smoke on a real run; expect clear failed state if no weights/Ultralytics exist.
- [ ] Commit `docs: add prediction analysis smoke steps`.

## Self-Review Checklist

- This slice moves toward the requested post-training model effect review.
- It does not claim full dashboard, plots, or export completion.
- Failed prediction jobs are persisted and explain why they failed.
- Failure rows retain image IDs so the next slice can jump from a failure sample into annotation.
