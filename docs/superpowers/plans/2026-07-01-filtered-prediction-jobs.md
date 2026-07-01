# Filtered Prediction Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow prediction analysis jobs to run on the current filtered image set, not only on `all`, `train`, `val`, or `test` version splits.

**Architecture:** Add a typed `image_filters` object to prediction job creation and threshold scan requests. The backend starts from the frozen DatasetVersion split manifest, applies the same image filter semantics used by the Image Browser, records the applied filter config in `predictions.json`, and the frontend exposes a single “Use image filters” switch in the Prediction Analysis controls.

**Tech Stack:** FastAPI, Pydantic, SQLAlchemy, SQLite JSON functions, React, TypeScript, Vitest.

---

### Task 1: Backend Filter Schema And Selection

**Files:**
- Modify: `backend/app/prediction/schemas.py`
- Modify: `backend/app/prediction/router.py`
- Modify: `backend/app/prediction/runner.py`
- Test: `backend/tests/test_prediction_api.py`

- [x] **Step 1: Add failing backend test**

Add a test that creates a completed run, starts a prediction job with `image_filters: {"platform": <first image platform>}`, and asserts that the predictor receives only matching images, the job `image_count` reflects that subset, and `predictions.json` includes the filter config.

- [x] **Step 2: Implement schemas**

Add `PredictionImageFilters` with fields `platform`, `label_status`, `class_id`, `edge_tag`, `failure_type`, `altitude_min`, and `altitude_max`. Add `image_filters: PredictionImageFilters | None` to `PredictionJobCreate` and `PredictionThresholdScanCreate`.

- [x] **Step 3: Implement runner filtering**

Change `list_job_images(db, run, image_scope, image_filters=None)` so it selects manifest image ids, then applies optional filters with SQLAlchemy `exists()` clauses for annotation status, class membership, edge tags, and prior prediction failure type.

- [x] **Step 4: Thread filters through router and artifacts**

Pass `request.image_filters` into `create_prediction_job` and `execute_prediction_job`. Store the normalized config on the in-memory job object, write it to `predictions.json` next to the prediction rows, and log the filter summary.

### Task 2: Frontend Controls And API

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add failing frontend test**

Set an Image Browser filter, enable “Use image filters”, run prediction analysis, and assert `createPredictionJob` receives `{ image_scope, confidence_threshold, image_filters }`.

- [x] **Step 2: Update API types**

Add `image_filters?: DatasetImageFilters` to prediction job and threshold scan request bodies.

- [x] **Step 3: Add UI state and controls**

Add `useImageFiltersForPrediction` state and a checkbox in Prediction Analysis. Show a concise active filter summary when enabled.

- [x] **Step 4: Send filters on job creation**

Use `toImageFilterRequest(imageFilters)` when the switch is enabled; omit `image_filters` when no filters are active.

### Task 3: Verification And Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document behavior**

Add a README note that prediction analysis can run on Image Browser filters.

- [x] **Step 2: Run focused tests**

Run:

```bash
cd backend && python -m pytest tests/test_prediction_api.py -v
cd frontend && npm test -- --run
```

- [x] **Step 3: Run full gates**

Run:

```bash
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 4: Commit**

Commit all tracked changes for this slice, excluding `image_dataset.zip`.
