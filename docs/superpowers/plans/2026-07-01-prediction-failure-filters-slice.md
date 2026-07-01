# Prediction Failure Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users filter prediction and failure samples by failure type, class, confidence, platform, altitude, and time so review can target hard cases before annotation correction.

**Architecture:** Extend the existing prediction list endpoint with optional query parameters and keep the response shape unchanged for compatibility. The frontend stores a small filter state, requests filtered rows from the backend, and keeps the existing `Open Image` correction flow.

**Tech Stack:** FastAPI query parameters, SQLAlchemy joins, React controlled inputs, Vitest.

---

### Task 1: Backend Prediction Filters

**Files:**
- Modify: `backend/app/prediction/router.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] **Step 1: Add query parameters**

Update `GET /api/prediction-jobs/{job_id}/predictions` to accept:
- `failure_type`: `all|matched|false_positive|false_negative`, default `all`
- `class_id`: optional integer
- `confidence_min`: optional float `0..1`
- `confidence_max`: optional float `0..1`
- `platform`: optional string
- `altitude_min`: optional float
- `altitude_max`: optional float
- `timestamp_min`: optional float
- `timestamp_max`: optional float

- [ ] **Step 2: Add filtering query**

Build a `select(Prediction)` query. Join `Image` only when image metadata filters are present. Apply query conditions and keep `order_by(Prediction.id)`.

- [ ] **Step 3: Test filters**

Extend `test_prediction_api.py` to call the prediction list endpoint with `failure_type=false_positive`, `class_id`, confidence bounds, and `platform=iris`. Assert only the expected filtered prediction rows are returned.

- [ ] **Step 4: Run focused backend test**

Run: `cd backend && python -m pytest tests/test_prediction_api.py -v`

Expected: prediction API tests pass.

### Task 2: Frontend Prediction Filter Controls

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add filter type**

Add `PredictionFilters` to `api.ts` and update `listPredictions(jobId, filters = {})` to serialize non-empty filters into query parameters.

- [ ] **Step 2: Add state and refresh helper**

Add `predictionFilters` state with default failure type `all` and empty string fields. Update prediction loading after job creation, threshold scan, and job refresh to call `listPredictions(jobId, toPredictionFilterRequest(predictionFilters))`.

- [ ] **Step 3: Add UI controls**

In `Prediction Analysis`, add a compact `Prediction sample filters` row with failure type select, class select, confidence min/max, platform, altitude min/max, timestamp min/max, `Apply` and `Reset` buttons.

- [ ] **Step 4: Cover UI behavior**

Update `App.test.tsx` to set failure type to `false_positive`, class to `target`, and confidence minimum to `0.5`, click `Apply`, and assert `listPredictions` receives the filter request.

- [ ] **Step 5: Run focused frontend test**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: App tests pass.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document filtered failure mining**

Add README bullets for filtering prediction samples by failure type, class, confidence, platform, altitude, and timestamp.

- [ ] **Step 2: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

Stage only project changes and this plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: add prediction failure filters"
```
