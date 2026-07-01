# Threshold Scan Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one-click confidence threshold scanning so a completed run can produce multiple prediction jobs and feed the existing experiment dashboard.

**Architecture:** Reuse prediction jobs as the durable unit of work. The backend accepts an explicit threshold list, executes one existing prediction job per threshold, and returns the created jobs. The frontend parses a compact comma-separated threshold list, submits the scan, loads the newest job's predictions, and refreshes the existing summary.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Vitest, Testing Library.

---

### Task 1: Backend API

**Files:**
- Modify: `backend/app/prediction/schemas.py`
- Modify: `backend/app/prediction/router.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] **Step 1: Add threshold scan schemas**

Add `PredictionThresholdScanCreate` with `image_scope` and `thresholds`, using Pydantic bounds to keep values from `0` through `1` and a list length from `1` through `20`. Add `PredictionThresholdScanRead` with `items: list[PredictionJobRead]`.

- [ ] **Step 2: Add endpoint**

Add `POST /api/training/runs/{run_id}/prediction-threshold-scan`. Load the `TrainingRun`, return 404 when absent, loop through `request.thresholds`, create each prediction job with the existing `create_prediction_job`, execute it with `execute_prediction_job(..., predictor=predict_images)`, and return `_read_job` for each job.

- [ ] **Step 3: Cover API behavior**

Extend `test_prediction_api.py` with a test that monkeypatches `app.prediction.router.predict_images`, posts thresholds `[0.15, 0.25, 0.5]`, asserts three completed jobs are returned with matching thresholds, asserts the run job list contains the newest scan job first, and asserts `/api/training/runs/{run_id}/summary` includes threshold points for all three thresholds.

- [ ] **Step 4: Run focused backend test**

Run: `cd backend && python -m pytest tests/test_prediction_api.py -v`

Expected: all prediction API tests pass.

### Task 2: Frontend API And Controls

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add API function and types**

Add `PredictionThresholdScanResponse` with `items: PredictionJob[]`. Add `createPredictionThresholdScan(runId, body)` posting `{ image_scope, thresholds }` to `/api/training/runs/${runId}/prediction-threshold-scan`.

- [ ] **Step 2: Add scan state and parser**

Add state for a threshold input string, defaulting to `0.15, 0.25, 0.35, 0.5, 0.65`, and a separate loading flag. Add a parser that splits commas, trims spaces, converts values to numbers, rejects empty input, non-numeric values, values outside `0..1`, and more than 20 thresholds.

- [ ] **Step 3: Add UI controls**

In `Prediction Analysis`, keep the single threshold job controls and add a second compact row for the threshold scan input plus a `Run Threshold Scan` button. Disable it while no run exists or while the scan is running.

- [ ] **Step 4: Wire scan handler**

The handler should parse thresholds, call `createPredictionThresholdScan`, merge returned jobs into `predictionJobs`, load predictions and logs for the newest returned job, refresh the run summary, and show parse/API errors in the existing prediction error banner.

- [ ] **Step 5: Cover UI behavior**

Extend `App.test.tsx` with a mocked `createPredictionThresholdScan`, click `Run Threshold Scan`, assert the API receives parsed threshold numbers, and assert the returned scan job appears in the prediction job list.

- [ ] **Step 6: Run focused frontend test**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: App test passes.

### Task 3: Docs, Full Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document threshold scan workflow**

Add README steps that explain comma-separated threshold scan input, the new endpoint, and that scan jobs automatically populate the experiment dashboard threshold table.

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

Stage only tracked project changes and the new plan document, preserving `image_dataset.zip` as untracked. Commit with:

```bash
git commit -m "feat: add threshold scan automation"
```
