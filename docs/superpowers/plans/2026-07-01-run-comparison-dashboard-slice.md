# Run Comparison Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact project-level comparison table so operators can compare recent YOLO training runs without opening each run one at a time.

**Architecture:** Add a read-only project experiment comparison endpoint that summarizes each run's status, config, latest metrics, latest prediction job counts, and best threshold point. The frontend loads it alongside run history and renders a dense table in the Experiment Dashboard area.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Comparison Summary

**Files:**
- Modify: `backend/app/experiments/schemas.py`
- Modify: `backend/app/experiments/summary.py`
- Modify: `backend/app/training/router.py`
- Test: `backend/tests/test_experiment_summary_api.py`

- [x] **Step 1: Add failing API test**

Create two completed runs with different metrics/prediction jobs, call `GET /api/projects/{project_id}/training/summary`, and assert rows are ordered by newest run first with mAP, loss, latest prediction counts, and best threshold.

- [x] **Step 2: Add schemas**

Add `RunComparisonRow` and `ProjectExperimentSummary` Pydantic models.

- [x] **Step 3: Implement aggregator**

Add `build_project_experiment_summary(db, project_id)` that reads recent runs, latest metrics, latest completed prediction jobs, and threshold recommendations.

- [x] **Step 4: Add route**

Expose the aggregator at `GET /api/projects/{project_id}/training/summary`.

### Task 2: Frontend Comparison Table

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add API client/types**

Add `ProjectExperimentSummary` and `getProjectTrainingSummary(projectId)`.

- [x] **Step 2: Load summary with runs**

Fetch project summary when loading/refreshing runs and clear it when no dataset/run exists.

- [x] **Step 3: Render comparison table**

Render run ID, status, model, epochs, mAP50, box loss, prediction outcome counts, best threshold, F1, and artifact path.

- [x] **Step 4: Add frontend coverage**

Extend the dashboard test to assert the comparison table renders and refreshes from the summary mock.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document comparison dashboard**

Mention the Run Comparison table in the experiment dashboard docs.

- [x] **Step 2: Verify**

Run:

```bash
cd backend && python -m pytest tests/test_experiment_summary_api.py -v
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [x] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
