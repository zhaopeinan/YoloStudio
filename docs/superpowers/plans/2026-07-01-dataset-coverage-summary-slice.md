# Dataset Coverage Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show dataset diversity and edge-case coverage so operators can see whether the training set spans platforms, altitude bands, classes, and hard-case tags before exporting or retraining.

**Architecture:** Add a read-only dataset coverage endpoint under the existing dataset API. The frontend loads it with quality data and renders compact coverage cards inside Training Prep.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Coverage Summary

**Files:**
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [x] **Step 1: Add failing API test**

Import the sample dataset, add two classes, annotate both images with different classes and edge tags, call `GET /api/datasets/{dataset_id}/coverage`, and assert total image/annotation counts plus platform, altitude band, class, and edge-tag rows.

- [x] **Step 2: Add schemas**

Add `CoverageBucket`, `ClassCoverageBucket`, `EdgeTagCoverageBucket`, and `DatasetCoverageSummary` Pydantic models.

- [x] **Step 3: Implement aggregator route**

Expose `GET /api/datasets/{dataset_id}/coverage`, verify the dataset exists, aggregate platform rows from images, altitude bands from image altitude, class rows from annotations joined to classes/images, and edge-tag rows from annotation JSON tags.

### Task 2: Frontend Coverage Panel

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add API types/client**

Add `DatasetCoverageSummary` and `getDatasetCoverage(datasetId)`.

- [x] **Step 2: Load coverage with dataset workspace**

Fetch coverage whenever a dataset is loaded/imported, after annotation saves, and after quality refresh paths that can change annotations.

- [x] **Step 3: Render coverage**

Add a compact `Dataset Coverage` panel in Training Prep showing top platforms, altitude bands, class annotation counts, and edge tags. Show an empty state before dataset load.

- [x] **Step 4: Add frontend coverage**

Extend the main app test to assert the coverage panel renders and that the coverage API is refreshed for the active dataset.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document coverage panel**

Mention the Dataset Coverage panel in the quality/training-prep docs.

- [x] **Step 2: Verify**

Run:

```bash
cd backend && python -m pytest tests/test_dataset_import_api.py::test_dataset_coverage_summary_reports_diversity -v
cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [x] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
