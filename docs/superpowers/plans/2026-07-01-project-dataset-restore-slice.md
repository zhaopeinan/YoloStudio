# Project Dataset Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local user reload existing projects and datasets after refreshing or restarting the app without re-importing source data.

**Architecture:** Add a compact project/dataset listing endpoint that returns projects with their imported datasets. The frontend loads this list on startup and after import, renders a dataset selector in Dataset Intake, and reuses the existing post-import hydration flow to load classes, images, quality, versions, and run history.

**Tech Stack:** FastAPI, SQLAlchemy, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Project Dataset Listing

**Files:**
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [x] **Step 1: Add schemas**

Add `ProjectDatasetRead`, `ProjectRead`, and `ProjectList` schemas with project ID/name and dataset ID/name/source type/status/image count.

- [x] **Step 2: Add endpoint**

Expose `GET /api/projects` from the datasets router module, returning projects ordered by ID and each project's datasets ordered by ID.

- [x] **Step 3: Test listing**

Import at least one dataset and assert `/api/projects` returns the project and dataset summary.

### Task 2: Frontend Restore Flow

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add API client**

Add `ProjectSummary`, `ProjectDatasetSummary`, and `listProjects()`.

- [x] **Step 2: Share dataset hydration**

Extract the existing post-import loading code into a helper that accepts a `DatasetImportResponse`-shaped object and loads classes, images, quality, versions, runs, predictions, summary, and exports.

- [x] **Step 3: Render restore controls**

Render a `Saved dataset` select and `Load Dataset` button. Disable load when no saved dataset is selected.

- [x] **Step 4: Test restore**

Update App tests to mock a saved dataset list, assert controls render, and assert clicking `Load Dataset` hydrates the workbench without calling `importDataset`.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document restore**

Mention that the Dataset Intake panel can load already imported datasets after app restart.

- [x] **Step 2: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

Stage only this slice and the plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: add saved dataset restore"
```
