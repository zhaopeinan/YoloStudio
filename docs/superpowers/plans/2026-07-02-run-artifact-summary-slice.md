# Run Artifact Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface generated run files in the UI so operators can confirm configs, logs, metrics, weights, plots, predictions, and exports without leaving the workbench.

**Architecture:** Add a read-only training run artifact summary endpoint that scans only inside the run artifact directory. The frontend loads the summary with run history and displays a compact artifact panel per run.

**Tech Stack:** FastAPI, pathlib, Pydantic, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Run Artifact Summary

**Files:**
- Modify: `backend/app/training/schemas.py`
- Modify: `backend/app/training/router.py`
- Test: `backend/tests/test_training_api.py`

- [x] **Step 1: Add failing API test**

Create a completed run, write representative files under its artifact root (`config.json`, `logs.txt`, `metrics.jsonl`, `ultralytics/weights/best.pt`, `ultralytics/results.png`, `exports/run-1.pt`), call `GET /api/training/runs/{run_id}/artifacts`, and assert category, relative path, size, and count data.

- [x] **Step 2: Add schemas**

Add `TrainingRunArtifact`, `TrainingRunArtifactSummary`, and fields for `run_id`, `artifact_root`, `items`, and `total_count`.

- [x] **Step 3: Implement safe scanner route**

Expose `GET /api/training/runs/{run_id}/artifacts`. Verify the run exists, resolve the artifact root, ignore missing roots, skip directories, cap returned files to a stable small limit, categorize known paths, and return relative POSIX paths plus byte size.

### Task 2: Frontend Artifact Panel

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add API client/types**

Add `TrainingRunArtifact`, `TrainingRunArtifactSummary`, and `getTrainingRunArtifacts(runId)`.

- [x] **Step 2: Load artifacts with runs**

Fetch artifact summaries for visible run history rows and refresh them after run history refreshes.

- [x] **Step 3: Render artifacts**

Render a dense artifact list inside each run row, showing category, relative path, and formatted file size. Keep empty state compact.

- [x] **Step 4: Add frontend coverage**

Extend the main integration test to assert artifact summaries load and render for Run #1.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document run artifact panel**

Mention the run artifact summary endpoint and UI panel in the training lifecycle docs.

- [x] **Step 2: Verify**

Run:

```bash
cd backend && python -m pytest tests/test_training_api.py::test_training_run_artifacts_lists_generated_files -v
cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
