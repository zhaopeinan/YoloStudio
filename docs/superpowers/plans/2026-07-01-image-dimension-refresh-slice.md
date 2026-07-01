# Image Dimension Refresh Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators repair old imported datasets whose images have missing width or height metadata after image-dimension parsing improves.

**Architecture:** Add a dataset maintenance endpoint that scans existing workspace image files and updates missing `images.width` / `images.height` values. Reuse the importer image-size parser so import-time and repair-time behavior stays identical.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TypeScript, Vitest.

---

### Task 1: Backend Refresh Endpoint

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [ ] Expose a safe helper for reading image dimensions from an existing workspace file.
- [ ] Add `DatasetDimensionRefreshSummary` with `dataset_id`, `scanned_count`, `updated_count`, and `missing_count`.
- [ ] Add `POST /api/datasets/{dataset_id}/refresh-image-dimensions`.
- [ ] Only update rows whose width or height is missing and whose file path resolves under `settings.workspace_root`.
- [ ] Return 404 for missing datasets.
- [ ] Prove a legacy-style null dimension row can be repaired and then appears with width and height in `/images`.

### Task 2: Frontend Repair Action

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] Add `DatasetDimensionRefreshSummary` and `refreshImageDimensions(datasetId)`.
- [ ] Add a Quality Review action button labeled `Refresh Dimensions`.
- [ ] Show a concise result message with scanned, updated, and still-missing counts.
- [ ] After refresh, reload image list and quality data so the `Missing dimensions` metric updates.
- [ ] Cover the button flow in Vitest.

### Task 3: Verification

**Commands:**
- `cd backend && python -m pytest -q tests/test_dataset_import_api.py`
- `cd backend && python -m pytest -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: refresh image dimensions`.
