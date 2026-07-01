# Dataset Filter And Class Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dataset browser filters and dataset-version class subset cropping so operators can build focused YOLO training snapshots from the project class library.

**Architecture:** Extend the existing image list endpoint with query filters over image metadata and annotation metadata. Extend dataset version creation with an optional `class_ids` list that filters exported images/labels and freezes only selected classes.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, pytest, React/Vite/TypeScript/Vitest.

---

## Tasks

### Task 1: Backend Image Filters

**Files:**
- Modify: `backend/app/datasets/router.py`
- Modify: `backend/tests/test_dataset_import_api.py`

- [ ] Add query parameters for platform, label status, class ID, edge tag, and altitude range.
- [ ] Keep pagination and total count aligned with the filtered query.
- [ ] Return annotated, unannotated, class-filtered, tag-filtered, and altitude-filtered image lists.

### Task 2: Backend Dataset Version Class Subset

**Files:**
- Modify: `backend/app/versions/schemas.py`
- Modify: `backend/app/versions/router.py`
- Modify: `backend/app/versions/exporter.py`
- Modify: `backend/tests/test_quality_version_api.py`

- [ ] Accept optional `class_ids` on version creation.
- [ ] Validate class IDs belong to the project and are active.
- [ ] Export only images with selected-class annotations when `class_ids` is supplied.
- [ ] Write YOLO labels using a compact frozen mapping for selected classes only.
- [ ] Persist the selected class IDs in `manifest.json`.

### Task 3: Frontend Dataset Filter And Version Controls

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add filter request type and query string builder for `listImages`.
- [ ] Add image browser filter controls.
- [ ] Reload images when filters apply.
- [ ] Add class subset checkboxes for version creation.
- [ ] Pass selected class IDs to the version creation API.

### Task 4: Verification And Docs

**Files:**
- Modify: `README.md`

- [ ] Document image filters and class subset version export.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
