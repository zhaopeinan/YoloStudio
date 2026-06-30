# Quality Review And Dataset Version Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the training-prep slice: dataset quality checks, deterministic split preview, frozen dataset version creation, YOLO label export, and `data.yaml` generation.

**Architecture:** Backend owns validation and version materialization under the managed workspace. Frontend surfaces actionable quality counts and a version creation button tied to the imported dataset/class/annotation state.

**Tech Stack:** FastAPI, SQLAlchemy, PyYAML or standard YAML string generation, pytest, React/Vite/TypeScript/Vitest.

---

## Scope

In scope:

- Dataset quality summary for a dataset.
- Checks for unannotated images, empty classes, invalid annotation geometry, tiny boxes, and unknown class references.
- Deterministic split assignment for annotated images.
- Dataset version model and version artifact directory.
- YOLO labels and `data.yaml` export.
- Frontend quality panel and create-version button.

Out of scope:

- Training execution.
- Full quality issue editing workflow.
- Advanced stratification beyond a deterministic class-aware best effort.

## Backend Requirements

Add models:

- `DatasetVersion`: project ID, dataset ID, name, class mapping JSON, split manifest JSON, artifact path, frozen flag.

Add endpoints:

- `GET /api/datasets/{dataset_id}/quality`
- `POST /api/datasets/{dataset_id}/versions`
- `GET /api/datasets/{dataset_id}/versions`

Quality response includes:

- `image_count`
- `annotated_image_count`
- `unannotated_image_count`
- `annotation_count`
- `class_count`
- `tiny_box_count`
- `invalid_box_count`
- `ready_for_training`
- `issues`

Version creation behavior:

- Reject if no classes exist, no annotations exist, or invalid boxes exist.
- Freeze active project classes ordered by ID into zero-based YOLO indices.
- Include only images with at least one annotation in splits.
- Deterministically split annotated images: 80% train, 10% val, 10% test, with at least one val image when there are at least two annotated images.
- Create `workspace/projects/<project_id>/versions/<version_id>/`.
- Copy or hard-link images into `images/train`, `images/val`, `images/test`.
- Write corresponding labels into `labels/train`, `labels/val`, `labels/test`.
- Write `data.yaml` with `path`, `train`, `val`, `test`, `names`.
- Write `manifest.json` with source image IDs, split, class mapping, and annotation IDs.

## Frontend Requirements

Add to the workbench:

- Quality panel showing counts and issues.
- Refresh quality after import, class creation, annotation save, and image selection changes.
- Create Dataset Version button, disabled until `ready_for_training`.
- Version result showing artifact path and split counts.

## Tasks

### Task 1: Backend Quality And Version APIs

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/app/quality/__init__.py`
- Create: `backend/app/quality/schemas.py`
- Create: `backend/app/quality/router.py`
- Create: `backend/app/versions/__init__.py`
- Create: `backend/app/versions/schemas.py`
- Create: `backend/app/versions/exporter.py`
- Create: `backend/app/versions/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_quality_version_api.py`

Steps:

- [ ] Write tests that import a sample zip, create a class, create annotations, verify quality summary, create a dataset version, and assert `data.yaml`, label files, and manifest exist.
- [ ] Implement the quality service and route.
- [ ] Implement dataset version model, schemas, exporter, and routes.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Commit with `feat: add quality review and dataset version export API`.

### Task 2: Frontend Quality And Version UI

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

Steps:

- [ ] Extend API client with `getQuality`, `createDatasetVersion`, and `listDatasetVersions`.
- [ ] Add test mocks and assert `Quality Review` and `Create Dataset Version` render.
- [ ] Add quality panel and version button/result to the workbench.
- [ ] Refresh quality after import and annotation save.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
- [ ] Commit with `feat: add quality review and version UI`.

### Task 3: End-To-End Verification

**Files:**
- Modify: `README.md`

Steps:

- [ ] Add README steps for quality review and dataset version export.
- [ ] Run all backend/frontend tests and build.
- [ ] Browser smoke: import dataset, create class, draw/save annotation, confirm quality readiness, create version, and inspect backend artifact files.
- [ ] Commit with `docs: add dataset version smoke steps`.

## Self-Review Checklist

- This plan advances the approved MVP toward training by making annotations exportable.
- It does not claim training exists yet.
- Tests must prove artifacts exist on disk, not only API responses.
- `image_dataset.zip` must remain untracked.
