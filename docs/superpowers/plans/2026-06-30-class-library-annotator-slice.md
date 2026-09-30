# Class Library And Annotator Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing import-preview scaffold into a usable dataset workbench where a user can import the provided zip into SQLite/workspace storage, create project-level classes, browse images, draw one or more bounding boxes, and persist annotations.

**Architecture:** Keep FastAPI as the owner of workspace files, SQLite rows, and JSON APIs. Keep React/Vite as a single operational workbench screen for this slice: scan/import, project classes, image browser, and lightweight SVG-over-image annotation canvas.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x, pytest, React, Vite, TypeScript, Vitest, Testing Library, lucide-react.

---

## Scope Check

This plan implements the next MVP slice from the approved design:

- Project-level class library.
- Dataset import from the already scanned local zip into the managed workspace.
- Image metadata persistence and file serving from workspace.
- Lightweight annotation save/load with normalized YOLO-style boxes, optional `track_id`, and manual edge-case tags stored as JSON.
- Frontend class creation, image browser, canvas drawing, annotation list, and save/load state.

This plan intentionally does not implement dataset version snapshots, YOLO label export, quality review, training, prediction, or export. Those come after annotations exist.

## File Structure

Backend files:

- `backend/app/db/models.py`: add `ClassDef` and `Annotation` entities.
- `backend/app/classes/__init__.py`: package marker.
- `backend/app/classes/schemas.py`: class request/response schemas.
- `backend/app/classes/router.py`: class list/create endpoints.
- `backend/app/annotations/__init__.py`: package marker.
- `backend/app/annotations/schemas.py`: annotation request/response schemas.
- `backend/app/annotations/router.py`: annotation list/replace endpoints.
- `backend/app/datasets/importer.py`: import zip images/metadata into workspace and SQLite.
- `backend/app/datasets/schemas.py`: add import/image/project response schemas.
- `backend/app/datasets/router.py`: add import, dataset image list, and image file endpoints.
- `backend/app/main.py`: include class and annotation routers.
- `backend/tests/test_class_annotation_api.py`: API tests for class creation and annotation replace/load.
- `backend/tests/test_dataset_import_api.py`: API tests for importing a sample zip and serving images.

Frontend files:

- `frontend/src/api.ts`: add typed clients for import, images, classes, and annotations.
- `frontend/src/App.tsx`: add import action, class library panel, image browser, annotator canvas, and save/load behavior.
- `frontend/src/styles.css`: layout and annotation styling.
- `frontend/src/App.test.tsx`: extend smoke test for class/import/annotation controls.

## Backend Behavior

### Dataset Import

`POST /api/datasets/import`

Request:

```json
{
  "source_path": "~/DevProjects/YOLO_Trainer/image_dataset.zip",
  "project_name": "YOLO Trainer Project",
  "dataset_name": "image_dataset"
}
```

Behavior:

- Reuse scanner logic for counts and metadata.
- Create a project when no existing project name matches.
- Create a dataset row linked to the project.
- Extract image files only into `workspace/projects/<project_id>/datasets/<dataset_id>/images/<group>/...`.
- Create one `Image` row per extracted image.
- Match `meta.jsonl` rows by filename and collection group.
- Store `platform`, `altitude`, `timestamp`, and full metadata JSON on `Image`.
- Do not extract labels because the current archive has none.

Response:

```json
{
  "project_id": 1,
  "dataset_id": 1,
  "project_name": "YOLO Trainer Project",
  "dataset_name": "image_dataset",
  "image_count": 1099,
  "groups": [
    {"name": "iris", "image_count": 433, "metadata_rows": 433},
    {"name": "vtol", "image_count": 666, "metadata_rows": 666}
  ]
}
```

### Image Browser

`GET /api/datasets/{dataset_id}/images?limit=50&offset=0`

Returns ordered image rows with `id`, `relative_path`, `platform`, `altitude`, `timestamp`, `annotation_count`, and `image_url`.

`GET /api/images/{image_id}/file`

Returns the image file from the managed workspace using `FileResponse`.

### Class Library

`GET /api/projects/{project_id}/classes`

Returns all active project classes ordered by ID.

`POST /api/projects/{project_id}/classes`

Creates a class with `name`, optional `color`, and optional `description`.

Rules:

- Class names are unique per project.
- Default color is assigned by the backend when omitted.
- Response includes `id`, `project_id`, `name`, `color`, `description`, and `active`.

### Annotations

`GET /api/images/{image_id}/annotations`

Returns annotations for an image with class metadata.

`PUT /api/images/{image_id}/annotations`

Replaces annotations for an image. Each annotation uses normalized bbox fields:

```json
{
  "class_id": 1,
  "x_center": 0.5,
  "y_center": 0.5,
  "width": 0.2,
  "height": 0.1,
  "track_id": "target-001",
  "edge_tags": ["occluded", "small"]
}
```

Rules:

- All bbox values must be within `(0, 1]` for width/height and `[0, 1]` for centers.
- The bbox must fit inside the image bounds after converting from center format.
- Class must belong to the same project as the image dataset.
- Replace semantics are deliberate for this slice; frontend sends the full current box list.

## Frontend Behavior

The first screen remains the workbench, not a landing page.

Add:

- Scan and import controls.
- Imported dataset summary.
- Class library panel with class name/color creation.
- Image browser list from imported dataset.
- Main image viewer with SVG overlay.
- Pointer drag creates a box when a class is selected.
- Existing boxes render over the image with class color.
- Annotation side panel allows class selection, track ID, edge tags text, delete, and save.
- Save sends full box list to `PUT /api/images/{image_id}/annotations`.

The annotator may use normalized coordinates and render boxes based on displayed image dimensions. It does not need zoom/pan or keyboard shortcuts in this slice.

## Tasks

### Task 1: Backend Import, Class, And Annotation APIs

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/app/classes/__init__.py`
- Create: `backend/app/classes/schemas.py`
- Create: `backend/app/classes/router.py`
- Create: `backend/app/annotations/__init__.py`
- Create: `backend/app/annotations/schemas.py`
- Create: `backend/app/annotations/router.py`
- Create: `backend/app/datasets/importer.py`
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_dataset_import_api.py`
- Create: `backend/tests/test_class_annotation_api.py`

- [ ] **Step 1: Write failing backend tests**

Create tests that:

- Build a temporary zip with two small valid PNG files and `meta.jsonl`.
- POST `/api/datasets/import` and assert project, dataset, image rows, and group counts.
- GET `/api/datasets/{dataset_id}/images` and assert `image_url` exists.
- GET `/api/images/{image_id}/file` and assert HTTP 200 image response.
- POST a project class and assert it is listed.
- PUT one annotation on an image and assert GET returns it with class name/color.
- PUT an invalid out-of-bounds annotation and assert HTTP 422 or 400.

Run:

```bash
cd backend
python -m pytest tests/test_dataset_import_api.py tests/test_class_annotation_api.py -v
```

Expected: fail because routes/models do not exist.

- [ ] **Step 2: Implement models, importer, schemas, and routers**

Implementation requirements:

- Add `ClassDef` and `Annotation` SQLAlchemy models.
- Use JSON column for `Annotation.edge_tags`.
- Importer must extract images into workspace and persist rows.
- Do not modify or delete original zip.
- Use FastAPI `HTTPException` for missing rows and bad requests.
- Include new routers in `backend/app/main.py`.

- [ ] **Step 3: Run backend tests**

Run:

```bash
cd backend
python -m pytest -v
```

Expected: all backend tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add backend/app backend/tests
git commit -m "feat: add dataset import classes and annotations API"
```

### Task 2: Frontend Import, Class Library, And Annotator UI

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: Write failing frontend test**

Extend the frontend test to mock:

- `importDataset`
- `listImages`
- `createClass`
- `listClasses`
- `getAnnotations`
- `replaceAnnotations`

Assert that the UI renders:

- `Import Dataset`
- `Class Library`
- `Image Browser`
- `Annotation`
- `Save Annotations`

Run:

```bash
cd frontend
npm test -- --run
```

Expected: fail until UI/API clients exist.

- [ ] **Step 2: Implement API client types and UI**

Implementation requirements:

- Preserve existing scan UI behavior.
- Add an import button that calls `/api/datasets/import`.
- After import, load classes and images for the returned project/dataset.
- Allow creating a class with name and color.
- Select the first class by default after creation.
- Select the first image by default after image list loads.
- Render selected image using `image_url`.
- Use an SVG overlay to draw boxes with pointer drag.
- Store new boxes as normalized YOLO center coordinates.
- Allow deleting boxes from the annotation list.
- Save annotations through `replaceAnnotations`.

- [ ] **Step 3: Run frontend checks**

Run:

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: tests and build pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add frontend/src
git commit -m "feat: add class library and annotator UI"
```

### Task 3: End-To-End Verification And README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add instructions for:

- Importing the provided zip from the UI.
- Creating a class.
- Selecting an image.
- Drawing and saving a box.

- [ ] **Step 2: Run all checks**

Run:

```bash
cd backend
python -m pytest -v
cd ../frontend
npm test -- --run
npm run build
```

Expected: backend tests pass, frontend tests pass, build succeeds.

- [ ] **Step 3: Browser smoke**

Run backend and frontend dev servers. In the browser:

- Scan the provided zip.
- Import the dataset.
- Create a class such as `target`.
- Select the first image.
- Draw a box.
- Save annotations.
- Reload or reselect the image and confirm the annotation persists.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md
git commit -m "docs: add annotation workflow smoke steps"
```

## Self-Review Checklist

- This plan advances the approved MVP by adding class and annotation primitives before training.
- It avoids training and version snapshots until annotations can be persisted.
- Backend and frontend tasks have separate write scopes.
- The existing `image_dataset.zip` remains untracked and must not be committed.
- Verification includes unit/API tests, frontend build, and a browser smoke of the annotation workflow.
