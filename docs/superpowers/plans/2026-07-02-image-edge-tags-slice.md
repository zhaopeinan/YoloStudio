# Image Edge Tags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators and prediction analysis tag whole images as edge cases, including false-positive and hard-negative images that may not have saved boxes.

**Architecture:** Add `edge_tags` to `Image` as JSON, expose it through image list/read responses, and add a lightweight image tag update endpoint. Dataset edge-tag filtering and coverage should count both image-level and annotation-level tags, while prediction persistence automatically adds `false_positive` to images with false-positive predictions.

**Tech Stack:** FastAPI, SQLAlchemy JSON fields, SQLite JSON queries, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Image Tags

**Files:**
- Modify: `backend/app/db/models.py`
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Modify: `backend/app/annotations/schemas.py`
- Modify: `backend/app/annotations/router.py`
- Modify: `backend/app/prediction/runner.py`
- Test: `backend/tests/test_dataset_import_api.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] **Step 1: Add failing image tag API/filter test**

Create a dataset, update one image with `["hard_negative"]`, assert the image list returns `edge_tags`, assert `edge_tag=hard_negative` returns that image, and assert dataset coverage includes an edge-tag row with `image_count: 1` and `annotation_count: 0`.

- [ ] **Step 2: Add model/schema fields**

Add `Image.edge_tags` as a JSON list defaulting to `[]`. Add `edge_tags` to `DatasetImageRead` and add image tag update request/response schemas.

- [ ] **Step 3: Add image tag endpoint and filter support**

Expose `PUT /api/images/{image_id}/tags`, strip/unique tags, persist them, and make dataset edge-tag filtering match either `images.edge_tags` or `annotations.edge_tags`.

- [ ] **Step 4: Count image tags in coverage**

Update `GET /api/datasets/{dataset_id}/coverage` so edge-tag rows include image-level tags even when no annotations exist.

- [ ] **Step 5: Auto-tag false-positive images**

When `persist_predictions` creates a false-positive prediction, merge `false_positive` into the source image's `edge_tags`. Add a regression test that runs prediction on an image with a false positive and confirms the image list can filter by `edge_tag=false_positive`.

### Task 2: Frontend Image Tag UI

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Add API client/types**

Add `edge_tags` to `DatasetImage`, add `updateImageTags(imageId, tags)`, and mock it in tests.

- [ ] **Step 2: Render image-level tags**

Show image tags in Image Browser rows and selected image metadata. Keep annotation edge tags unchanged.

- [ ] **Step 3: Edit selected image tags**

Add a compact selected-image tag editor with presets including `hard_negative`, `occluded`, `camouflaged`, `low_light`, `dense`, and `false_positive`. Save via `PUT /api/images/{image_id}/tags`, update the selected image row locally, and refresh coverage.

- [ ] **Step 4: Cover UI behavior**

Extend the main integration test to toggle `hard negative` on the selected image, assert `updateImageTags` receives `["hard_negative"]`, and assert the image row displays the tag.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document image tags**

Explain that image-level tags capture hard negatives and false-positive frames, and that Image Browser `Edge tag` filter searches both image and annotation tags.

- [ ] **Step 2: Verify**

Run:

```bash
cd backend && python -m pytest tests/test_dataset_import_api.py::test_image_edge_tags_filter_and_coverage -v
cd backend && python -m pytest tests/test_prediction_api.py::test_prediction_job_tags_false_positive_images -v
cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
