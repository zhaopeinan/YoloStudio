# Quality Issue Drilldown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn dataset quality review counts into actionable image/box issue samples that open directly in the annotator.

**Architecture:** Keep the existing `/quality` summary endpoint for readiness gates and add a focused `/quality/issues` endpoint that returns image-level and annotation-level evidence. The frontend loads the issue samples beside the summary and reuses the existing image selection/annotation loading flow to open a sample for correction.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, Vitest, pytest.

---

### Task 1: Backend Quality Issues API

**Files:**
- Modify: `backend/app/quality/schemas.py`
- Modify: `backend/app/quality/router.py`
- Test: `backend/tests/test_quality_version_api.py`

- [ ] **Step 1: Add schemas**

Add `DatasetQualityIssue` with issue type, severity, message, image fields, optional annotation/class/geometry fields, and `DatasetQualityIssueList` with dataset ID, pagination, total, and items.

- [ ] **Step 2: Build issue rows**

Add a helper that emits:
- `unannotated_image` for images with no annotations.
- `tiny_box` for boxes smaller than 10 pixels on either axis when image dimensions are known.
- `invalid_box` for boxes outside normalized frame bounds.

- [ ] **Step 3: Add endpoint**

Expose `GET /api/datasets/{dataset_id}/quality/issues?limit=50&offset=0&issue_type=all`, returning deterministic image ID / annotation ID ordering.

- [ ] **Step 4: Test endpoint**

Extend the quality test to assert unannotated image and tiny box issues include image IDs, annotation IDs, messages, and pagination fields.

### Task 2: Frontend Issue Samples

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add API client**

Add `DatasetQualityIssue`, `DatasetQualityIssueListResponse`, and `listQualityIssues(datasetId)`.

- [ ] **Step 2: Load issue samples**

Fetch quality issues after import and during `refreshTrainingPrep`, keep them in state, and clear them when no dataset is loaded.

- [ ] **Step 3: Render clickable samples**

Show up to the returned issue samples under Quality Review with issue type, message, image path, and a button named `Open Issue`.

- [ ] **Step 4: Open sample in annotator**

When a quality sample is opened, ensure the image exists in `images`, clear prediction overlay, set `selectedImageId`, and let the existing annotation effect load saved boxes.

- [ ] **Step 5: Test UI path**

Update the App test mock so a quality issue sample appears. Click `Open Issue` and assert the image row/annotation save path still works.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [ ] **Step 1: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Commit**

Stage only this slice and the plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: add quality issue drilldown"
```
