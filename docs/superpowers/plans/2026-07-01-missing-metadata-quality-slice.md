# Missing Metadata Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface imported images with missing platform, altitude, timestamp, or metadata-row evidence in Quality Review.

**Architecture:** Treat missing metadata as a warning-only dataset quality issue. The backend counts images whose raw metadata row is empty or whose normalized platform/altitude/timestamp fields are missing, then returns image-level `missing_metadata` issue samples through the existing quality issue endpoint. The frontend renders a visible metric and uses the existing Open Issue path.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Missing Metadata Quality Signal

**Files:**
- Modify: `backend/app/quality/schemas.py`
- Modify: `backend/app/quality/router.py`
- Test: `backend/tests/test_quality_version_api.py`

- [x] **Step 1: Add summary field**

Add `missing_metadata_count: int` to `DatasetQualitySummary`.

- [x] **Step 2: Add detector**

Add a helper that treats an image as missing metadata when `image.metadata_` is empty, `image.platform` is empty, `image.altitude` is `None`, or `image.timestamp` is `None`.

- [x] **Step 3: Include summary issue**

Populate `missing_metadata_count` and append a warning message such as `1 image is missing platform, altitude, timestamp, or source metadata.` Do not change `ready_for_training`.

- [x] **Step 4: Add issue rows**

Add `missing_metadata` to supported issue types and return image-level warning rows with image ID, path, URL, and a message listing missing fields.

- [x] **Step 5: Test backend behavior**

Import a fixture with one fully metadated image and one image without a metadata row, assert the count/message/issue row, and assert version export can still proceed when labels are otherwise valid.

### Task 2: Frontend Missing Metadata Visibility

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Update API types**

Add `missing_metadata_count` to `DatasetQualitySummary` and add `missing_metadata` to `DatasetQualityIssue.issue_type`.

- [x] **Step 2: Render metric**

Show a `Missing metadata` metric in the Quality Review panel.

- [x] **Step 3: Test issue visibility**

Update the App test mock to include a missing metadata issue and assert the metric plus formatted `missing metadata` issue render.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [x] **Step 1: Full verification**

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
git commit -m "feat: flag missing image metadata"
```
