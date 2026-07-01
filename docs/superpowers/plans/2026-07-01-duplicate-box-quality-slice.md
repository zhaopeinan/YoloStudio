# Duplicate Box Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add duplicate bounding-box detection to dataset quality review so repeated labels are visible, actionable, and blocked before dataset version export.

**Architecture:** Treat duplicate boxes as a dataset quality issue when a later annotation on the same image and class overlaps an earlier annotation with IoU >= 0.95. Reuse the existing quality summary and issue sample endpoints, and add a visible frontend metric beside the other quality counters.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Duplicate Quality Gate

**Files:**
- Modify: `backend/app/quality/schemas.py`
- Modify: `backend/app/quality/router.py`
- Modify: `backend/app/versions/exporter.py`
- Test: `backend/tests/test_quality_version_api.py`

- [x] **Step 1: Add summary field**

Add `duplicate_box_count: int` to `DatasetQualitySummary`.

- [x] **Step 2: Add duplicate detector**

Build deterministic duplicate issue rows by scanning annotations ordered by image ID and annotation ID. For each annotation, compare it with earlier annotations on the same image and class. If IoU is at least `0.95`, emit the later annotation as a `duplicate_box` issue with `error` severity.

- [x] **Step 3: Include duplicate count and readiness gate**

Populate `duplicate_box_count`, append a human-readable issue message, and set `ready_for_training` false when duplicates exist.

- [x] **Step 4: Block version export**

Add `dataset has duplicate boxes` to dataset version export blockers when `duplicate_box_count > 0`.

- [x] **Step 5: Test backend behavior**

Create duplicate annotations in the quality test, assert duplicate count/message/issues, assert the duplicate issue references the second annotation, and assert version creation rejects the dataset.

### Task 2: Frontend Duplicate Visibility

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Update API types**

Add `duplicate_box_count` to `DatasetQualitySummary` and add `duplicate_box` to `DatasetQualityIssue.issue_type`.

- [x] **Step 2: Render duplicate metric**

Show a `Duplicate boxes` metric in the Quality Review panel.

- [x] **Step 3: Test frontend visibility**

Update the App test mock to include a duplicate issue and assert the metric plus formatted `duplicate box` issue sample render.

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
git commit -m "feat: add duplicate box quality checks"
```
