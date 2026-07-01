# Image Dimensions Quality Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface imported images whose pixel dimensions could not be read, so annotators understand why tiny-box checks may not apply.

**Architecture:** Reuse the existing dataset quality summary and issue APIs. Treat missing dimensions as a warning, like missing metadata, and do not block version export.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TypeScript, Vitest.

---

### Task 1: Backend Quality Signal

**Files:**
- Modify: `backend/app/quality/schemas.py`
- Modify: `backend/app/quality/router.py`
- Test: `backend/tests/test_quality_version_api.py`

- [ ] Add `missing_image_dimensions_count` to the quality summary schema.
- [ ] Add `missing_image_dimensions` to the supported quality issue type set.
- [ ] Count images where `Image.width` or `Image.height` is `None`.
- [ ] Add the summary message `1 image has unreadable image dimensions.` or plural equivalent.
- [ ] Add warning issue rows with message `Image width or height could not be read.`
- [ ] Prove warning-only behavior by exporting a version successfully when dimensions are missing but annotations are valid.

### Task 2: Frontend Quality Display

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] Add `missing_image_dimensions_count` to `DatasetQualitySummary`.
- [ ] Add `missing_image_dimensions` to the issue type union.
- [ ] Render a `Missing dimensions` metric in Quality Review.
- [ ] Add a Vitest case that shows the warning metric, issue label, issue message, and opens the affected image.

### Task 3: Verification

**Commands:**
- `cd backend && python -m pytest -q tests/test_quality_version_api.py`
- `cd backend && python -m pytest -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: flag unreadable image dimensions`.
