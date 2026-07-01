# Image Failure Type Filter Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators turn prediction failure mining into a direct image-browser annotation queue by filtering dataset images by prediction failure type.

**Architecture:** Reuse persisted `Prediction.failure_type` rows instead of adding a new tag table. Extend the dataset image list endpoint with an optional `failure_type` query that keeps images having at least one matching prediction. Add the same filter to the frontend image browser and include it in reset/apply flows.

**Tech Stack:** FastAPI, SQLAlchemy, React, TypeScript, Vitest, pytest.

---

### Task 1: Backend Image Filter

**Files:**
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] Add `failure_type` query parameter with values `all`, `matched`, `false_positive`, and `false_negative`.
- [ ] For non-`all` values, filter dataset images through `Prediction.image_id`.
- [ ] Keep existing class, edge-tag, metadata, and label-status filters composable with the failure filter.
- [ ] Add an integration test that creates prediction rows and asserts the dataset image list returns only the matching failure sample.

### Task 2: Frontend Image Browser Filter

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add `failure_type` to `DatasetImageFilters`.
- [ ] Add a compact `Failure` select to Image Browser filters.
- [ ] Include it in apply and reset requests.
- [ ] Update the main app test to assert `listImages` receives `failure_type`.

### Task 3: Verification

**Commands:**
- `cd backend && python -m pytest tests/test_prediction_api.py -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: filter images by prediction failure`.
