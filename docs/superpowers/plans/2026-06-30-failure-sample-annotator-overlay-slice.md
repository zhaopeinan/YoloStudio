# Failure Sample Annotator Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator open prediction failure samples in the annotator and see prediction boxes overlaid with ground-truth boxes.

**Architecture:** The backend exposes a per-job/per-image review detail combining image metadata, ground-truth annotations, and predictions. The frontend reuses the current annotation canvas, selects the image referenced by a prediction row, loads review details, and renders prediction boxes as a separate overlay style.

**Tech Stack:** FastAPI, SQLAlchemy, pytest, React/Vite/TypeScript/Vitest.

---

## Scope

In scope:

- `GET /api/prediction-jobs/{job_id}/images/{image_id}/review`
- Response includes image, ground-truth annotations, predictions, and failure counts for that image.
- Prediction rows get an `Open Image` action.
- Annotation canvas overlays prediction boxes with confidence/failure type.
- Prediction overlay can be cleared when selecting ordinary images.

Out of scope:

- Editing prediction boxes as annotations.
- Confusion matrix and curve charts.
- Multi-image review carousel.

## Tasks

### Task 1: Backend review detail API

**Files:**
- Modify: `backend/app/prediction/schemas.py`
- Modify: `backend/app/prediction/router.py`
- Modify: `backend/tests/test_prediction_api.py`

- [ ] Extend tests to fetch review detail for a failure image.
- [ ] Add response schemas.
- [ ] Implement review endpoint.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Commit `feat: add prediction review detail API`.

### Task 2: Frontend failure sample overlay

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add API client for review detail.
- [ ] Add `Open Image` action on prediction rows.
- [ ] Select matching image in browser and load annotations.
- [ ] Render prediction overlay boxes.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
- [ ] Commit `feat: link prediction failures to annotator overlay`.

### Task 3: Docs and smoke

**Files:**
- Modify: `README.md`

- [ ] Document failure sample review.
- [ ] Verify backend/frontend tests.
- [ ] Commit `docs: add failure sample review steps`.

## Self-Review Checklist

- This slice directly advances training-result visual review.
- It reuses the lightweight annotator rather than adding a separate viewer.
- The next slice can build on this for correction workflow and prediction-to-annotation promotion.
