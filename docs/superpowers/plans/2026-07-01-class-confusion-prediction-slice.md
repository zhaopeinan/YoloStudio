# Class Confusion Prediction Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect wrong-class predictions as class-confusion samples and correctly map YOLO version class indexes back to project class IDs.

**Architecture:** Keep `Prediction.failure_type` as a string field and add `class_confusion` as a supported value. Prediction persistence resolves each raw YOLO prediction through the dataset version `class_mapping`, matches same-class boxes first, then high-IoU cross-class boxes as class confusions, and auto-tags the matched ground-truth annotation.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, React, TypeScript, pytest, Vitest.

---

### Task 1: Backend Mapping And Confusion Detection

**Files:**
- Modify: `backend/app/prediction/runner.py`
- Modify: `backend/app/prediction/router.py`
- Modify: `backend/app/datasets/router.py`
- Modify: `backend/app/experiments/summary.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] Resolve raw prediction `class_id` values through `DatasetVersion.class_mapping` when they are YOLO class indexes.
- [ ] Preserve compatibility with existing tests and injected predictors that already provide database class IDs.
- [ ] Match predictions in two passes: exact class + IoU threshold first, then cross-class + IoU threshold as `class_confusion`.
- [ ] Store class-confusion predictions with the predicted database class ID and `matched_annotation_id`.
- [ ] Auto-tag the matched ground-truth annotation with `class_confusion`, without adding `reviewed_prediction`.
- [ ] Let prediction list, image list, and review counts filter/count `class_confusion`.
- [ ] Include `class_confusion` in experiment class outcomes and confusion matrix rows.

### Task 2: Frontend Support

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add `class_confusion` to dataset image and prediction failure filter types.
- [ ] Add `Class confusion` options to Image Browser and Prediction Analysis failure selects.
- [ ] Show class-confusion counts in prediction summary and overlay review banner.
- [ ] Render class-confusion boxes with distinct warning color and readable label text.
- [ ] Extend the dashboard test fixture to include one class-confusion row and assert the UI can filter it.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document `class_confusion` as a separate failure type and auto edge tag.
- [ ] Run `cd backend && python -m pytest tests/test_prediction_api.py -v`.
- [ ] Run `cd backend && python -m pytest tests/test_experiment_summary_api.py -v`.
- [ ] Run `cd frontend && npm test -- --run`.
- [ ] Run `cd frontend && npm run build`.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Run `git diff --check`.

Expected result: all commands pass. Commit the slice as `feat: detect prediction class confusion`.
