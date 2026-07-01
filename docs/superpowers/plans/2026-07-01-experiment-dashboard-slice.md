# Experiment Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a run-level experiment dashboard summary with metric curves, class outcome rows, confusion matrix, and confidence-threshold scan data.

**Architecture:** Build a backend summary from existing `RunMetric`, `PredictionJob`, and `Prediction` rows instead of inventing a separate storage model. The frontend loads the latest run summary with the run list and renders small operational charts/tables in the existing workbench style.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, pytest, React/Vite/TypeScript/Vitest.

---

## Tasks

### Task 1: Backend Experiment Summary API

**Files:**
- Create: `backend/app/experiments/__init__.py`
- Create: `backend/app/experiments/schemas.py`
- Create: `backend/app/experiments/summary.py`
- Modify: `backend/app/training/router.py`
- Create: `backend/tests/test_experiment_summary_api.py`

- [ ] Add Pydantic schemas for metric series, class rows, confusion cells, threshold rows, and run summary.
- [ ] Aggregate `RunMetric` rows into named epoch series.
- [ ] Aggregate completed prediction jobs into threshold rows ordered by confidence threshold.
- [ ] Aggregate the latest completed prediction job into per-class matched/false-positive/false-negative counts.
- [ ] Aggregate matched predictions into a class-level confusion matrix.
- [ ] Expose `GET /api/training/runs/{run_id}/summary`.
- [ ] Test summary output on a run with metrics and predictions.

### Task 2: Frontend Experiment Dashboard

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add summary API types and function.
- [ ] Load the latest run summary whenever the latest run changes or prediction jobs refresh.
- [ ] Render training curves from metric series.
- [ ] Render class outcome rows and confusion matrix.
- [ ] Render threshold scan rows.
- [ ] Surface empty states without blocking existing run history.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`

- [ ] Document the experiment dashboard summary endpoint and UI.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
