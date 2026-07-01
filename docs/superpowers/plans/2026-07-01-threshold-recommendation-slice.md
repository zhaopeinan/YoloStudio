# Threshold Recommendation Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make threshold scans actionable by reporting F1 for each threshold and recommending the best confidence threshold.

**Architecture:** Extend the existing run experiment summary contract. Compute precision, recall, F1, and best threshold in the backend so the frontend only renders evidence.

**Tech Stack:** FastAPI, Pydantic, SQLAlchemy, pytest, React, TypeScript, Vitest.

---

### Task 1: Backend Summary Enrichment

**Files:**
- Modify: `backend/app/experiments/schemas.py`
- Modify: `backend/app/experiments/summary.py`
- Test: `backend/tests/test_experiment_summary_api.py`

- [ ] Add `f1` to every `ThresholdPoint`.
- [ ] Add `ThresholdRecommendation` with `job_id`, `confidence_threshold`, `precision`, `recall`, and `f1`.
- [ ] Add `threshold_recommendation` to `RunExperimentSummary`.
- [ ] Select the point with highest F1; break ties by higher recall, then higher confidence threshold.
- [ ] Keep `threshold_recommendation` as `null` when there are no completed prediction jobs.

### Task 2: Frontend Dashboard Display

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] Add TypeScript fields for `f1` and `threshold_recommendation`.
- [ ] Render a compact best-threshold summary in the Experiment Dashboard.
- [ ] Add an F1 column to the Threshold Scan table.
- [ ] Update Vitest dashboard coverage to assert best threshold and F1 rendering.

### Task 3: Verification

**Commands:**
- `cd backend && python -m pytest -q tests/test_experiment_summary_api.py`
- `cd backend && python -m pytest -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: recommend threshold scan balance`.
