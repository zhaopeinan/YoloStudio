# Annotator Correction Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn prediction failure review from a passive overlay into a correction loop inside the lightweight annotator.

**Architecture:** Keep the existing prediction review endpoint and annotation save API. Add frontend-only review controls that toggle ground-truth and prediction layers, render prediction labels with confidence, and promote false-positive prediction boxes into annotation drafts that the operator can review and save.

**Tech Stack:** React/Vite/TypeScript/Vitest with the existing FastAPI annotation API.

---

## Tasks

### Task 1: Annotator Review Controls

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Add `showGroundTruthLayer` and `showPredictionLayer` state.
- [ ] Render ground-truth boxes only when the GT layer is enabled.
- [ ] Render prediction boxes only when the prediction layer is enabled.
- [ ] Add compact checkbox controls and a legend in the prediction review banner.

### Task 2: Prediction-To-Annotation Drafts

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add a helper that converts a prediction into a `DraftBox`.
- [ ] Add an `Add as annotation` action for false-positive prediction rows in the review banner.
- [ ] Add `false_positive` and `reviewed_prediction` tags to promoted prediction drafts.
- [ ] Add a `Mark reviewed` action for false-negative rows that tags the matched ground-truth annotation.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`

- [ ] Document layer toggles and correction workflow.
- [ ] Run frontend tests and build.
- [ ] Run backend tests to guard the unchanged API contract.
