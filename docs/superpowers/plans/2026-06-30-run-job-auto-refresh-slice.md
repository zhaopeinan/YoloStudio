# Run And Prediction Auto Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep training and prediction monitor panels current while jobs are queued, preparing, or running.

**Architecture:** Reuse existing list/log endpoints and add frontend polling effects keyed by the imported project and latest job IDs. Poll only active statuses, refresh logs for active or already-open rows, and stop automatically when terminal statuses arrive.

**Tech Stack:** React/Vite/TypeScript/Vitest with existing FastAPI endpoints.

---

## Tasks

### Task 1: Training Polling

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Add status helpers for active and terminal task states.
- [ ] Poll project training runs while any run is `queued`, `preparing`, or `running`.
- [ ] Refresh logs for active runs automatically.
- [ ] Show compact auto-refresh status in run history.

### Task 2: Prediction Polling

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] Poll prediction jobs for the latest run while a job is `queued` or `running`.
- [ ] Refresh the latest job predictions and logs during polling.
- [ ] Preserve manual `Open Image` review behavior while lists refresh.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`

- [ ] Document automatic monitor refresh.
- [ ] Run frontend tests and build.
- [ ] Run backend tests to protect the unchanged API surface.
