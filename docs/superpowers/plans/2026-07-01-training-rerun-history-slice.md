# Training Rerun From History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a local operator quickly reuse a previous training run's configuration for iterative YOLO experiments.

**Architecture:** Reuse the existing `TrainingRun.config` payload and `POST /api/training/runs` endpoint. Add frontend helpers that normalize config values back into the Training Setup controls, plus Run History actions to load a run's config or immediately rerun it on the current latest dataset version.

**Tech Stack:** React, TypeScript, existing FastAPI training API, Vitest, Testing Library.

---

### Task 1: Config Hydration Helpers

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add config reader helpers**

Create helpers to read string, number, boolean, and augmentation values from `TrainingRun.config` with safe fallbacks to current defaults.

- [x] **Step 2: Add `applyRunConfigToForm`**

Populate model, epochs, image size, batch size, device, strategy name, augmentation, TTA, and threshold-scan controls from a selected run.

### Task 2: Run History Actions

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add `Load Config` action**

Render a secondary button on each Run History row. Clicking it updates Training Setup controls without starting a new run.

- [x] **Step 2: Add `Rerun` action**

Render a secondary action that starts a new run using the selected historical config and the current latest dataset version. Disable it when no version exists or a run is already active.

- [x] **Step 3: Surface status**

Show a compact message such as `Loaded config from Run #1` or `Started rerun from Run #1`.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document reruns**

Mention that Run History can load or rerun previous configs for iterative experiments.

- [x] **Step 2: Verify**

Run:

```bash
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 3: Commit**

Commit this slice without adding `image_dataset.zip`.
