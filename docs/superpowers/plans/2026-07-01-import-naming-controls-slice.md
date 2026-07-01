# Import Naming Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local operator choose the project name and dataset name at import time so multiple datasets can reuse a project-level class library.

**Architecture:** The backend already accepts `project_name` and `dataset_name`, so this is a frontend workflow slice. Add controlled inputs beside the dataset path, pass their trimmed values to `importDataset`, and document the behavior.

**Tech Stack:** React, TypeScript, Vitest, README docs.

---

### Task 1: Frontend Import Inputs

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add state**

Add `projectName` and `datasetName` state with existing defaults: `YOLO Trainer Project` and `image_dataset`.

- [ ] **Step 2: Render controls**

Add `Project name` and `Dataset name` inputs in the Dataset Intake panel. Disable import when path, project name, or dataset name is blank.

- [ ] **Step 3: Pass names to API**

Call `importDataset(datasetPath, projectName.trim(), datasetName.trim())`.

- [ ] **Step 4: Test request body**

Update the App test mock so `importDataset` is a spy and assert it receives the custom project and dataset names after user edits.

### Task 2: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document names**

Mention that datasets imported into the same project name reuse the project-level class library.

- [ ] **Step 2: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

Stage only this slice and the plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: add import naming controls"
```
