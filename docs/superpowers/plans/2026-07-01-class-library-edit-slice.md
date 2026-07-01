# Class Library Edit Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators correct project-level class names and colors after a dataset has already been imported or annotated.

**Architecture:** Add a focused class update endpoint on the existing class router. The frontend keeps class editing local to the Class Library panel, updates the shared class state after a save, and relies on existing derived maps so filters, version subsets, and annotation labels render the edited class.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TypeScript, Vitest, Testing Library.

---

### Task 1: Backend Class Update API

**Files:**
- Modify: `backend/app/classes/schemas.py`
- Modify: `backend/app/classes/router.py`
- Test: `backend/tests/test_class_annotation_api.py`

- [ ] Add `ClassUpdate` with optional `name`, `color`, and `description`.
- [ ] Add `PATCH /api/projects/{project_id}/classes/{class_id}`.
- [ ] Validate project and class ownership.
- [ ] Reject blank names and duplicate class names within the project.
- [ ] Return the updated `ClassRead`.
- [ ] Prove annotations read back with the edited class name and color.

### Task 2: Frontend API And Editing State

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] Add `updateClass(projectId, classId, body)`.
- [ ] Track the class currently being edited plus draft name/color.
- [ ] Add Edit, Save, and Cancel actions in each class row.
- [ ] Replace the updated class in `classes` while preserving selected class and version subset IDs.
- [ ] Refresh quality prep after a successful update.

### Task 3: Frontend Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Mock `updateClass`.
- [ ] Edit the default `target` class to `vehicle`.
- [ ] Assert the API receives the edited name and color.
- [ ] Assert the class chip, version subset, and annotation box title display the updated name.
- [ ] Style edit rows without changing the workbench layout density.

### Task 4: Verification

**Commands:**
- `cd backend && python -m pytest tests/test_class_annotation_api.py -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: edit class library entries`.
