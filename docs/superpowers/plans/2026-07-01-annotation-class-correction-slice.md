# Annotation Class Correction Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators correct the class assigned to an existing annotation box without deleting and redrawing it.

**Architecture:** Reuse the existing full-replace annotation API because each saved annotation already carries `class_id`. Add a per-box class selector in the frontend editor, update the draft annotation metadata from the project class library, and prove the saved payload contains the corrected class ID.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, existing FastAPI annotation API.

---

### Task 1: Box Class Selector

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Add a compact class selector to each `.box-editor`.
- [ ] Populate it from the project-level `classes` list.
- [ ] When a class is selected, update `class_id`, `class_name`, and `class_color` on the draft box.
- [ ] Keep the existing title swatch/name in sync with the selected class.

### Task 2: Frontend Regression Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`

- [ ] Load the saved dataset fixture.
- [ ] Create a second class with the existing create-class flow.
- [ ] Open an image with an existing `target` annotation.
- [ ] Change that annotation's class to `vehicle`.
- [ ] Save annotations and assert `replaceAnnotations` receives `class_id: 2` for the corrected box.

### Task 3: Verification

**Commands:**
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: edit annotation class assignments`.
