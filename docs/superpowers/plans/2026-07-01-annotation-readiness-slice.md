# Annotation Readiness Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the annotation workbench self-explanatory when a dataset is loaded but classes, selected image, or selected class are missing.

**Architecture:** Keep the existing single-page React flow. Add derived readiness steps in `App.tsx`, render them inside the annotation panel, and auto-select the first available project class when loading or creating classes.

**Tech Stack:** React, TypeScript, Vitest, Testing Library.

---

### Task 1: Readiness State

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] Add a small derived `annotationReadinessSteps` array with labels and complete flags for dataset, class library, selected image, and selected class.
- [ ] Auto-select the first class when classes load and no selected class is set.
- [ ] Auto-select a newly created class after `Create Class` succeeds.

### Task 2: Readiness UI

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Render the readiness steps at the top of the Annotation panel.
- [ ] Keep the existing image canvas and box editor unchanged.
- [ ] Replace the generic empty text with state-specific guidance for missing dataset, class, image, or class selection.

### Task 3: Test Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`

- [ ] Assert the initial annotation panel shows missing dataset/class/image/class-selection readiness states.
- [ ] Assert importing or loading a dataset with classes auto-selects the first class.
- [ ] Assert creating a class in an empty project selects it and clears the class-readiness blocker.

### Task 4: Verification

**Commands:**
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: clarify annotation readiness`.
