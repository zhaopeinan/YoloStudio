# Annotation Geometry And Copy Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lightweight annotator more useful by allowing precise bbox geometry edits and copying annotations from adjacent images.

**Architecture:** Reuse the existing in-memory draft boxes and `replaceAnnotations` save path. Geometry edits mutate draft normalized coordinates in the right-side box editor; copy actions load annotations from the previous or next image in the current browser list and convert them to draft boxes for the selected image.

**Tech Stack:** React/Vite/TypeScript/Vitest.

---

## Tasks

### Task 1: Bbox Geometry Inputs

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add numeric inputs for normalized x center, y center, width, and height.
- [ ] Clamp coordinates and dimensions to safe normalized ranges.
- [ ] Keep overlay rectangles updated immediately as values change.

### Task 2: Adjacent Image Copy

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add `Copy Previous` and `Copy Next` buttons in the box panel.
- [ ] Load adjacent image annotations through the existing API.
- [ ] Convert copied annotations into unsaved draft boxes for the current image.
- [ ] Surface copy errors in the annotation panel.

### Task 3: Verification

- [ ] Run `cd frontend && npm test -- --run && npm run build`.
- [ ] Run `git diff --check`.
