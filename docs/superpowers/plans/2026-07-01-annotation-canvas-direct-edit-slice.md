# Annotation Canvas Direct Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lightweight annotator more usable by letting operators select an existing box on the image canvas, drag it to reposition it, and nudge it precisely from the box editor.

**Architecture:** Reuse the existing draft annotation state and full `replaceAnnotations` save path. Add transient UI state for the selected box and active move drag; render each ground-truth box as an interactive SVG group; mutate normalized `x_center` and `y_center` through the same draft update helper used by numeric edits.

**Tech Stack:** React, TypeScript, SVG pointer events, Vitest, Testing Library.

---

### Task 1: Selection And Drag State

**Files:**
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add selection state**

Track `selectedAnnotationId` and a move drag state containing the local box ID, start pointer, and original center.

- [x] **Step 2: Select boxes from canvas and editor**

Clicking an existing SVG box selects it. Clicking a box editor also selects the matching box. Deleting a selected box clears the selection.

- [x] **Step 3: Drag selected boxes**

Pointer down on a box starts a move drag. Pointer move updates `x_center` and `y_center` while preserving width and height and clamping inside the normalized image.

### Task 2: Precise Nudge Controls

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add nudge helper**

Add `nudgeAnnotation(localId, dx, dy)` and clamp the center so the full box remains inside the canvas.

- [x] **Step 2: Add editor controls**

Render four small arrow icon buttons on each box editor. Buttons nudge by `0.01` normalized units and select that box.

- [x] **Step 3: Style selected state**

Highlight the selected canvas box and matching editor row without changing layout dimensions.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document direct edits**

Update the annotation smoke workflow to mention selecting, dragging, and nudging boxes before saving.

- [x] **Step 2: Verify**

Run:

```bash
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
