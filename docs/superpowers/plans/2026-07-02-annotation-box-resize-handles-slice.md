# Annotation Box Resize Handles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators resize existing annotation boxes directly on the image canvas by dragging corner handles.

**Architecture:** Keep the existing annotation draft state and save path. Add a transient resize state alongside the existing move state, render four SVG corner handles for the selected ground-truth box, and update normalized center/size by dragging the selected corner while clamping to image bounds and minimum size.

**Tech Stack:** React, TypeScript, SVG pointer events, Vitest, Testing Library.

---

### Task 1: Frontend Canvas Resize Interaction

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add failing resize test**

Add a test that loads the dataset, opens `iris/frame002.jpg`, drags the selected annotation's bottom-right resize handle from `(725, 650)` to `(800, 700)` on a mocked 1000x1000 canvas, saves annotations, and asserts the saved draft has `x_center: 0.675`, `y_center: 0.6`, `width: 0.4`, and `height: 0.3`.

- [x] **Step 2: Add resize state and geometry helper**

Add `BoxResizeHandle` and `BoxResizeState`. Store the active box ID, handle name, and original left/top/right/bottom edges. Implement `resizeActiveAnnotation(point)` so each handle changes the intended edges and clamps the final rectangle through the same normalized annotation shape.

- [x] **Step 3: Render selected-box handles**

Update `BoxRect` to render four small corner rectangles only when selected. Each handle gets a stable `data-testid` such as `resize-handle-annotation-101-bottom-right`, an accessible label such as `Resize box target bottom right`, and a pointerdown callback that starts resize without starting move.

- [x] **Step 4: Style handles**

Add handle fill/stroke and cursor styles for diagonal resize. Keep the selected box visible and usable on dense imagery.

### Task 2: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document resizing**

Update the annotation smoke workflow to mention dragging a selected box corner to resize it before saving.

- [x] **Step 2: Verify**

Run:

```bash
cd frontend && npm test -- --run src/App.test.tsx -t "resizes an existing annotation box"
cd frontend && npm test -- --run
cd frontend && npm run build
cd backend && python -m pytest -v
git diff --check
```

- [ ] **Step 3: Commit**

Commit the slice without adding `image_dataset.zip`.
