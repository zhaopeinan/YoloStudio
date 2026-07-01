# YOLO Label Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import existing YOLO `data.yaml` classes and `labels/*.txt` boxes so the built-in annotator can correct existing datasets instead of starting from blank images.

**Architecture:** Extend dataset import source bundles with optional class names and label rows keyed by image group/path. During import, create any missing project classes in YOLO index order, copy images, create `Image` rows, then insert valid `Annotation` rows for matched labels.

**Tech Stack:** FastAPI, SQLAlchemy, pathlib/zipfile, React/Vitest smoke coverage through existing import flow, pytest.

---

### Task 1: Backend Label Parsing

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [ ] **Step 1: Add dataset fixture**

Create a zip or folder fixture containing:
- `data.yaml` with two classes.
- `images/train/frame001.png`.
- `labels/train/frame001.txt` containing a valid YOLO row.

- [ ] **Step 2: Parse class names**

Add a small parser for common `names` forms:
- `names: [target, decoy]`
- `names: {0: target, 1: decoy}`
- indented mapping below `names:`

- [ ] **Step 3: Parse label rows**

Read `.txt` files under a `labels` path, ignore invalid lines, and convert valid five-column rows into normalized annotation candidates.

### Task 2: Persist Classes And Annotations

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [ ] **Step 1: Create project classes**

When imported labels exist, create missing classes in YOLO class-index order using stable default colors. Reuse existing project classes by exact name if present.

- [ ] **Step 2: Match labels to images**

Map `labels/train/foo.txt` to image relative path candidates such as `images/train/foo.png`, `images/train/foo.jpg`, and any imported source image with the same basename in the same group.

- [ ] **Step 3: Insert annotations**

After each image row is flushed, insert `Annotation` rows for valid matching labels with class IDs from the imported class map.

- [ ] **Step 4: Verify API behavior**

Assert imported dataset image rows report `annotation_count == 1`, `listClasses` returns imported classes, and `GET /api/images/{id}/annotations` returns the imported box.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [ ] **Step 1: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Commit**

Stage only this slice and the plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: import yolo labels"
```
