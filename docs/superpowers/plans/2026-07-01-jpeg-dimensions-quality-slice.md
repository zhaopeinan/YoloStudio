# JPEG Dimensions Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read JPEG image dimensions during import so JPG drone datasets participate correctly in tiny-box quality review and image browsing.

**Architecture:** Extend the existing lightweight image-size helper in `backend/app/datasets/importer.py` to parse JPEG SOF markers without adding external dependencies. Add width and height to the dataset image API so the frontend and tests can verify dimensions.

**Tech Stack:** FastAPI, SQLAlchemy, React TypeScript, pytest, Vitest.

---

### Task 1: Backend JPEG Dimension Parsing

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Modify: `backend/app/datasets/schemas.py`
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [x] **Step 1: Add JPEG fixture test**

Create a tiny JPEG header fixture with SOF0 width/height and assert imported dataset image rows expose `width` and `height`.

- [x] **Step 2: Add API fields**

Add `width: int | None` and `height: int | None` to `DatasetImageRead` and populate them in `list_dataset_images`.

- [x] **Step 3: Parse JPEG dimensions**

Add `_jpeg_size(path)` that walks JPEG markers, skips APP/DQT/etc. segments, and returns dimensions from SOF0/SOF2-style markers.

- [x] **Step 4: Keep fallback safe**

Return `(None, None)` for malformed or unsupported images instead of breaking import.

### Task 2: Frontend Type Coverage

**Files:**
- Modify: `frontend/src/api.ts`
- Test: `frontend/src/App.test.tsx`

- [x] **Step 1: Add dimensions to type**

Add `width: number | null` and `height: number | null` to `DatasetImage`.

- [x] **Step 2: Update test mocks**

Include dimensions in mocked image rows so TypeScript covers the new API shape.

### Task 3: Verification And Commit

**Files:**
- All files above.

- [x] **Step 1: Full verification**

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
git commit -m "feat: read jpeg image dimensions"
```
