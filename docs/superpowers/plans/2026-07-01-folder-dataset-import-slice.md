# Folder Dataset Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local YOLO Trainer scan and import either a `.zip` archive or a normal local dataset folder.

**Architecture:** Keep zip behavior intact and add a source-level scanner/importer that branches only at file enumeration and file copy time. Both source types produce the same scan summary, workspace image layout, image metadata rows, and UI workflow.

**Tech Stack:** FastAPI, SQLAlchemy, pathlib/shutil/zipfile, React, Vitest, pytest.

---

### Task 1: Backend Source Scanner

**Files:**
- Modify: `backend/app/datasets/scanner.py`
- Modify: `backend/app/datasets/router.py`
- Test: `backend/tests/test_dataset_scanner.py`
- Test: `backend/tests/test_dataset_scan_api.py`

- [ ] **Step 1: Add folder scan tests**

Create a temporary dataset folder with `yolo_dataset/vtol/images/raw/*.jpg`, `yolo_dataset/iris/images/raw/*.jpg`, and per-group `meta.jsonl`. Assert `scan_dataset_source(folder)` returns three images, two groups, three metadata rows, and the same altitude/timestamp rollups as zip scanning.

- [ ] **Step 2: Implement source scanner**

Add `scan_dataset_source(source_path: Path)` that resolves the path, raises `FileNotFoundError` for missing sources, scans folders with `Path.rglob("*")`, scans zips through the existing archive reader, and rejects non-zip files with `Expected a .zip file or directory`.

- [ ] **Step 3: Update scan API**

Route `/api/datasets/scan` through `scan_dataset_source`, change the missing-source message to `Dataset source was not found`, and keep invalid zip errors as HTTP 400 responses.

### Task 2: Backend Folder Import

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Modify: `backend/app/datasets/schemas.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [ ] **Step 1: Add folder import API test**

Create a temporary dataset folder with two valid PNG images and matching `meta.jsonl`; import it through `/api/datasets/import`; assert two images are persisted, platform/altitude metadata is available through `/images`, and the image file endpoint serves the copied PNG bytes.

- [ ] **Step 2: Share import pipeline**

Refactor `import_dataset` so zip and folder sources share project creation, dataset creation, image copy destination building, metadata lookup, image sizing, and image row creation. Set `source_type` to `zip` or `folder` in the `Dataset` row.

- [ ] **Step 3: Preserve zip compatibility**

Keep `_image_relative_member_path`, metadata lookup by `(group, basename)`, existing workspace layout, and existing zip tests passing.

### Task 3: Frontend And Docs

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `README.md`

- [ ] **Step 1: Update intake copy**

Rename the heading and input label from zip-only language to dataset-source language while keeping the default sample path pointed at `image_dataset.zip`.

- [ ] **Step 2: Update frontend test**

Update the App test to query `Dataset path` and keep the existing scan/import workflow intact.

- [ ] **Step 3: Document folder support**

Update README examples so users know `source_path` accepts either a `.zip` file or a folder.

### Task 4: Verification And Commit

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
git commit -m "feat: support folder dataset import"
```
