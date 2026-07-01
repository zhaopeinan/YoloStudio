# Import Warning Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface YOLO labels that reference unknown class indexes in Quality Review instead of silently dropping them.

**Architecture:** During import, count label rows whose class index has no corresponding imported project class and store compact warning records in `Dataset.metadata`. The quality summary and issue list read those records and expose `unknown_class_reference` as actionable data quality evidence.

**Tech Stack:** FastAPI, SQLAlchemy JSON metadata, Pydantic, pytest.

---

### Task 1: Import Warning Capture

**Files:**
- Modify: `backend/app/datasets/importer.py`
- Test: `backend/tests/test_dataset_import_api.py`

- [ ] **Step 1: Add fixture**

Create a labeled YOLO zip where `data.yaml` has one class but `labels/train/frame001.txt` contains class index `2`.

- [ ] **Step 2: Record warnings**

When an annotation class index is missing from `class_by_index`, do not insert the annotation. Add a warning record with type `unknown_class_reference`, image path, class index, and count.

### Task 2: Quality API Exposure

**Files:**
- Modify: `backend/app/quality/router.py`
- Modify: `backend/app/quality/schemas.py`
- Test: `backend/tests/test_quality_version_api.py`

- [ ] **Step 1: Summary count**

Add `unknown_class_reference_count` to `DatasetQualitySummary` and append an issue message when the count is positive.

- [ ] **Step 2: Issue samples**

Add `unknown_class_reference` rows to `/quality/issues`, using dataset metadata warning records. These rows are dataset/image-path evidence and may not have an image ID when the image row was not created; for imported image rows, include image ID and URL.

- [ ] **Step 3: Tests**

Assert quality summary and issue list report the unknown class reference for the fixture.

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
git commit -m "feat: surface import label warnings"
```
