# Version Image Metadata Manifest Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze image-level metadata into dataset version manifests so training versions can support later temporal review, target consistency checks, and situation-map alignment.

**Architecture:** Keep YOLO label files standard and unchanged. Extend each image entry in `manifest.json` and `DatasetVersion.split_manifest` with width, height, platform, altitude, timestamp, and source metadata copied from the imported image row at version creation time.

**Tech Stack:** FastAPI backend, SQLAlchemy models, existing dataset version exporter, pytest.

---

### Task 1: Manifest Metadata Fields

**Files:**
- Modify: `backend/app/versions/exporter.py`

- [ ] Add `width`, `height`, `platform`, `altitude`, `timestamp`, and `metadata` to each manifest image entry.
- [ ] Keep `source_relative_path`, `export_image`, `export_label`, `split`, and `annotations` unchanged.
- [ ] Preserve annotation-level `track_id` and `edge_tags` as they are today.

### Task 2: Regression Coverage

**Files:**
- Modify: `backend/tests/test_quality_version_api.py`

- [ ] Create a dataset version from the standard import fixture.
- [ ] Read the generated manifest JSON from disk.
- [ ] Assert the first image entry includes platform, altitude, timestamp, dimensions, and metadata copied from the source image row.
- [ ] Assert the persisted API version payload still reports the same split counts and class mapping.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document that `manifest.json` freezes image metadata for temporal/situation-map review.
- [ ] Run focused version tests and full verification.

**Commands:**
- `cd backend && python -m pytest tests/test_quality_version_api.py -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: freeze image metadata in versions`.
