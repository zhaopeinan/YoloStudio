# GridMask Training Dataset Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing GridMask training option produce a real run-level augmented YOLO dataset instead of only recording a strategy flag.

**Architecture:** Keep frozen dataset versions immutable. When a training run has `augmentation.gridmask` enabled, derive `gridmask_dataset/` under that run's artifact directory, copy labels and `data.yaml`, apply deterministic grid masks to copied images, and pass the derived `data.yaml` to Ultralytics. If GridMask is disabled, continue training from the original dataset version path.

**Tech Stack:** FastAPI backend, SQLAlchemy run metadata, Pillow for image masking when available, pytest.

---

### Task 1: GridMask Dataset Builder

**Files:**
- Modify: `backend/app/training/runner.py`

- [ ] Add a helper that copies a version artifact into `runs/<run_id>/gridmask_dataset`.
- [ ] Copy `labels/` and non-image metadata without changing YOLO label files.
- [ ] For every image under `images/train`, `images/val`, and `images/test`, write a masked copy with deterministic grid spacing based on image size.
- [ ] Rewrite the derived `data.yaml` path to the derived dataset root.
- [ ] Return the derived `data.yaml` path and log where it was created.

### Task 2: Training Runner Integration

**Files:**
- Modify: `backend/app/training/runner.py`

- [ ] In `execute_training_run`, read the run config before calling `YOLO.train`.
- [ ] If `augmentation.gridmask` is enabled, build the derived dataset after validating the original version `data.yaml`.
- [ ] Pass the derived `data.yaml` path to `model.train`.
- [ ] If Pillow is unavailable or a source image cannot be processed, fail the run with a clear message.

### Task 3: Tests And Documentation

**Files:**
- Modify: `backend/tests/test_training_api.py`
- Modify: `README.md`

- [ ] Unit-test the GridMask helper using a tiny exported version fixture.
- [ ] Assert the derived dataset has masked images, unchanged labels, and a `data.yaml` rooted at `gridmask_dataset`.
- [ ] Update training docs to say GridMask generates a run-local augmented dataset.

### Task 4: Verification

**Commands:**
- `cd backend && python -m pytest tests/test_training_api.py -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: apply gridmask training dataset`.
