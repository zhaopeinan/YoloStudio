# Training Augmentation Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single augmentation preset text field with structured YOLO-oriented augmentation controls for small/dense/occluded targets.

**Architecture:** Add a nested `augmentation` object to training run creation while preserving `augmentation_preset` for compatibility. Persist the full object in `run.config`, write it to `config.json`, and pass Ultralytics-supported fields into `model.train()`.

**Tech Stack:** FastAPI, Pydantic, SQLAlchemy, pytest, React/Vite/TypeScript/Vitest.

---

## Tasks

### Task 1: Backend Structured Augmentation Config

**Files:**
- Modify: `backend/app/training/schemas.py`
- Modify: `backend/app/training/runner.py`
- Modify: `backend/tests/test_training_api.py`

- [ ] Add `TrainingAugmentationConfig` with mosaic, mixup, copy-paste, hsv, translate, scale, fliplr, erasing, and gridmask fields.
- [ ] Persist the augmentation object in run config.
- [ ] Keep top-level `augmentation_preset` for existing clients.
- [ ] Pass Ultralytics-supported augmentation kwargs into `model.train()`.
- [ ] Keep unsupported `gridmask` recorded in config as a local strategy flag.

### Task 2: Frontend Training Strategy Controls

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add frontend augmentation config type.
- [ ] Replace the single augmentation text field with compact numeric controls and toggles.
- [ ] Send the nested augmentation object when starting training.
- [ ] Assert key strategy fields appear in the create-run request.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`

- [ ] Document structured augmentation controls and config persistence.
- [ ] Run `cd backend && python -m pytest -v`.
- [ ] Run `cd frontend && npm test -- --run && npm run build`.
