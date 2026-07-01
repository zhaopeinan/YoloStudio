# Prediction TTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the training setup `TTA` option affect real prediction jobs by enabling Ultralytics test-time augmentation during inference.

**Architecture:** Keep `tta` as a run-level config field. Prediction jobs read `run.config["tta"]` and pass it into `YOLO.predict(..., augment=True)` when enabled, so manual prediction jobs, threshold scans, and post-training threshold scans all share the same behavior.

**Tech Stack:** FastAPI backend, existing prediction runner, pytest monkeypatching.

---

### Task 1: Backend TTA Prediction

**Files:**
- Modify: `backend/app/prediction/runner.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] **Step 1: Add helper**

Add `run_uses_tta(run: TrainingRun) -> bool` that returns `bool(run.config.get("tta"))`.

- [ ] **Step 2: Pass augment to Ultralytics**

Update `predict_images` so the `model.predict` call includes `augment=run_uses_tta(run)`. Keep existing `conf` and `verbose=False` arguments.

- [ ] **Step 3: Test model.predict arguments**

Add a backend test that monkeypatches `sys.modules["ultralytics"]` with a fake `YOLO` class, creates fake weights at `<run.artifact_path>/ultralytics/weights/best.pt`, sets `run.config["tta"] = True`, calls `predict_images`, and asserts the fake model saw `augment=True`.

- [ ] **Step 4: Run focused tests**

Run: `cd backend && python -m pytest tests/test_prediction_api.py -v`

Expected: prediction tests pass.

### Task 2: Docs, Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document TTA effect**

Update README to state that enabling `TTA` passes Ultralytics `augment=True` to prediction jobs and threshold scans.

- [ ] **Step 2: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

Stage only this slice's tracked changes and the new plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: enable tta for prediction jobs"
```
