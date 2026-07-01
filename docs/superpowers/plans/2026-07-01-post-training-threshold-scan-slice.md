# Post-Training Threshold Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the training setup `threshold_scan` option execute a default confidence scan after a successful training run.

**Architecture:** Keep prediction jobs as the only persisted evaluation unit. Add a small training-runner hook that checks `run.config.threshold_scan` after training completes, then creates and executes prediction jobs for default thresholds. The hook writes run log lines so the UI's existing run/prediction refresh surfaces the automatic work.

**Tech Stack:** FastAPI background tasks, SQLAlchemy sessions, existing prediction runner, pytest.

---

### Task 1: Training Runner Hook

**Files:**
- Modify: `backend/app/training/runner.py`
- Test: `backend/tests/test_training_api.py`

- [ ] **Step 1: Add default thresholds**

Add `DEFAULT_THRESHOLD_SCAN_VALUES = (0.15, 0.25, 0.35, 0.5, 0.65)` near the training status constants.

- [ ] **Step 2: Add helper**

Add `run_post_training_threshold_scan(run_id, bind=None, settings=None, predictor=None)`. It should open a fresh session, load the completed `TrainingRun`, return immediately when missing or `config.threshold_scan` is false, append a run log line, create and execute one prediction job per default threshold with `image_scope="all"`, then append a completion log line.

- [ ] **Step 3: Call helper after success**

In `execute_training_run`, after `complete_training_run(run_id, bind=bind)`, call `run_post_training_threshold_scan(run_id, bind=bind, settings=settings or get_settings())`. Do not call it after failed training.

- [ ] **Step 4: Cover helper behavior**

Add a unit/integration test that creates a completed run with `threshold_scan: True`, monkeypatches prediction output, calls the helper directly, and asserts five completed prediction jobs exist, the summary threshold table includes five thresholds, and run logs include start/completion lines.

- [ ] **Step 5: Run focused backend tests**

Run: `cd backend && python -m pytest tests/test_training_api.py tests/test_prediction_api.py -v`

Expected: training and prediction tests pass.

### Task 2: Frontend And Docs Surface

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `README.md`

- [ ] **Step 1: Clarify toggle text**

Rename the training toggle label from `Threshold scan` to `Auto threshold scan` to distinguish it from manual prediction scanning.

- [ ] **Step 2: Update tests if label changes affect them**

If the App test queries the old label, update it to the new label.

- [ ] **Step 3: Document automatic behavior**

Update README training and prediction sections to state that enabling `Auto threshold scan` runs default thresholds after successful training and populates the experiment dashboard.

- [ ] **Step 4: Full verification**

Run:

```bash
cd backend && python -m pytest -v
cd frontend && npm test -- --run
cd frontend && npm run build
git diff --check
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

Stage only tracked project changes and this plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: run threshold scan after training"
```
