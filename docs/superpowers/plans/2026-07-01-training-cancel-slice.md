# Training Cancel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local single-user training cancellation path so active runs can move to `cancelled`, write a log line, and release the one-active-run lock.

**Architecture:** Treat cancellation as a durable status transition. The backend exposes a small cancel endpoint that only accepts `queued`, `preparing`, or `running` runs, sets `status="cancelled"`, records `ended_at`, and appends logs. The frontend shows a `Cancel Run` action for active runs and refreshes run state afterward.

**Tech Stack:** FastAPI, SQLAlchemy, React, Vitest, pytest.

---

### Task 1: Backend Cancellation

**Files:**
- Modify: `backend/app/training/runner.py`
- Modify: `backend/app/training/router.py`
- Test: `backend/tests/test_training_api.py`

- [ ] **Step 1: Add cancel helper**

Add `cancel_training_run(run_id, bind=None) -> bool`. It should return `False` when the run is missing or not in `ACTIVE_STATUSES`; otherwise set `status="cancelled"`, `ended_at=_now()`, clear neither config nor artifacts, commit, append `training cancelled`, and return `True`.

- [ ] **Step 2: Add API endpoint**

Add `POST /api/training/runs/{run_id}/cancel` returning `TrainingRunRead`. Return 404 when the run does not exist, 400 when it is not active, otherwise call the helper and return the refreshed run.

- [ ] **Step 3: Test cancellation**

Extend `test_training_api.py`: create a run that remains `running`, cancel it, assert status is `cancelled`, assert logs contain `training cancelled`, assert a new run can be queued afterward, and assert cancelling a completed run returns 400.

- [ ] **Step 4: Focused backend verification**

Run: `cd backend && python -m pytest tests/test_training_api.py -v`

Expected: training API tests pass.

### Task 2: Frontend Cancel Control

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add API client**

Add `cancelTrainingRun(runId: number): Promise<TrainingRun>` in `api.ts`.

- [ ] **Step 2: Add handler**

Add `handleCancelTrainingRun(runId)` in `App.tsx`, track the cancelling run ID, call the API, merge the returned run into `runs`, refresh logs, and refresh training runs.

- [ ] **Step 3: Add button**

In each run row, show `Cancel Run` only when the run status is active. Disable while that run is being cancelled.

- [ ] **Step 4: Test UI**

Update `App.test.tsx` mock API and active-run test to click `Cancel Run`, assert API call, and assert the run row can show `cancelled`.

- [ ] **Step 5: Focused frontend verification**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: App tests pass.

### Task 3: Docs, Full Verification, Commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document cancellation**

Update README training lifecycle to include `cancelled` and the `Cancel Run` action.

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

Stage only this slice and the plan file, leaving `image_dataset.zip` untracked. Commit with:

```bash
git commit -m "feat: add training run cancellation"
```
