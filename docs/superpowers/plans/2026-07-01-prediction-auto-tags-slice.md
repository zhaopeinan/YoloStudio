# Prediction Auto Tags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist prediction-derived failure evidence back onto ground-truth annotations so false-negative samples become reusable data-mining queues.

**Architecture:** Keep the existing `Annotation.edge_tags` JSON array as the storage surface. During prediction persistence, any unmatched annotation that creates a `false_negative` prediction row receives the `false_negative` tag without duplicating existing tags or adding manual-review tags.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite JSON columns, pytest.

---

### Task 1: Backend False-Negative Auto Tags

**Files:**
- Modify: `backend/app/prediction/runner.py`
- Test: `backend/tests/test_prediction_api.py`

- [ ] Add a small tag merge helper that preserves existing order and appends new unique tags.
- [ ] When `persist_predictions` creates a `false_negative` row for an unmatched annotation, merge `false_negative` into that annotation's `edge_tags`.
- [ ] Do not add `reviewed_prediction`; that tag remains reserved for explicit human review in the annotator.
- [ ] Extend the prediction API test to assert the missed annotation now includes `false_negative`.
- [ ] Extend the same test to assert `/api/datasets/{dataset_id}/images?edge_tag=false_negative` returns the missed image.

### Task 2: Documentation

**Files:**
- Modify: `README.md`

- [ ] Document that prediction jobs automatically tag missed ground-truth boxes with `false_negative`.
- [ ] Document that the Image Browser `Edge tag` filter can be used as a follow-up queue for these auto-tagged samples.

### Task 3: Verification

**Commands:**
- `cd backend && python -m pytest tests/test_prediction_api.py -v`
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: tag false-negative annotations`.
