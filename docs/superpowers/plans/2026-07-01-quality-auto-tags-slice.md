# Quality Auto Tags Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn quality review findings into persistent annotation edge tags so operators can build correction queues from tiny, duplicate, and invalid boxes.

**Architecture:** Reuse the existing `Annotation.edge_tags` JSON array and quality issue builders. Add an explicit backend action that applies tags for annotation-level quality issues, and expose it as an operator-triggered button in the Quality Review panel.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite JSON columns, React, TypeScript, pytest, Vitest.

---

### Task 1: Backend Apply Quality Tags

**Files:**
- Modify: `backend/app/quality/schemas.py`
- Modify: `backend/app/quality/router.py`
- Test: `backend/tests/test_quality_version_api.py`

- [ ] Add `QualityTagApplyRequest` with `issue_type`, defaulting to `all`.
- [ ] Add `QualityTagApplySummary` with dataset ID, issue type, scanned issue count, updated annotation count, and applied tag count.
- [ ] Add `POST /api/datasets/{dataset_id}/quality/apply-tags`.
- [ ] Apply only annotation-level tags for `tiny_box`, `duplicate_box`, and `invalid_box`.
- [ ] Merge tags without removing manual tags or duplicating existing tags.
- [ ] Add tests for applying all tags and reapplying without duplicates.

### Task 2: Frontend Control

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] Add API types and `applyQualityTags(datasetId, issueType)`.
- [ ] Add an `Apply Auto Tags` button in Quality Review, disabled when no dataset or no applicable issues exist.
- [ ] Refresh quality, issue samples, image list, and currently selected annotations after applying tags.
- [ ] Show a compact summary such as `3 tags applied to 2 annotations`.
- [ ] Add a Vitest flow that applies duplicate-box tags and asserts the API call and summary.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Document quality auto tags and Image Browser edge-tag follow-up.
- [ ] Run focused backend and frontend tests.
- [ ] Run full backend tests, frontend tests, frontend build, and `git diff --check`.

Expected result: all commands pass. Commit the slice as `feat: apply quality issue tags`.
