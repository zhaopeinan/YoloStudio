# Quality Issue Filter Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators filter quality issue samples by issue type so large datasets can be reviewed one problem class at a time.

**Architecture:** Reuse the existing backend `issue_type` query parameter. Add frontend state and a select control that refreshes only the issue sample list while keeping summary metrics intact.

**Tech Stack:** React, TypeScript, Vitest, Testing Library.

---

### Task 1: API Contract

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] Add a `DatasetQualityIssueType` union including `all` and all backend-supported issue types.
- [ ] Update `listQualityIssues(datasetId, issueType)` to append `?issue_type=<value>`.

### Task 2: UI And State

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Add `qualityIssueType` state with default `all`.
- [ ] Use the selected issue type when loading datasets and refreshing quality prep.
- [ ] Render a `Quality issue type` select above issue samples.
- [ ] Refresh only issue samples when the select changes.

### Task 3: Frontend Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`

- [ ] Select `Duplicate boxes`.
- [ ] Assert `listQualityIssues` is called with `duplicate_box`.
- [ ] Assert the duplicate issue remains openable from the filtered list.

### Task 4: Verification

**Commands:**
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: filter quality issue samples`.
