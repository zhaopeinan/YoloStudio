# Image Browser Pagination Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators browse beyond the first 50 dataset images so large training sets can be previewed, filtered, and annotated page by page.

**Architecture:** Reuse the existing backend `limit` and `offset` query parameters. Extend the frontend image-list API with pagination options, track the current image page response in `App.tsx`, and render Previous/Next controls in the Image Browser.

**Tech Stack:** React, TypeScript, Vitest, Testing Library.

---

### Task 1: API Options

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] Add optional `limit` and `offset` arguments to `listImages`.
- [ ] Preserve the current default page size of 50.
- [ ] Keep filter serialization unchanged.

### Task 2: Frontend State And Controls

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Track `imagePage` metadata from `DatasetImageListResponse`.
- [ ] Reset to offset 0 when filters are applied or reset.
- [ ] Render range text and Previous/Next buttons above the image list.
- [ ] Disable Previous on the first page and Next on the last page.

### Task 3: Test Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`

- [ ] Mock more than one image page.
- [ ] Assert the initial load requests offset 0.
- [ ] Click Next and assert offset 50 is requested.
- [ ] Click Previous and assert offset 0 is requested again.
- [ ] Assert applying filters resets pagination to offset 0.

### Task 4: Verification

**Commands:**
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: paginate image browser`.
