# Edge Tag Presets Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make edge-case mining consistent by giving annotators quick toggles for common hard-case tags while preserving free-form tags.

**Architecture:** Keep the existing `edge_tags` array and backend validation unchanged. Add frontend-only preset toggles that update the same draft annotation state used by the free-form tag input.

**Tech Stack:** React, TypeScript, Vitest, Testing Library.

---

### Task 1: Preset Toggle UI

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] Add a small `edgeTagPresets` list: `occluded`, `camouflaged`, `low_light`, `small`, `dense`, `hard_negative`.
- [ ] Add a `toggleTag` helper that adds or removes one tag while preserving tag order and avoiding duplicates.
- [ ] Render preset buttons under each annotation `Edge tags` input.
- [ ] Give selected preset buttons an active visual state.

### Task 2: Frontend Coverage

**Files:**
- Modify: `frontend/src/App.test.tsx`

- [ ] Toggle `camouflaged` and `hard_negative` on an annotation.
- [ ] Save annotations.
- [ ] Assert `replaceAnnotations` receives both preset tags.

### Task 3: Verification

**Commands:**
- `cd frontend && npm test -- --run`
- `cd frontend && npm run build`
- `cd backend && python -m pytest -v`
- `git diff --check`

Expected result: all commands pass. Commit the slice as `feat: add edge tag presets`.
