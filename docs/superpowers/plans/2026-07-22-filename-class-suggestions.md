# 文件名类别建议与负样本标注 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据图像文件名提供可人工修正的类别建议，并让标注员明确保存“待标注、已标注、已确认无目标”三种状态。

**Architecture:** 使用现有 `Image.metadata` JSON 保存 `annotation_status`，不新增数据库迁移；后端在图像列表和标注保存接口中返回/校验状态。前端新增纯函数解析文件名建议，App 负责草稿状态和保存，标注工具栏与图像列表负责展示和操作。

**Tech Stack:** FastAPI、Pydantic、SQLAlchemy、React、TypeScript、Vitest、Pytest。

---

### Task 1: 扩展图像与标注接口的状态契约

**Files:**
- Modify: `backend/app/datasets/schemas.py:75-92`
- Modify: `backend/app/datasets/router.py:137-220`
- Modify: `backend/app/annotations/schemas.py:1-70`
- Modify: `backend/app/annotations/router.py:1-90`
- Modify: `frontend/src/api.ts:74-95,183-195,781-793`
- Test: `backend/tests/test_dataset_import_api.py`
- Test: `backend/tests/test_class_annotation_api.py`

- [ ] **Step 1: Write failing backend tests**

Add tests that create one image, assert old metadata returns `annotation_status: "unreviewed"`, save a positive annotation and assert `annotated`, then save an empty annotation with `annotation_status: "negative"` and assert `negative`.

```python
def test_annotation_status_round_trip(client, image_id, class_id):
    image = client.get(f"/api/datasets/1/images").json()["items"][0]
    assert image["annotation_status"] == "unreviewed"

    positive = client.put(
        f"/api/images/{image_id}/annotations",
        json={"annotations": [{
            "class_id": class_id,
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.2,
            "height": 0.2,
        }]},
    )
    assert positive.json()["annotation_status"] == "annotated"

    negative = client.put(
        f"/api/images/{image_id}/annotations",
        json={"annotations": [], "annotation_status": "negative"},
    )
    assert negative.json()["annotation_status"] == "negative"
```

- [ ] **Step 2: Run the focused backend test and verify it fails**

Run: `cd backend && pytest -q tests/test_dataset_import_api.py -k annotation_status`

Expected: FAIL because image and annotation responses do not yet expose `annotation_status`.

- [ ] **Step 3: Add the status schema and metadata helpers**

Define one shared literal in the annotations schema:

```python
AnnotationStatus = Literal["unreviewed", "annotated", "negative"]

class AnnotationReplace(BaseModel):
    annotations: list[AnnotationWrite]
    annotation_status: AnnotationStatus | None = None

class AnnotationList(BaseModel):
    items: list[AnnotationRead]
    annotation_status: AnnotationStatus
```

In the image route, read `image.metadata_`, default missing/invalid values to `unreviewed`, and return the status in `DatasetImageRead`. On annotation replacement, reject `negative` when the request contains boxes, set `annotated` when boxes are present and no explicit status is supplied, and persist the requested status in `image.metadata_` before committing.

- [ ] **Step 4: Expand frontend API types and request helpers**

Add `AnnotationStatus`, `annotation_status` to `DatasetImage`, `AnnotationListResponse`, and `AnnotationReplace` payloads. Change `replaceAnnotations` to accept an optional status and send:

```ts
body: JSON.stringify({ annotations, annotation_status: status })
```

- [ ] **Step 5: Run backend regression tests**

Run: `cd backend && pytest -q tests/test_dataset_import_api.py tests/test_class_annotation_api.py`

Expected: PASS, including existing empty-annotation compatibility tests.

### Task 2: Implement deterministic filename suggestion logic

**Files:**
- Create: `frontend/src/annotation-suggestions.ts`
- Create: `frontend/src/annotation-suggestions.test.ts`

- [ ] **Step 1: Write the failing pure-function tests**

Cover normalization, a unique `firet` match to `fire_truck`, a unique token match for `suv_camo`, no match, and multiple matches returning candidates without a selected result.

```ts
expect(suggestFilenameClasses("scene_firet_001.jpg", classes)).toMatchObject({
  selected: "fire_truck",
});
expect(suggestFilenameClasses("scene_unknown_001.jpg", classes).selected).toBeNull();
```

- [ ] **Step 2: Run the focused frontend test and verify it fails**

Run: `cd frontend && npm test -- --run src/annotation-suggestions.test.ts`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Add normalized token matching**

Implement `normalizeFilenameToken` by lowercasing and removing non-alphanumeric separators. Match exact normalized class names first, then configured aliases/prefixes such as `firet -> fire_truck`; return `{ selected, candidates }`. Only set `selected` for one unambiguous candidate. Never create a category or annotation box.

- [ ] **Step 4: Run the helper tests**

Run: `cd frontend && npm test -- --run src/annotation-suggestions.test.ts`

Expected: PASS.

### Task 3: Add annotation status and suggestion controls to the workbench

**Files:**
- Modify: `frontend/src/annotation-workbench.ts:1-35`
- Modify: `frontend/src/AnnotationToolbar.tsx:5-165`
- Modify: `frontend/src/App.tsx:297-340,780-890,2048-2105,3600-4065`
- Modify: `frontend/src/api.ts:74-95,139-147`
- Modify: `frontend/src/styles.css` in annotation toolbar/image list sections
- Test: `frontend/src/AnnotationToolbar.test.tsx`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Write failing component tests**

Add tests for:

- a filename suggestion badge and “采用建议” control;
- changing the selected drawing class after accepting a suggestion;
- the “确认无目标” action being enabled only when there are no boxes;
- saving a negative image with zero boxes and showing `已确认无目标`;
- adding a box to a negative image changing the draft to `annotated`;
- next-image selection preferring `unreviewed` images.

- [ ] **Step 2: Update image and draft state in App**

Track `annotationStatus` for the selected image. Load it from `DatasetImage`/`AnnotationListResponse`, reset it when switching images, set it to `annotated` when a box is created, and return it to `unreviewed` when the last box is deleted. Add `handleConfirmNegative` that refuses when boxes exist and calls the save path with `negative`.

The save path must submit the explicit status and update the selected image’s `annotation_status` and `annotation_count` only after the API succeeds. A failed save must retain the draft and status.

- [ ] **Step 3: Add the toolbar controls**

Pass the filename suggestion and status into `AnnotationToolbar`. Render the suggestion, an “采用建议” button that only changes `selectedClassId`, and a “确认无目标” button. Keep the existing class selector and save controls; disable negative confirmation while boxes exist or while saving.

- [ ] **Step 4: Add image-list status and suggestion display**

Show each image filename, `建议：<类别>` when unambiguous, and one of `待标注`, `已标注`, or `已确认无目标`. Preserve existing box-count text and image selection behavior.

- [ ] **Step 5: Update next-image selection and filters**

Extend the selectable image type with `annotation_status`; choose the next `unreviewed` image before falling back to the next image. Extend image filter types and backend query parameters only if the existing filter drawer needs explicit negative/unreviewed filtering; preserve the existing `unannotated` API behavior for compatibility.

- [ ] **Step 6: Run focused frontend tests**

Run: `cd frontend && npm test -- --run src/AnnotationToolbar.test.tsx src/App.test.tsx src/annotation-workbench.test.ts`

Expected: PASS.

### Task 4: Complete regression coverage and verify the workflow

**Files:**
- Modify: `backend/tests/test_dataset_import_api.py`
- Modify: `backend/tests/test_class_annotation_api.py`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/annotation-workbench.test.ts`

- [ ] **Step 1: Add negative-state backend validation tests**

Verify that `negative` plus a non-empty annotation list returns 400, that legacy empty saves remain `unreviewed`, and that status survives a subsequent image-list request.

- [ ] **Step 2: Run the complete backend suite**

Run: `cd backend && pytest -q`

Expected: all backend tests pass.

- [ ] **Step 3: Run the complete frontend suite and build**

Run:

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: all Vitest tests pass and Vite produces a production build.

- [ ] **Step 4: Run formatting/static checks**

Run: `cd backend && ruff check app tests`; then from the repository root run `git diff --check`.

Expected: no lint or whitespace errors.

- [ ] **Step 5: Verify the browser workflow**

Open `http://127.0.0.1:5173/#annotation`, select an image whose filename contains `firet`, confirm the suggestion selects `fire_truck` without creating a box, confirm an empty image as a negative sample, add a box to a negative image, and verify the status changes to `已标注` after saving.
