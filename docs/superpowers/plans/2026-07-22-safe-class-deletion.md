# 类别安全删除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为项目类别增加确认删除能力，仅允许删除没有标注和数据集版本引用的类别。

**Architecture:** 后端扩展类别响应中的使用计数，并新增带引用保护的 DELETE 接口；前端在类别行展示删除操作和保护原因，确认弹窗后调用接口。删除校验以后端数据为准，避免绕过前端造成历史标签或冻结版本损坏。

**Tech Stack:** FastAPI、SQLAlchemy、Pydantic、React 18、TypeScript、Vitest、Testing Library。

---

### Task 1: 后端类别使用计数和安全删除接口

**Files:**
- Modify: `~/DevProjects/YOLO_Trainer/backend/app/classes/schemas.py`
- Modify: `~/DevProjects/YOLO_Trainer/backend/app/classes/router.py`
- Test: `~/DevProjects/YOLO_Trainer/backend/tests/test_class_annotation_api.py`

- [x] **Step 1: 写后端失败测试**

增加接口测试：创建两个项目类别；为第一个类别写入一条图像标注；创建一个数据集版本引用第二个类别；验证 `GET /api/projects/{project_id}/classes` 返回对应 `annotation_count` 和 `version_count`；验证删除无引用类别返回 200 并从列表消失；删除有标注类别和被版本引用类别都返回 409；错误项目和错误类别返回 404。

- [x] **Step 2: 运行测试确认失败**

运行：

```bash
cd ~/DevProjects/YOLO_Trainer/backend
pytest tests/test_class_annotation_api.py -q
```

预期：新增断言失败，因为类别响应没有使用计数，也没有 DELETE 路由。

- [x] **Step 3: 扩展类别响应和计数查询**

在 `ClassRead` 增加 `annotation_count: int = 0` 和 `version_count: int = 0`。在 router 中增加 `_class_usage_counts`，用 `Annotation` 统计项目类别引用，用 `DatasetVersion.class_mapping` 在 Python 中检查所有版本是否包含 `str(class_id)`；`_read_class` 接受两个计数字典并写入响应。列表接口一次构造计数后传给每个类别。

- [x] **Step 4: 实现 DELETE 校验和删除**

新增 `@router.delete("/{project_id}/classes/{class_id}", response_model=ClassRead)`：先校验项目和类别归属，再统计标注与版本引用；任一计数大于零返回 `HTTPException(status_code=409, detail=...)`；无引用时保存删除前的 `ClassRead`，删除 `ClassDef`、提交并返回保存对象。删除接口不触碰 `Annotation`、`DatasetVersion` 或图像数据。

- [x] **Step 5: 运行后端测试确认通过**

运行 `cd ~/DevProjects/YOLO_Trainer/backend && pytest tests/test_class_annotation_api.py -q`，预期该文件全部通过。

### Task 2: 前端 API、状态和确认删除交互

**Files:**
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/api.ts`
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/App.tsx`
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/styles.css`
- Test: `~/DevProjects/YOLO_Trainer/frontend/src/App.test.tsx`

- [x] **Step 1: 写前端失败测试**

扩展 `ProjectClass` mock 数据的两个计数字段；增加测试验证未使用类别显示可用删除按钮，点击后出现确认弹窗，取消不会调用 API，确认后调用 `deleteClass` 并从列表移除；增加已使用类别删除按钮禁用及保护文案测试。

- [x] **Step 2: 运行前端测试确认失败**

运行 `cd ~/DevProjects/YOLO_Trainer/frontend && npx vitest run src/App.test.tsx -t "类别删除"`，预期失败，因为没有 `deleteClass` API 和删除控件。

- [x] **Step 3: 增加前端删除 API 和状态**

在 `api.ts` 增加 `deleteClass(projectId, classId)`，调用 `DELETE /api/projects/{projectId}/classes/{classId}`。在 `App.tsx` 增加 `classDeleteTarget` 和 `isDeletingClass` 状态；删除成功后从 `classes` 与 `versionClassIds` 移除，若删除的是当前类别则选中第一个剩余类别，调用 `refreshTrainingPrep`；失败时保留列表并显示后端错误。

- [x] **Step 4: 渲染保护状态和确认弹窗**

在类别行增加删除图标按钮。`annotation_count === 0 && version_count === 0` 时可点击，否则禁用并设置标题说明引用原因。确认弹窗显示类别名称和“此操作不可恢复”，取消只关闭弹窗，确认调用删除处理函数。保留现有编辑按钮和固定六类选择器。

- [x] **Step 5: 增加样式并运行前端相关测试**

增加删除按钮、保护提示和确认弹窗样式，沿用现有按钮与模态框视觉规范。运行 `cd ~/DevProjects/YOLO_Trainer/frontend && npx vitest run src/App.test.tsx -t "类别删除|固定类别|编辑"`，预期全部通过。

### Task 3: 全量验证和提交

**Files:**
- Modify: none

- [x] **Step 1: 运行后端全量测试**

运行 `cd ~/DevProjects/YOLO_Trainer/backend && pytest -q`，预期全量通过。

- [x] **Step 2: 运行前端全量测试和构建**

运行 `cd ~/DevProjects/YOLO_Trainer/frontend && npm test -- --run && npm run build`，预期 71 项既有测试加新增测试全部通过，Vite 构建成功。

- [x] **Step 3: 检查差异并提交**

运行 `cd ~/DevProjects/YOLO_Trainer && git diff --check && git status --short`，确认只暂存本次类别删除相关文件和计划文档；提交信息使用 `feat: add safe class deletion`。
