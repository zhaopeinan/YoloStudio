# 固定六类类别预设 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将类别管理改为固定六类复选选择并批量启用，取消手动输入，同时保持旧项目类别和标注工作流兼容。

**Architecture:** 在现有 `App.tsx` 中增加不可变的六类预设及选择状态，调用已有逐项创建类别接口补齐缺失类别；现有类别列表、编辑接口和后端数据结构不变。使用当前 CSS 体系增加预设卡片样式，并在 `App.test.tsx` 中覆盖批量选择行为。

**Tech Stack:** React 18、TypeScript、Vite、Vitest、Testing Library、lucide-react。

---

### Task 1: 增加固定类别预设状态和批量创建行为

**Files:**
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/App.tsx`
- Test: `~/DevProjects/YOLO_Trainer/frontend/src/App.test.tsx`

- [x] **Step 1: 写批量类别选择的失败测试**

在 `App.test.tsx` 的类别管理测试附近增加测试，验证类别页显示六个固定复选框、默认全部选中；取消 `suv_camo` 后点击“启用所选类别”，只为五个缺失类别调用 `createClass`，并验证请求名称和颜色来自预设。增加已有类别场景，验证同名类别不重复调用创建接口。

- [x] **Step 2: 运行新增测试确认当前实现失败**

运行：

```bash
cd ~/DevProjects/YOLO_Trainer/frontend
npx vitest run src/App.test.tsx -t "固定类别"
```

预期：失败，因为当前页面只提供“类别名称”输入框，没有固定预设复选框和批量启用按钮。

- [x] **Step 3: 增加预设常量和选择状态**

在 `App.tsx` 的默认颜色常量附近增加：

```ts
const fixedClassPresets = [
  { name: "fire_truck", color: "#e45756" },
  { name: "person_white", color: "#2f80ed" },
  { name: "prius_hybrid", color: "#27ae60" },
  { name: "car_lexus", color: "#f2994a" },
  { name: "prius_hybrid_camo", color: "#9b51e0" },
  { name: "suv_camo", color: "#1f6f78" },
] as const;
```

增加 `selectedFixedClassNames` 状态，初始值为全部预设名称；在清空工作区时恢复全部选中。移除类别创建专用的名称、颜色和创建中状态，保留编辑状态用于历史类别兼容。

- [x] **Step 4: 用批量处理函数替换逐个创建函数**

将 `handleCreateClass` 改为 `handleEnableFixedClasses`，处理逻辑固定为：没有导入数据集或没有选择时直接返回；按预设顺序过滤出当前 `classes` 中不存在的名称；对缺失项依次调用现有 `createClass(importedDataset.project_id, { name, color })`；将新对象追加到 `classes`、`versionClassIds`，并在完成后选中第一个可用类别、刷新训练准备数据；若本次前项目无类别则提交到 `annotation`。任一步失败时保留当前选择并显示 `类别启用失败` 错误。

- [x] **Step 5: 运行新增测试确认实现通过**

运行同一条 Vitest 命令，预期新增的固定类别测试全部通过。

### Task 2: 重做类别管理页面为固定复选卡片

**Files:**
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/App.tsx`
- Modify: `~/DevProjects/YOLO_Trainer/frontend/src/styles.css`
- Test: `~/DevProjects/YOLO_Trainer/frontend/src/App.test.tsx`

- [x] **Step 1: 替换手动输入表单**

在类别管理页把 `.class-form` 的名称输入、颜色输入和“创建类别”按钮替换为 `fieldset`：每个预设渲染一个带 `input type="checkbox"` 的固定选项，checkbox 的 `checked` 由 `selectedFixedClassNames.includes(preset.name)` 控制，`onChange` 只更新选择数组；已有同名类别旁显示“已启用”。增加按钮“启用所选类别”，禁用条件为未导入数据集、正在处理或选择数为 0。

- [x] **Step 2: 保留历史类别列表并调整文案**

将空列表文案改为“请先启用至少一个类别以绘制边界框。”，将类别库说明改为“选择本项目要标注的目标类别”。现有类别按钮、编辑功能和 `selectedClassId` 选择行为保持不变。

- [x] **Step 3: 增加预设卡片样式和响应式布局**

在 `styles.css` 中新增 `.fixed-class-picker`、`.fixed-class-option`、`.fixed-class-option.selected`、`.fixed-class-option input`、`.fixed-class-option-color` 和 `.fixed-class-option-status` 样式；使用两列网格展示六项，窄屏切换为单列，选中态使用现有青绿色边框与浅色背景，不新增渐变或大面积装饰。

- [x] **Step 4: 更新原有创建类别测试**

将原来输入 `类别名称` 并点击 `创建类别` 的测试改为勾选/取消固定类别并点击“启用所选类别”；保留“首个类别后进入标注”和“编辑已有类别”的回归覆盖。

- [x] **Step 5: 运行类别相关测试**

运行：

```bash
cd ~/DevProjects/YOLO_Trainer/frontend
npx vitest run src/App.test.tsx -t "类别|固定类别|首个类别|编辑"
```

预期：相关测试通过。

### Task 3: 全量验证

**Files:**
- Modify: none

- [x] **Step 1: 运行前端全量测试**

运行 `cd ~/DevProjects/YOLO_Trainer/frontend && npm test -- --run`，预期全部测试通过。

- [x] **Step 2: 运行前端构建**

运行 `cd ~/DevProjects/YOLO_Trainer/frontend && npm run build`，预期 TypeScript 检查和 Vite 构建成功。

- [x] **Step 3: 检查差异格式和状态**

运行 `cd ~/DevProjects/YOLO_Trainer && git diff --check && git status --short`，确认无空白错误，且不触碰用户已有的无关未跟踪文件。
