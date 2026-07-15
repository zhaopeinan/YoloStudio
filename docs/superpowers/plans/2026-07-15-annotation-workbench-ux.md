# YOLO Trainer 标注工作台体验优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将桌面工作流导航移动到顶部，并把图像标注页改造成适合连续快速标注的画布优先三栏工作台。

**Architecture:** `WorkflowShell` 只调整全局导航结构，不改变步骤状态和 hash 行为。`App` 继续拥有标注业务状态与画布事件，通过纯函数处理初始选图和“保存并下一张”，并抽出低耦合的 `AnnotationToolbar` 与 `AnnotationFilterDrawer`；画布和检查器保留在 `App` 内以避免重写现有 SVG 交互。

**Tech Stack:** React 18、TypeScript、Vite、Vitest、Testing Library、lucide-react、CSS Grid/Flexbox

---

## 文件结构

- Create: `frontend/src/annotation-workbench.ts`：初始选图、下一张未标注图像和高级筛选计数纯函数。
- Create: `frontend/src/annotation-workbench.test.ts`：标注工作台纯函数测试。
- Create: `frontend/src/AnnotationToolbar.tsx`：当前类别、图像导航、图层、保存状态和保存命令。
- Create: `frontend/src/AnnotationFilterDrawer.tsx`：高级筛选抽屉。
- Modify: `frontend/src/WorkflowShell.tsx`：桌面顶部横向步骤导航。
- Modify: `frontend/src/WorkflowShell.test.tsx`：顶部导航结构和精简步骤文案测试。
- Modify: `frontend/src/App.tsx`：接入新组件、连续保存流程、三栏检查器和移动标签。
- Modify: `frontend/src/App.test.tsx`：标注页结构、筛选抽屉、保存并下一张和失败保护测试。
- Modify: `frontend/src/styles.css`：顶部导航、固定高度三栏、抽屉、检查器和移动标签样式。

### Task 1: 将工作流导航移动到顶部

**Files:**
- Modify: `frontend/src/WorkflowShell.tsx`
- Modify: `frontend/src/WorkflowShell.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写顶部导航失败测试**

在 `frontend/src/WorkflowShell.test.tsx` 的第一个测试中增加结构断言：

```tsx
const shell = screen.getByTestId("workflow-shell");
expect(shell).toHaveClass("workflow-shell-top");
expect(screen.queryByRole("complementary")).not.toBeInTheDocument();

const navigation = screen.getByRole("navigation", { name: "工作流步骤" });
expect(navigation).toHaveClass("workflow-navigation-horizontal");
expect(navigation).toHaveTextContent("项目与数据");
expect(navigation).not.toHaveTextContent("扫描、导入或加载数据集");
```

- [ ] **Step 2: 运行导航测试并确认失败**

Run: `cd frontend && npm test -- --run src/WorkflowShell.test.tsx`

Expected: FAIL，缺少 `workflow-shell-top`、`workflow-navigation-horizontal`，且仍存在 `complementary` 左侧栏。

- [ ] **Step 3: 改造 `WorkflowShell` 结构**

将桌面导航移出 `aside`，放到现有移动选择器之前，并只显示编号、名称和状态：

```tsx
<div className="workflow-shell workflow-shell-top" data-testid="workflow-shell">
  <nav
    aria-label="工作流步骤"
    className="workflow-navigation workflow-navigation-horizontal"
  >
    {steps.map((step) => (
      <button
        type="button"
        key={step.id}
        className={step.id === currentStep ? "workflow-step active" : "workflow-step"}
        aria-current={step.id === currentStep ? "step" : undefined}
        data-availability={step.availability}
        title={step.lockedReason}
        onClick={() => onNavigate(step.id)}
      >
        <span className="workflow-step-number">{step.number}</span>
        <strong>{step.label}</strong>
        <span className={`workflow-step-status ${step.status}`}>
          {formatStepStatus(step)}
        </span>
      </button>
    ))}
  </nav>
```

在该导航之后原样保留当前 `.workflow-mobile-picker` 和 `.workflow-content` 节点，并删除包裹导航的 `.workflow-sidebar` 节点。

保留现有移动选择器、导航提示和非标注页前后步骤操作。

- [ ] **Step 4: 添加顶部导航样式**

在 `frontend/src/styles.css` 中将原两列外壳改成全宽结构：

```css
.workflow-shell-top {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
}

.workflow-navigation-horizontal {
  display: grid;
  grid-template-columns: repeat(6, minmax(150px, 1fr));
  gap: 6px;
  overflow-x: auto;
  padding: 4px 2px 8px;
}

.workflow-navigation-horizontal .workflow-step {
  grid-template-columns: 28px minmax(0, 1fr) auto;
  min-width: 150px;
  border-bottom: 3px solid transparent;
}

.workflow-navigation-horizontal .workflow-step.active {
  border-bottom-color: #1f6f78;
}

@media (max-width: 820px) {
  .workflow-navigation-horizontal { display: none; }
}
```

删除或覆盖只服务旧 `.workflow-sidebar` 两列布局的规则。

- [ ] **Step 5: 运行导航测试并提交**

Run: `cd frontend && npm test -- --run src/WorkflowShell.test.tsx`

Expected: 3 tests passed。

```bash
git add frontend/src/WorkflowShell.tsx frontend/src/WorkflowShell.test.tsx frontend/src/styles.css
git commit -m "feat: move workflow navigation to top"
```

### Task 2: 建立连续标注选择逻辑

**Files:**
- Create: `frontend/src/annotation-workbench.ts`
- Create: `frontend/src/annotation-workbench.test.ts`

- [ ] **Step 1: 编写选图纯函数失败测试**

创建 `frontend/src/annotation-workbench.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import {
  countAdvancedImageFilters,
  findNextAnnotationImageId,
  selectInitialAnnotationImageId,
} from "./annotation-workbench";

const images = [
  { id: 1, annotation_count: 1 },
  { id: 2, annotation_count: 0 },
  { id: 3, annotation_count: 1 },
  { id: 4, annotation_count: 0 },
];

describe("annotation workbench selection", () => {
  it("保持仍存在的当前图像，否则优先第一张未标注图像", () => {
    expect(selectInitialAnnotationImageId(images, 3)).toBe(3);
    expect(selectInitialAnnotationImageId(images, 99)).toBe(2);
    expect(selectInitialAnnotationImageId([{ id: 1, annotation_count: 1 }], null)).toBe(1);
  });

  it("优先查找当前位置后的未标注图像，再退回普通下一张", () => {
    expect(findNextAnnotationImageId(images, 1)).toBe(2);
    expect(findNextAnnotationImageId(images, 2)).toBe(4);
    expect(findNextAnnotationImageId(images, 4)).toBeNull();
    expect(findNextAnnotationImageId(images.slice(0, 3), 2)).toBe(3);
  });

  it("只统计高级筛选条件", () => {
    expect(countAdvancedImageFilters({
      platform: "vtol",
      class_id: "2",
      edge_tag: "occluded",
      failure_type: "all",
      altitude_min: "",
      altitude_max: "30",
    })).toBe(4);
  });
});
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `cd frontend && npm test -- --run src/annotation-workbench.test.ts`

Expected: FAIL，无法找到 `./annotation-workbench`。

- [ ] **Step 3: 实现纯函数**

创建 `frontend/src/annotation-workbench.ts`：

```ts
type SelectableImage = { id: number; annotation_count: number };

export type AdvancedImageFilters = {
  platform: string;
  class_id: string;
  edge_tag: string;
  failure_type: string;
  altitude_min: string;
  altitude_max: string;
};

export function selectInitialAnnotationImageId(
  images: SelectableImage[],
  currentId: number | null,
): number | null {
  if (images.some((image) => image.id === currentId)) return currentId;
  return images.find((image) => image.annotation_count === 0)?.id ?? images[0]?.id ?? null;
}

export function findNextAnnotationImageId(
  images: SelectableImage[],
  currentId: number,
): number | null {
  const currentIndex = images.findIndex((image) => image.id === currentId);
  if (currentIndex < 0) return selectInitialAnnotationImageId(images, null);
  const remaining = images.slice(currentIndex + 1);
  return remaining.find((image) => image.annotation_count === 0)?.id ?? remaining[0]?.id ?? null;
}

export function countAdvancedImageFilters(filters: AdvancedImageFilters): number {
  return Object.entries(filters).filter(([key, value]) =>
    key === "failure_type" ? value !== "all" : value.trim() !== "",
  ).length;
}
```

- [ ] **Step 4: 运行纯函数测试并提交**

Run: `cd frontend && npm test -- --run src/annotation-workbench.test.ts`

Expected: 3 tests passed。

```bash
git add frontend/src/annotation-workbench.ts frontend/src/annotation-workbench.test.ts
git commit -m "feat: add continuous annotation selection rules"
```

### Task 3: 创建标注工具栏和高级筛选抽屉

**Files:**
- Create: `frontend/src/AnnotationToolbar.tsx`
- Create: `frontend/src/AnnotationFilterDrawer.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 编写工具栏与抽屉失败测试**

在 `frontend/src/App.test.tsx` 的已导入数据集场景新增测试：

```tsx
it("以核心工具栏和高级筛选抽屉组织标注操作", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");

  expect(screen.getByLabelText("标注工具栏")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "保存并下一张" })).toBeInTheDocument();
  expect(screen.queryByLabelText("标注就绪状态")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("图像高级筛选")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: /高级筛选/ }));
  expect(screen.getByLabelText("图像高级筛选")).toBeInTheDocument();
  expect(screen.getByLabelText("平台")).toBeInTheDocument();
});
```

- [ ] **Step 2: 运行目标测试并确认失败**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "核心工具栏和高级筛选抽屉"`

Expected: FAIL，找不到“标注工具栏”和“保存并下一张”。

- [ ] **Step 3: 创建 `AnnotationToolbar`**

组件 props 明确承载展示状态和命令，不拥有业务数据：

```tsx
type AnnotationToolbarProps = {
  classes: ProjectClass[];
  selectedClassId: number | null;
  selectedImageIndex: number;
  imageCount: number;
  annotationsDirty: boolean;
  isSaving: boolean;
  error: string | null;
  canGoPrevious: boolean;
  canGoNext: boolean;
  showReviewLayers: boolean;
  showGroundTruth: boolean;
  showPrediction: boolean;
  onClassChange: (classId: number) => void;
  onPrevious: () => void;
  onNext: () => void;
  onSave: () => void;
  onSaveAndNext: () => void;
  canCopyPrevious: boolean;
  canCopyNext: boolean;
  onCopyPrevious: () => void;
  onCopyNext: () => void;
  onGroundTruthChange: (checked: boolean) => void;
  onPredictionChange: (checked: boolean) => void;
};
```

根节点使用 `aria-label="标注工具栏"`。保存状态文案按优先级计算：错误、保存中、有未保存修改、已保存。主按钮名称固定为“保存并下一张”。上一张和下一张使用 lucide `ChevronLeft`、`ChevronRight` 图标并保留准确 `aria-label`。增加 `MoreHorizontal` 图标按钮控制工具栏内的命令菜单，菜单包含“复制上一张标注”和“复制下一张标注”，分别使用上述 `canCopy*` 与 `onCopy*` props。

- [ ] **Step 4: 创建 `AnnotationFilterDrawer`**

组件接收高级筛选值、类别、预测失败类型选项和更新回调。使用原生 `<dialog>` 以外的受控侧边层，避免测试环境兼容问题：

```tsx
<section className="annotation-filter-drawer" aria-label="图像高级筛选">
  <header>
    <strong>高级筛选</strong>
    <button type="button" className="icon-button" aria-label="关闭高级筛选" onClick={onClose}>
      <X size={18} />
    </button>
  </header>
  <label htmlFor="advanced-filter-platform">平台</label>
  <input id="advanced-filter-platform" value={filters.platform} onChange={(event) => onChange("platform", event.target.value)} />
  <label htmlFor="advanced-filter-class">类别</label>
  <select id="advanced-filter-class" value={filters.class_id} onChange={(event) => onChange("class_id", event.target.value)}>
    <option value="">全部类别</option>
    {classes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
  </select>
  <label htmlFor="advanced-filter-result">识别结果</label>
  <select id="advanced-filter-result" value={filters.failure_type} onChange={(event) => onChange("failure_type", event.target.value)}>
    {failureOptions.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
  </select>
  <label htmlFor="advanced-filter-edge-tag">边缘案例标签</label>
  <input id="advanced-filter-edge-tag" value={filters.edge_tag} onChange={(event) => onChange("edge_tag", event.target.value)} />
  <div className="range-row">
    <label htmlFor="advanced-filter-altitude-min">最低高度<input id="advanced-filter-altitude-min" type="number" value={filters.altitude_min} onChange={(event) => onChange("altitude_min", event.target.value)} /></label>
    <label htmlFor="advanced-filter-altitude-max">最高高度<input id="advanced-filter-altitude-max" type="number" value={filters.altitude_max} onChange={(event) => onChange("altitude_max", event.target.value)} /></label>
  </div>
  <footer>
    <button type="button" className="secondary-button" onClick={onReset}>清除筛选</button>
    <button type="button" onClick={onApply}>应用筛选</button>
  </footer>
</section>
```

- [ ] **Step 5: 在 `App` 接入基础筛选与抽屉状态**

新增：

```ts
const [isImageFilterDrawerOpen, setIsImageFilterDrawerOpen] = useState(false);
const advancedFilterCount = countAdvancedImageFilters(imageFilters);
```

左栏只保留文件名搜索输入、标注状态、带计数的“高级筛选”按钮、分页和图像列表。文件名搜索在前端使用 `relative_path` 的小写包含匹配，不改变后端请求；其余条件继续调用现有 `listImages`。

删除 `annotationReadinessSteps` 展示，保留简化后的 `annotationReady` 布尔计算。

- [ ] **Step 6: 运行目标测试并提交**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "核心工具栏和高级筛选抽屉"`

Expected: PASS。

```bash
git add frontend/src/AnnotationToolbar.tsx frontend/src/AnnotationFilterDrawer.tsx frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: add focused annotation controls"
```

### Task 4: 实现三栏画布和当前框检查器

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写检查器失败测试**

在现有边界框类别修改测试旁增加：

```tsx
it("只在检查器中展开当前边界框", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");

  expect(screen.getByLabelText("边界框检查器")).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: /选择边界框/ })).toHaveLength(1);
  expect(screen.queryByLabelText("归一化坐标")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: /选择边界框/ }));
  await user.click(screen.getByRole("button", { name: "展开坐标参数" }));
  expect(screen.getByLabelText("归一化坐标")).toBeInTheDocument();
});
```

- [ ] **Step 2: 运行检查器测试并确认失败**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "只在检查器中展开当前边界框"`

Expected: FAIL，缺少“边界框检查器”和折叠坐标入口。

- [ ] **Step 3: 重组标注 JSX**

将现有 `.annotation-class-selector` 和 `.annotation-workbench` 两个顶层区块合并为单个 `<section className="annotation-workspace" aria-label="标注工作台">`。其第一个子节点是 `AnnotationToolbar`；第二个子节点 `.annotation-workspace-grid` 依次包裹现有图像浏览内容、画布内容和边界框编辑内容，并分别使用 `.annotation-browser-pane`、`.annotation-canvas-pane`、`.annotation-inspector-pane`。`AnnotationFilterDrawer` 作为该 section 的最后一个受控子节点，仅在 `isImageFilterDrawerOpen` 为真时渲染。

新增 `selectedAnnotation` memo，通过 `selectedAnnotationId` 获取当前框。右栏顶部渲染紧凑框列表，完整表单只绑定 `selectedAnnotation`。使用受控按钮展开归一化坐标：

```tsx
const [showAnnotationGeometry, setShowAnnotationGeometry] = useState(false);

<button type="button" className="inspector-disclosure" onClick={() => setShowAnnotationGeometry((v) => !v)}>
  {showAnnotationGeometry ? "收起坐标参数" : "展开坐标参数"}
</button>
{showAnnotationGeometry ? (
  <div className="geometry-grid" aria-label="归一化坐标">
    {(["x_center", "y_center", "width", "height"] as const).map((field) => (
      <label key={field}>
        {field === "x_center" ? "X" : field === "y_center" ? "Y" : field === "width" ? "W" : "H"}
        <input
          type="number"
          step="0.001"
          value={selectedAnnotation[field]}
          onChange={(event) => updateAnnotationGeometry(selectedAnnotation.local_id, field, Number(event.target.value))}
        />
      </label>
    ))}
  </div>
) : null}
```

边缘标签、轨迹 ID、类别、删除和微调继续调用现有处理函数。预测审查摘要移动到检查器底部，并保留现有图层和修正逻辑。

- [ ] **Step 4: 添加固定高度三栏样式**

```css
.annotation-workspace {
  position: relative;
  display: grid;
  gap: 10px;
  min-width: 0;
}

.annotation-workspace-grid {
  display: grid;
  grid-template-columns: minmax(240px, 280px) minmax(420px, 1fr) minmax(280px, 320px);
  min-height: 640px;
  height: calc(100vh - 330px);
  max-height: 820px;
  gap: 10px;
}

.annotation-browser-pane,
.annotation-canvas-pane,
.annotation-inspector-pane {
  min-width: 0;
  overflow: auto;
  border: 1px solid #d9e3ea;
  border-radius: 8px;
  background: #fff;
}

.annotation-canvas-pane {
  display: grid;
  place-items: center;
  background: #111820;
}
```

画布图像约束使用容器可用尺寸，不裁切：`max-width: 100%`、`max-height: 100%`、`object-fit: contain`。

- [ ] **Step 5: 运行标注相关测试并提交**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "annotation|标注|边界框"`

Expected: 所有匹配测试通过。

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: build canvas-first annotation workspace"
```

### Task 5: 实现保存并下一张及移动端标签

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写保存成功和失败测试**

```tsx
it("保存成功后前往下一张未标注图像", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");

  await user.click(screen.getByRole("button", { name: "保存并下一张" }));
  await flushPromises();

  expect(apiMock.replaceAnnotations).toHaveBeenCalled();
  expect(screen.getByRole("img", { name: "iris/frame002.jpg" })).toBeInTheDocument();
});

it("保存失败时保留当前图像和草稿", async () => {
  apiMock.replaceAnnotations.mockRejectedValueOnce(new Error("磁盘写入失败"));
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");
  const currentImage = screen.getByRole("img", { name: "iris/frame001.jpg" });

  await user.click(screen.getByRole("button", { name: "保存并下一张" }));
  await flushPromises();

  expect(currentImage).toBeInTheDocument();
  expect(screen.getByText("磁盘写入失败")).toBeInTheDocument();
});
```

- [ ] **Step 2: 让保存函数返回成功状态**

将 `handleSaveAnnotations` 改为 `Promise<boolean>`：成功路径末尾 `return true`，失败路径 `return false`。所有原“保存标注”按钮继续用 `void handleSaveAnnotations()`。

新增：

```ts
async function handleSaveAndNextImage() {
  if (!selectedImageId) return;
  const saved = await handleSaveAnnotations();
  if (!saved) return;
  const nextId = findNextAnnotationImageId(images, selectedImageId);
  if (nextId) {
    setSelectedImageId(nextId);
    setAnnotationNotice(null);
  } else {
    setAnnotationNotice("当前列表已完成");
  }
}
```

新增 `annotationNotice` 状态并在工具栏下方显示。手动图像导航复用现有未保存确认，不绕过草稿保护。

- [ ] **Step 3: 编写移动标签测试**

```tsx
it("提供移动端标注视图标签并保持选中框状态", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");

  const tabs = screen.getByRole("tablist", { name: "标注视图" });
  expect(within(tabs).getByRole("tab", { name: "画布" })).toHaveAttribute("aria-selected", "true");
  await user.click(within(tabs).getByRole("tab", { name: "属性" }));
  expect(screen.getByLabelText("边界框检查器")).toBeInTheDocument();
});
```

- [ ] **Step 4: 实现移动标签和响应式显示**

新增状态：

```ts
type AnnotationMobileView = "canvas" | "images" | "inspector";
const [annotationMobileView, setAnnotationMobileView] = useState<AnnotationMobileView>("canvas");
```

渲染 `role="tablist" aria-label="标注视图"`，三个按钮分别设置 `aria-selected` 和 `data-mobile-active`。CSS 在桌面隐藏标签；`820px` 以下显示标签，并只显示匹配 `data-pane` 的工作区栏。

- [ ] **Step 5: 运行保存和移动测试并提交**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "保存成功|保存失败|移动端标注视图"`

Expected: 3 tests passed。

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: streamline continuous annotation flow"
```

### Task 6: 完整回归与浏览器验收

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 运行前端完整测试**

Run: `cd frontend && npm test -- --run`

Expected: 全部测试通过，无未处理 promise、React `act` 或可访问性相关错误。

- [ ] **Step 2: 运行生产构建**

Run: `cd frontend && npm run build`

Expected: TypeScript 与 Vite 构建成功。

- [ ] **Step 3: 运行后端回归测试**

Run: `cd backend && .venv/bin/pytest`

Expected: 现有后端测试全部通过；本任务不改变后端代码。

- [ ] **Step 4: 验证桌面标注工作流**

在 `http://127.0.0.1:5173/#annotation` 验证：

- 顶部显示六步横向导航，页面左侧不存在工作流栏。
- 首屏显示标注工具栏和完整三栏，画布宽于左右栏。
- 高级筛选默认关闭，打开、应用和清除均正常。
- 画框后显示“有未保存修改”，保存后显示“已保存”。
- “保存并下一张”成功切换到下一张未标注图像。
- 边界框列表、画布选框和右侧检查器同步。
- 长文件名使用省略显示且不撑破图像栏。

- [ ] **Step 5: 验证 `390px` 移动布局**

验证移动步骤选择器、画布/图像/属性标签、按钮文字、属性输入和图像列表无重叠或横向溢出。确认切换标签后草稿与选中框保持。

- [ ] **Step 6: 检查代码差异并提交收尾修复**

Run: `git diff --check && git status --short`

Expected: 无空白错误；只包含本计划文件和已知未跟踪文件。

存在浏览器验收修复时：

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css frontend/src/AnnotationToolbar.tsx frontend/src/AnnotationFilterDrawer.tsx frontend/src/annotation-workbench.ts frontend/src/annotation-workbench.test.ts frontend/src/WorkflowShell.tsx frontend/src/WorkflowShell.test.tsx
git commit -m "fix: polish annotation workbench usability"
```
