# YOLO Trainer 分步工作流导航实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将纵向堆叠的 YOLO Trainer 单页工作台改造成六步导航，每次只展示一个职责明确的业务页面，并保留现有标注、训练、预测和导出行为。

**Architecture:** 新增纯函数步骤状态模块和独立导航外壳，`App` 继续持有现有业务状态及 API 调用。现有六组 JSX 通过当前步骤条件渲染；URL hash 记录步骤，导航请求统一处理锁定原因和未保存标注确认。

**Tech Stack:** React 18、TypeScript、Vite、Vitest、Testing Library、lucide-react、CSS Grid/Flexbox

---

## 文件结构

- Create: `frontend/src/workflow.ts`：步骤类型、元数据、状态计算、锁定原因和 hash 解析。
- Create: `frontend/src/workflow.test.ts`：纯函数状态与 hash 测试。
- Create: `frontend/src/WorkflowShell.tsx`：全局外壳、桌面步骤导航、移动步骤选择器、前后步骤按钮和导航提示。
- Create: `frontend/src/WorkflowShell.test.tsx`：导航可用性、锁定点击和移动选择测试。
- Modify: `frontend/src/App.tsx`：接入步骤状态，只渲染当前页面，增加自动推进、URL hash、失败样本跳转和未保存标注保护。
- Modify: `frontend/src/App.test.tsx`：将原有全页面测试改为按步骤导航，并新增关键工作流回归测试。
- Modify: `frontend/src/styles.css`：工作流外壳、侧边导航、移动选择器、单页内容和标注页布局。

### Task 1: 建立步骤状态模型

**Files:**
- Create: `frontend/src/workflow.ts`
- Create: `frontend/src/workflow.test.ts`

- [ ] **Step 1: 编写步骤状态失败测试**

创建 `frontend/src/workflow.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import {
  buildWorkflowSteps,
  parseWorkflowHash,
  type WorkflowProgress,
} from "./workflow";

const emptyProgress: WorkflowProgress = {
  hasDataset: false,
  classCount: 0,
  annotatedImageCount: 0,
  versionCount: 0,
  runCount: 0,
  predictionJobCount: 0,
  exportCount: 0,
};

describe("工作流步骤", () => {
  it("只开放项目与数据步骤", () => {
    const steps = buildWorkflowSteps(emptyProgress);
    expect(steps.map((step) => [step.id, step.availability, step.status])).toEqual([
      ["dataset", "available", "in_progress"],
      ["classes", "locked", "not_started"],
      ["annotation", "locked", "not_started"],
      ["quality", "locked", "not_started"],
      ["training", "locked", "not_started"],
      ["evaluation", "locked", "not_started"],
    ]);
    expect(steps[1].lockedReason).toBe("请先导入或加载数据集");
  });

  it("根据业务证据推进步骤状态", () => {
    const steps = buildWorkflowSteps({
      hasDataset: true,
      classCount: 2,
      annotatedImageCount: 8,
      versionCount: 1,
      runCount: 1,
      predictionJobCount: 1,
      exportCount: 0,
    });
    expect(steps.map((step) => step.status)).toEqual([
      "completed",
      "completed",
      "completed",
      "completed",
      "completed",
      "in_progress",
    ]);
    expect(steps.every((step) => step.availability === "available")).toBe(true);
  });

  it("解析有效步骤 hash 并拒绝未知值", () => {
    expect(parseWorkflowHash("#annotation")).toBe("annotation");
    expect(parseWorkflowHash("#unknown")).toBe("dataset");
    expect(parseWorkflowHash("")).toBe("dataset");
  });
});
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `cd frontend && npm test -- --run src/workflow.test.ts`

Expected: FAIL，提示无法找到 `./workflow`。

- [ ] **Step 3: 实现步骤类型和状态计算**

创建 `frontend/src/workflow.ts`：

```ts
export type WorkflowStep =
  | "dataset"
  | "classes"
  | "annotation"
  | "quality"
  | "training"
  | "evaluation";

export type WorkflowStepStatus = "not_started" | "in_progress" | "completed";
export type WorkflowStepAvailability = "available" | "locked";

export type WorkflowProgress = {
  hasDataset: boolean;
  classCount: number;
  annotatedImageCount: number;
  versionCount: number;
  runCount: number;
  predictionJobCount: number;
  exportCount: number;
};

export type WorkflowStepItem = {
  id: WorkflowStep;
  number: number;
  label: string;
  description: string;
  availability: WorkflowStepAvailability;
  status: WorkflowStepStatus;
  lockedReason?: string;
};

export const workflowStepOrder: WorkflowStep[] = [
  "dataset",
  "classes",
  "annotation",
  "quality",
  "training",
  "evaluation",
];

const stepCopy: Record<WorkflowStep, Pick<WorkflowStepItem, "label" | "description">> = {
  dataset: { label: "项目与数据", description: "扫描、导入或加载数据集" },
  classes: { label: "类别管理", description: "维护项目级检测类别" },
  annotation: { label: "图像标注", description: "绘制、修正并保存边界框" },
  quality: { label: "质量与版本", description: "审查数据并冻结训练版本" },
  training: { label: "模型训练", description: "配置、启动和监控训练" },
  evaluation: { label: "评估与导出", description: "分析预测并导出模型" },
};

function item(
  id: WorkflowStep,
  number: number,
  availability: WorkflowStepAvailability,
  status: WorkflowStepStatus,
  lockedReason?: string,
): WorkflowStepItem {
  return { id, number, ...stepCopy[id], availability, status, lockedReason };
}

export function buildWorkflowSteps(progress: WorkflowProgress): WorkflowStepItem[] {
  const hasClasses = progress.classCount > 0;
  const hasAnnotations = progress.annotatedImageCount > 0;
  const hasVersions = progress.versionCount > 0;
  const hasRuns = progress.runCount > 0;
  const hasEvaluation = progress.predictionJobCount > 0 || progress.exportCount > 0;

  return [
    item("dataset", 1, "available", progress.hasDataset ? "completed" : "in_progress"),
    item(
      "classes",
      2,
      progress.hasDataset ? "available" : "locked",
      hasClasses ? "completed" : progress.hasDataset ? "in_progress" : "not_started",
      progress.hasDataset ? undefined : "请先导入或加载数据集",
    ),
    item(
      "annotation",
      3,
      progress.hasDataset && hasClasses ? "available" : "locked",
      hasAnnotations
        ? "completed"
        : progress.hasDataset && hasClasses
          ? "in_progress"
          : "not_started",
      !progress.hasDataset ? "请先导入或加载数据集" : !hasClasses ? "请先创建至少一个类别" : undefined,
    ),
    item(
      "quality",
      4,
      progress.hasDataset ? "available" : "locked",
      hasVersions ? "completed" : progress.hasDataset ? "in_progress" : "not_started",
      progress.hasDataset ? undefined : "请先导入或加载数据集",
    ),
    item(
      "training",
      5,
      hasVersions ? "available" : "locked",
      hasRuns ? "completed" : hasVersions ? "in_progress" : "not_started",
      hasVersions ? undefined : "请先创建冻结数据集版本",
    ),
    item(
      "evaluation",
      6,
      hasRuns ? "available" : "locked",
      hasEvaluation ? "completed" : hasRuns ? "in_progress" : "not_started",
      hasRuns ? undefined : "请先创建训练任务",
    ),
  ];
}

export function parseWorkflowHash(hash: string): WorkflowStep {
  const value = hash.replace(/^#/, "") as WorkflowStep;
  return workflowStepOrder.includes(value) ? value : "dataset";
}
```

- [ ] **Step 4: 运行步骤测试**

Run: `cd frontend && npm test -- --run src/workflow.test.ts`

Expected: 3 tests passed。

- [ ] **Step 5: 提交步骤模型**

```bash
git add frontend/src/workflow.ts frontend/src/workflow.test.ts
git commit -m "feat: model guided workflow steps"
```

### Task 2: 创建桌面和移动导航外壳

**Files:**
- Create: `frontend/src/WorkflowShell.tsx`
- Create: `frontend/src/WorkflowShell.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写导航外壳失败测试**

创建 `frontend/src/WorkflowShell.test.tsx`，使用 `buildWorkflowSteps` 生成无数据集状态并断言：

```tsx
import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WorkflowShell } from "./WorkflowShell";
import { buildWorkflowSteps } from "./workflow";

describe("WorkflowShell", () => {
  it("展示步骤状态并把锁定点击交给调用方", () => {
    const onNavigate = vi.fn();
    render(
      <WorkflowShell
        currentStep="dataset"
        steps={buildWorkflowSteps({
          hasDataset: false,
          classCount: 0,
          annotatedImageCount: 0,
          versionCount: 0,
          runCount: 0,
          predictionJobCount: 0,
          exportCount: 0,
        })}
        navigationNotice={null}
        onNavigate={onNavigate}
      >
        <p>数据集页面内容</p>
      </WorkflowShell>,
    );

    expect(screen.getByRole("navigation", { name: "工作流步骤" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /1 项目与数据/ })).toHaveAttribute(
      "aria-current",
      "step",
    );
    fireEvent.click(screen.getByRole("button", { name: /2 类别管理/ }));
    expect(onNavigate).toHaveBeenCalledWith("classes");
    expect(screen.getByText("数据集页面内容")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: 运行测试并确认组件不存在**

Run: `cd frontend && npm test -- --run src/WorkflowShell.test.tsx`

Expected: FAIL，提示无法找到 `./WorkflowShell`。

- [ ] **Step 3: 实现导航外壳**

`WorkflowShell` props 固定为：

```ts
type WorkflowShellProps = {
  currentStep: WorkflowStep;
  steps: WorkflowStepItem[];
  navigationNotice: string | null;
  onNavigate: (step: WorkflowStep) => void;
  children: ReactNode;
};
```

组件必须包含：

```tsx
<div className="workflow-shell">
  <aside className="workflow-sidebar">
    <nav aria-label="工作流步骤" className="workflow-navigation">
      {steps.map((step) => (
        <button
          type="button"
          key={step.id}
          className={step.id === currentStep ? "workflow-step active" : "workflow-step"}
          aria-current={step.id === currentStep ? "step" : undefined}
          aria-disabled={step.availability === "locked"}
          onClick={() => onNavigate(step.id)}
        >
          <span className="workflow-step-number">{step.number}</span>
          <span className="workflow-step-copy">
            <strong>{step.label}</strong>
            <small>{step.description}</small>
          </span>
          <span className={`workflow-step-status ${step.status}`}>
            {step.availability === "locked"
              ? "暂不可用"
              : step.status === "completed"
                ? "已完成"
                : step.status === "in_progress"
                  ? "进行中"
                  : "未开始"}
          </span>
        </button>
      ))}
    </nav>
  </aside>

  <div className="workflow-mobile-picker">
    <label htmlFor="workflow-step-select">当前步骤</label>
    <select
      id="workflow-step-select"
      value={currentStep}
      onChange={(event) => onNavigate(event.target.value as WorkflowStep)}
    >
      {steps.map((step) => (
        <option key={step.id} value={step.id}>
          {step.number}. {step.label} · {step.availability === "locked" ? "暂不可用" : "可进入"}
        </option>
      ))}
    </select>
  </div>

  <section className="workflow-content" aria-label="当前工作流页面">
    {navigationNotice ? <div className="warning-banner">{navigationNotice}</div> : null}
    {children}
    {currentStep !== "annotation" ? (
      <footer className="workflow-page-actions">
        {currentIndex > 0 ? (
          <button
            type="button"
            className="secondary-button"
            onClick={() => onNavigate(steps[currentIndex - 1].id)}
          >
            上一步：{steps[currentIndex - 1].label}
          </button>
        ) : <span />}
        {currentIndex < steps.length - 1 ? (
          <button type="button" onClick={() => onNavigate(steps[currentIndex + 1].id)}>
            下一步：{steps[currentIndex + 1].label}
          </button>
        ) : null}
      </footer>
    ) : null}
  </section>
</div>
```

其中 `currentIndex` 为 `steps.findIndex((step) => step.id === currentStep)`。下一步即使锁定也保留可点击，由 `App` 的统一导航处理显示解锁原因；标注页不显示通用页脚，以免干扰连续标注操作。

- [ ] **Step 4: 添加外壳样式**

在 `frontend/src/styles.css` 添加桌面两列布局、固定侧栏、步骤按钮稳定高度和移动选择器：

```css
.workflow-shell {
  display: grid;
  grid-template-columns: minmax(220px, 260px) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.workflow-sidebar {
  position: sticky;
  top: 16px;
  min-width: 0;
}

.workflow-navigation {
  display: grid;
  gap: 6px;
}

.workflow-step {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  min-height: 68px;
  width: 100%;
  padding: 10px;
  color: #172026;
  text-align: left;
  background: transparent;
}

.workflow-step.active {
  background: #dff6f2;
  box-shadow: inset 3px 0 #1f6f78;
}

.workflow-step[aria-disabled="true"] {
  opacity: 0.58;
}

.workflow-step-copy,
.workflow-step-copy strong,
.workflow-step-copy small {
  display: block;
  min-width: 0;
}

.workflow-mobile-picker {
  display: none;
}

.workflow-content {
  min-width: 0;
}

.workflow-page-actions {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 16px;
}

@media (max-width: 820px) {
  .workflow-shell {
    grid-template-columns: minmax(0, 1fr);
  }

  .workflow-sidebar {
    display: none;
  }

  .workflow-mobile-picker {
    display: grid;
    gap: 6px;
  }
}
```

- [ ] **Step 5: 运行导航测试和构建**

Run: `cd frontend && npm test -- --run src/WorkflowShell.test.tsx && npm run build`

Expected: 导航测试通过，生产构建成功。

- [ ] **Step 6: 提交导航外壳**

```bash
git add frontend/src/WorkflowShell.tsx frontend/src/WorkflowShell.test.tsx frontend/src/styles.css
git commit -m "feat: add guided workflow shell"
```

### Task 3: 将 App 重组为单步骤页面

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 将主测试改为单页可见性测试**

在 `App.test.tsx` 初始测试中断言：

```ts
expect(await screen.findByRole("heading", { name: "扫描本地数据集" })).toBeInTheDocument();
expect(screen.queryByRole("heading", { name: "类别库" })).not.toBeInTheDocument();
expect(screen.queryByRole("heading", { name: "标注" })).not.toBeInTheDocument();
expect(screen.queryByRole("heading", { name: "训练设置" })).not.toBeInTheDocument();
expect(screen.queryByRole("heading", { name: "预测分析" })).not.toBeInTheDocument();
```

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"`

Expected: FAIL，因为当前页面仍同时渲染所有模块。

- [ ] **Step 2: 在 App 中建立工作流状态**

导入模块：

```ts
import { WorkflowShell } from "./WorkflowShell";
import {
  buildWorkflowSteps,
  parseWorkflowHash,
  type WorkflowStep,
} from "./workflow";
```

添加状态与计算：

```ts
const [activeWorkflowStep, setActiveWorkflowStep] = useState<WorkflowStep>(() =>
  parseWorkflowHash(window.location.hash),
);
const [navigationNotice, setNavigationNotice] = useState<string | null>(null);

const workflowSteps = useMemo(
  () =>
    buildWorkflowSteps({
      hasDataset: Boolean(importedDataset),
      classCount: classes.length,
      annotatedImageCount: coverage?.annotated_image_count ?? 0,
      versionCount: versions.length,
      runCount: runs.length,
      predictionJobCount: predictionJobs.length,
      exportCount: exports.length,
    }),
  [importedDataset, classes.length, coverage, versions.length, runs.length, predictionJobs.length, exports.length],
);
```

- [ ] **Step 3: 添加统一步骤导航处理**

```ts
function navigateToWorkflowStep(stepId: WorkflowStep) {
  const step = workflowSteps.find((item) => item.id === stepId);
  if (!step || step.availability === "locked") {
    setNavigationNotice(step?.lockedReason ?? "该步骤暂不可用");
    return;
  }
  setNavigationNotice(null);
  setActiveWorkflowStep(stepId);
  window.history.replaceState(null, "", `#${stepId}`);
}
```

添加 `hashchange` 监听；有效且开放的 hash 正常切换，锁定 hash 保持当前页面并显示原因，无效 hash 回退到 `dataset`。

- [ ] **Step 4: 用 WorkflowShell 包裹现有页面**

保留顶部产品栏和紧凑状态条。将六个现有 JSX 区域分别包入条件：

```tsx
<WorkflowShell
  currentStep={activeWorkflowStep}
  steps={workflowSteps}
  navigationNotice={navigationNotice}
  onNavigate={navigateToWorkflowStep}
>
  {activeWorkflowStep === "dataset" ? datasetStepContent : null}
  {activeWorkflowStep === "classes" ? classesStepContent : null}
  {activeWorkflowStep === "annotation" ? annotationStepContent : null}
  {activeWorkflowStep === "quality" ? qualityStepContent : null}
  {activeWorkflowStep === "training" ? trainingStepContent : null}
  {activeWorkflowStep === "evaluation" ? evaluationStepContent : null}
</WorkflowShell>
```

具体归属：

- `datasetStepContent`：扫描和导入面板。
- `classesStepContent`：类别库面板。
- `annotationStepContent`：图像浏览器和标注面板；类别列表改为画布上方紧凑选择条。
- `qualityStepContent`：质量审查、覆盖情况和版本导出。
- `trainingStepContent`：训练设置和训练记录。
- `evaluationStepContent`：预测分析、实验看板和模型导出。

- [ ] **Step 5: 更新主集成测试的导航顺序**

原主流程在导入后按以下顺序显式进入步骤：

```ts
await user.click(screen.getByRole("button", { name: /2 类别管理/ }));
expect(await screen.findByRole("heading", { name: "类别库" })).toBeInTheDocument();

await user.click(screen.getByRole("button", { name: /3 图像标注/ }));
expect(await screen.findByRole("heading", { name: "标注" })).toBeInTheDocument();

await user.click(screen.getByRole("button", { name: /4 质量与版本/ }));
expect(await screen.findByRole("heading", { name: "质量审查" })).toBeInTheDocument();

await user.click(screen.getByRole("button", { name: /5 模型训练/ }));
expect(await screen.findByRole("heading", { name: "训练设置" })).toBeInTheDocument();

await user.click(screen.getByRole("button", { name: /6 评估与导出/ }));
expect(await screen.findByRole("heading", { name: "预测分析" })).toBeInTheDocument();
```

- [ ] **Step 6: 运行 App 测试并修复仅由页面拆分引起的定位变化**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: 现有 15 个 App 测试通过。

- [ ] **Step 7: 提交单步骤页面重组**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: split workbench into guided steps"
```

### Task 4: 添加自动推进和 URL hash 恢复

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 添加自动推进测试**

新增测试覆盖：

```ts
it("导入数据集后进入类别管理，创建首个类别后进入图像标注", async () => {
  const user = userEvent.setup();
  render(<App />);

  await user.click(await screen.findByRole("button", { name: "导入数据集" }));
  expect(await screen.findByRole("heading", { name: "类别库" })).toBeInTheDocument();

  await user.type(screen.getByLabelText("类别名称"), "target");
  await user.click(screen.getByRole("button", { name: "创建类别" }));
  expect(await screen.findByRole("heading", { name: "标注" })).toBeInTheDocument();
});
```

新增 hash 测试：先加载数据集，再设置 `window.location.hash = "#quality"` 并触发 `hashchange`，断言质量页面恢复；设置 `#training` 但没有版本时，断言保持当前页面且显示“请先创建冻结数据集版本”。无效 hash 断言回退到项目与数据。

- [ ] **Step 2: 运行新增测试并确认失败**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "进入类别管理|hash"`

Expected: FAIL，尚未自动推进或恢复步骤。

- [ ] **Step 3: 在成功处理函数中推进步骤**

在数据集导入和加载成功后调用：

```ts
setActiveWorkflowStep("classes");
window.history.replaceState(null, "", "#classes");
```

创建类别前记录 `const isFirstClass = classes.length === 0`；成功后仅在 `isFirstClass` 时进入 `annotation`。

- [ ] **Step 4: 实现 hash 恢复与回退**

监听浏览器 `hashchange` 并复用 `navigateToWorkflowStep`：开放步骤正常切换，锁定步骤保持当前页面并显示 `lockedReason`，无效 hash 回退到 `dataset`。组件卸载时清理监听器。页面刷新后若尚未加载数据集，依设计将锁定 hash 回退到项目与数据。

- [ ] **Step 5: 运行自动推进和完整 App 测试**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: 全部 App 测试通过。

- [ ] **Step 6: 提交自动推进**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: advance guided workflow automatically"
```

### Task 5: 保护未保存标注并保持失败样本回流

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 添加未保存离开测试**

在已加载数据集和标注页上画一个草稿框，然后点击“质量与版本”：

```ts
expect(screen.getByRole("alert")).toHaveTextContent("当前图像有未保存的标注修改");
expect(screen.getByRole("button", { name: "留在标注页" })).toBeInTheDocument();
expect(screen.getByRole("button", { name: "放弃修改并离开" })).toBeInTheDocument();
expect(screen.getByRole("heading", { name: "标注" })).toBeInTheDocument();
```

点击“放弃修改并离开”后断言质量页面出现。

- [ ] **Step 2: 添加失败样本回流测试**

在评估页点击预测样本的“打开图像”，断言：

```ts
expect(await screen.findByRole("heading", { name: "标注" })).toBeInTheDocument();
expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(10);
expect(await screen.findByText("预测结果叠加")).toBeInTheDocument();
```

- [ ] **Step 3: 运行新增测试并确认失败**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "未保存|失败样本"`

Expected: FAIL，导航尚未拦截或切换步骤。

- [ ] **Step 4: 添加标注脏状态**

```ts
const [annotationsDirty, setAnnotationsDirty] = useState(false);
const [pendingWorkflowStep, setPendingWorkflowStep] = useState<WorkflowStep | null>(null);
```

在画框、新增预测框、修改、移动、缩放、删除、微调、复制相邻框和标记审查时设置 `true`；加载标注和保存成功后设置 `false`。

导航离开标注页时：

```ts
if (activeWorkflowStep === "annotation" && annotationsDirty && stepId !== "annotation") {
  setPendingWorkflowStep(stepId);
  return;
}
```

渲染应用内确认条，留在页面时清空 pending；放弃时先重新加载当前图像标注，再进入 pending 步骤。

- [ ] **Step 5: 失败样本打开时切换到标注页**

在现有预测样本打开处理函数中，设置选中图像和 `activeReview` 后调用：

```ts
setActiveWorkflowStep("annotation");
window.history.replaceState(null, "", "#annotation");
```

- [ ] **Step 6: 运行 App 测试**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: 全部 App 测试通过。

- [ ] **Step 7: 提交标注保护与回流**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: protect annotation drafts across steps"
```

### Task 6: 响应式收尾和完整验证

**Files:**
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 增加导航可访问性断言**

在测试中确认当前步骤拥有 `aria-current="step"`，锁定步骤拥有 `aria-disabled="true"`，移动选择器标签为“当前步骤”，导航提示使用 `role="status"` 或 `role="alert"`。

- [ ] **Step 2: 完成页面布局样式**

调整现有布局：

- 数据集页和类别页最大内容宽度为 `960px`。
- 标注页使用 `minmax(220px, 300px) minmax(0, 1fr)`，类别选择器位于画布顶部。
- 质量、训练和评估页保持现有内部网格，但不再与其他业务页面同时占据纵向空间。
- `820px` 以下隐藏侧栏、显示移动选择器、标注区域改为单列。
- 固定侧栏不得遮挡页面底部，步骤文字不得溢出。

- [ ] **Step 3: 运行自动化验证**

Run: `cd frontend && npm test -- --run`

Expected: 所有前端测试通过。

Run: `cd frontend && npm run build`

Expected: TypeScript 和 Vite 构建成功。

Run: `cd backend && .venv/bin/python -m pytest -v`

Expected: 51 个后端测试通过。

Run: `git diff --check`

Expected: 无输出，退出码为 0。

- [ ] **Step 4: 浏览器验证桌面工作流**

在 `http://127.0.0.1:5173/` 验证：

- 初始只显示项目与数据页面。
- 锁定步骤点击显示准确原因。
- 导入、类别创建和标注步骤按设计推进。
- 每个主页面只包含对应职责。
- 顶部状态信息紧凑，路径仍正确省略。
- 训练和预测自动刷新在离开页面后继续工作。

- [ ] **Step 5: 浏览器验证窄屏**

在 `390x844` 视口验证：

- 桌面侧栏隐藏，顶部步骤选择器显示。
- 页面无横向滚动。
- 选择锁定步骤显示原因并保持当前页。
- 标注画布、图像列表和编辑器按单列排列。
- 中文步骤名、按钮和状态不重叠。

- [ ] **Step 6: 提交响应式收尾**

```bash
git add frontend/src/styles.css frontend/src/App.test.tsx
git commit -m "fix: polish guided workflow responsiveness"
```

## 最终验收

- [ ] 六个步骤状态和锁定原因符合设计。
- [ ] 主内容一次只显示一个业务页面。
- [ ] 导入和第一个类别创建可以自动推进。
- [ ] URL hash 可以恢复开放步骤并回退锁定步骤。
- [ ] 未保存标注离开时出现应用内确认。
- [ ] 失败样本可以切换到标注页并保持预测叠加。
- [ ] 桌面侧边导航和移动步骤选择器均可用。
- [ ] 前端测试、前端构建、后端测试和浏览器验证通过。
- [ ] `git status --short` 只保留用户原有的未跟踪文件或与本功能无关的改动。
