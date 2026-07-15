# YOLO Trainer 中文界面实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 YOLO Trainer 的用户界面准确统一为简体中文，并确保工作空间和数据库长路径不会溢出状态卡片。

**Architecture:** 静态界面文案直接在现有 React 组件中中文化；重复出现的状态、失败类型、质量问题和边缘案例标签放入独立的本地化模块，由展示层统一格式化。后端枚举、API、数据值、模型名、路径和原始日志保持不变。

**Tech Stack:** React 18、TypeScript、Vite、Vitest、Testing Library、CSS Grid/Flexbox

---

## 文件结构

- Create: `frontend/src/localization.ts`：集中管理训练状态、预测失败类型、质量问题、边缘案例标签和通用回退文案。
- Create: `frontend/src/localization.test.ts`：验证所有受控枚举的中文显示及未知值回退。
- Modify: `frontend/src/App.tsx`：替换用户可见文案并使用集中本地化函数。
- Modify: `frontend/src/App.test.tsx`：将交互查询和断言更新为中文，并覆盖状态与路径展示。
- Modify: `frontend/src/styles.css`：修复状态卡片文本容器收缩和路径省略。

### Task 1: 建立集中术语映射

**Files:**
- Create: `frontend/src/localization.ts`
- Create: `frontend/src/localization.test.ts`

- [ ] **Step 1: 编写失败测试**

创建 `frontend/src/localization.test.ts`：

```ts
import { describe, expect, it } from "vitest";
import {
  formatEdgeTag,
  formatFailureType,
  formatQualityIssueType,
  formatRunStatus,
} from "./localization";

describe("界面本地化", () => {
  it("翻译训练与任务状态", () => {
    expect(formatRunStatus("queued")).toBe("排队中");
    expect(formatRunStatus("preparing")).toBe("准备中");
    expect(formatRunStatus("running")).toBe("运行中");
    expect(formatRunStatus("completed")).toBe("已完成");
    expect(formatRunStatus("failed")).toBe("失败");
    expect(formatRunStatus("cancelled")).toBe("已取消");
  });

  it("翻译预测失败类型", () => {
    expect(formatFailureType("matched")).toBe("匹配正确");
    expect(formatFailureType("false_positive")).toBe("误报");
    expect(formatFailureType("false_negative")).toBe("漏报");
    expect(formatFailureType("class_confusion")).toBe("类别混淆");
  });

  it("翻译质量问题与边缘案例标签", () => {
    expect(formatQualityIssueType("tiny_box")).toBe("极小边界框");
    expect(formatQualityIssueType("duplicate_box")).toBe("重复边界框");
    expect(formatEdgeTag("occluded")).toBe("遮挡");
    expect(formatEdgeTag("camouflaged")).toBe("伪装");
    expect(formatEdgeTag("low_light")).toBe("弱光");
    expect(formatEdgeTag("hard_negative")).toBe("困难负样本");
  });

  it("保留未知数据值", () => {
    expect(formatRunStatus("custom_status")).toBe("custom_status");
    expect(formatFailureType("custom_failure")).toBe("custom_failure");
    expect(formatEdgeTag("operator_tag")).toBe("operator_tag");
  });
});
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `cd frontend && npm test -- --run src/localization.test.ts`

Expected: FAIL，提示无法找到 `./localization`。

- [ ] **Step 3: 实现最小本地化模块**

创建 `frontend/src/localization.ts`，定义完整映射：

```ts
const runStatusLabels: Record<string, string> = {
  queued: "排队中",
  preparing: "准备中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  pending: "等待中",
};

const failureTypeLabels: Record<string, string> = {
  all: "全部",
  matched: "匹配正确",
  false_positive: "误报",
  false_negative: "漏报",
  class_confusion: "类别混淆",
};

const qualityIssueLabels: Record<string, string> = {
  all: "全部问题",
  tiny_box: "极小边界框",
  invalid_box: "无效边界框",
  duplicate_box: "重复边界框",
  unannotated_image: "未标注图像",
  missing_metadata: "缺少元数据",
  missing_image_dimensions: "缺少图像尺寸",
  unknown_class_reference: "未知类别引用",
};

const edgeTagLabels: Record<string, string> = {
  occluded: "遮挡",
  camouflaged: "伪装",
  low_light: "弱光",
  small: "小目标",
  dense: "密集目标",
  hard_negative: "困难负样本",
  tiny_box: "极小边界框",
  invalid_box: "无效边界框",
  duplicate_box: "重复边界框",
  false_positive: "误报",
  false_negative: "漏报",
  class_confusion: "类别混淆",
  reviewed_prediction: "已审查预测",
};

export function formatRunStatus(value: string) {
  return runStatusLabels[value] ?? value;
}

export function formatFailureType(value: string) {
  return failureTypeLabels[value] ?? value;
}

export function formatQualityIssueType(value: string) {
  return qualityIssueLabels[value] ?? value;
}

export function formatEdgeTag(value: string) {
  return edgeTagLabels[value] ?? value;
}
```

- [ ] **Step 4: 运行本地化单元测试**

Run: `cd frontend && npm test -- --run src/localization.test.ts`

Expected: 4 tests passed。

- [ ] **Step 5: 提交术语映射**

```bash
git add frontend/src/localization.ts frontend/src/localization.test.ts
git commit -m "feat: add Chinese UI terminology"
```

### Task 2: 中文化数据集、质量审查与标注工作台

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 将主交互测试改为中文查询并确认失败**

在 `frontend/src/App.test.tsx` 的主流程测试中至少更新以下断言和交互名称：

```ts
expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
expect(screen.getByRole("button", { name: "加载数据集" })).toBeInTheDocument();
expect(screen.getByRole("button", { name: "扫描数据集" })).toBeInTheDocument();
expect(screen.getByRole("button", { name: "导入数据集" })).toBeInTheDocument();
expect(screen.getByText("类别库")).toBeInTheDocument();
expect(screen.getByText("图像浏览器")).toBeInTheDocument();
expect(screen.getByText("质量审查")).toBeInTheDocument();
expect(screen.getByRole("button", { name: "保存标注" })).toBeInTheDocument();
```

将后续点击查询同步改为 `应用筛选`、`下一页`、`上一页`、`打开问题图像`、`创建数据集版本`、`复制下一张` 和 `保存标注`。

- [ ] **Step 2: 运行主流程测试并确认中文查询失败**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"`

Expected: FAIL，首个失败点为找不到中文按钮或标题。

- [ ] **Step 3: 翻译数据集导入和质量审查区域**

在 `frontend/src/App.tsx` 中执行以下准确映射，并同步翻译相关空状态、成功提示和 `aria-label`：

```text
Local Detection Workbench -> 本地目标检测工作台
Workspace -> 工作空间
Database -> 数据库
Devices -> 计算设备
Dataset Intake -> 数据集导入
Scan Local Dataset -> 扫描本地数据集
Dataset path -> 数据集路径
Scan Dataset -> 扫描数据集
Import Dataset -> 导入数据集
Project name -> 项目名称
Dataset name -> 数据集名称
Saved dataset -> 已保存数据集
Load Dataset -> 加载数据集
Training Prep -> 训练准备
Quality Review -> 质量审查
Ready to export -> 可以导出
Apply Auto Tags -> 应用自动标签
Refresh Dimensions -> 刷新图像尺寸
Frozen Dataset -> 冻结数据集
Version Export -> 版本导出
Version name -> 版本名称
Class subset -> 类别子集
Create Dataset Version -> 创建数据集版本
Dataset Coverage -> 数据集覆盖情况
Platforms -> 平台
Altitude -> 高度
Classes -> 类别
Edge Tags -> 边缘案例标签
```

扫描结果表使用“分组、图像数、元数据、高度”，数量摘要使用“张图像”和“个边界框”。

- [ ] **Step 4: 翻译类别库、图像浏览器和标注器**

使用以下映射：

```text
Project Labels -> 项目标签
Class Library -> 类别库
Class name -> 类别名称
Class color -> 类别颜色
Create Class -> 创建类别
Save -> 保存
Cancel -> 取消
Dataset Frames -> 数据集图像
Image Browser -> 图像浏览器
Platform -> 平台
Label status -> 标注状态
Annotated -> 已标注
Unannotated -> 未标注
Failure -> 识别结果
Edge tag -> 边缘案例标签
Apply Filters -> 应用筛选
Reset -> 重置
Previous -> 上一页
Next -> 下一页
Draw And Review -> 标注与审查
Annotation -> 标注
Boxes -> 边界框
Copy Previous -> 复制上一张
Copy Next -> 复制下一张
Save Annotations -> 保存标注
Class -> 类别
Track ID -> 目标轨迹 ID
Edge tags -> 边缘案例标签
```

标注就绪状态显示为“就绪｜数据集已加载”等中文组合；画框、选择图像、选择类别和保存失败等指导文案全部翻译。预设标签按钮调用 `formatEdgeTag`，底层标签值保持英文。

- [ ] **Step 5: 使用集中格式化函数**

在 `App.tsx` 顶部导入：

```ts
import {
  formatEdgeTag,
  formatFailureType,
  formatQualityIssueType,
  formatRunStatus,
} from "./localization";
```

删除 `App.tsx` 内重复的 `formatFailureType` 和通用 `formatIssueType`；质量问题调用 `formatQualityIssueType`，边缘标签调用 `formatEdgeTag`。

- [ ] **Step 6: 运行标注主流程测试**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"`

Expected: 主流程测试通过。

- [ ] **Step 7: 提交数据集和标注工作台中文化**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: localize dataset annotation workflow"
```

### Task 3: 中文化训练、预测、实验看板与导出

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 添加训练与预测中文断言并确认失败**

在 `App.test.tsx` 主流程中加入：

```ts
expect(screen.getByText("训练设置")).toBeInTheDocument();
expect(screen.getByRole("button", { name: "开始训练" })).toBeInTheDocument();
expect(screen.getByText("训练记录")).toBeInTheDocument();
expect(await screen.findByText("实验看板")).toBeInTheDocument();
expect(await screen.findByText("任务对比")).toBeInTheDocument();
expect(await screen.findByText("类别检测结果")).toBeInTheDocument();
expect(await screen.findByText("阈值扫描")).toBeInTheDocument();
expect(await screen.findByText("模型导出")).toBeInTheDocument();
```

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"`

Expected: FAIL，找不到训练或预测区域中文文案。

- [ ] **Step 2: 翻译训练设置与训练记录**

使用以下映射，并翻译所有加载、取消、重跑、日志和空状态文案：

```text
Model Training -> 模型训练
Training Setup -> 训练设置
Model preset or local weights -> 模型预设或本地权重
Epochs -> 训练轮数
Image size -> 图像尺寸
Batch -> 批大小
Device -> 计算设备
Augmentation strategy -> 数据增强策略
Strategy name -> 策略名称
Auto threshold scan -> 自动阈值扫描
Start Training Run -> 开始训练
Experiments -> 实验
Run History -> 训练记录
Load Config -> 加载配置
Rerun -> 重新训练
Cancel Run -> 取消训练
Run Artifacts -> 训练产物
```

训练任务列表中的 `run.status` 使用 `formatRunStatus(run.status)`；`Run #1` 显示为“训练任务 #1”。原始指标名和日志保持不变。

- [ ] **Step 3: 翻译预测分析和标注回流**

使用以下映射：

```text
Model Review -> 模型评估
Prediction Analysis -> 预测分析
Image scope -> 图像范围
Confidence threshold -> 置信度阈值
Use image filters -> 使用图像筛选条件
Run Prediction Analysis -> 开始预测分析
Scan thresholds -> 扫描阈值
Run Threshold Scan -> 执行阈值扫描
Failure type -> 结果类型
Prediction class -> 预测类别
Apply Sample Filters -> 应用样本筛选
Prediction overlay -> 预测叠加层
GT -> 真实标注（GT）
Pred -> 模型预测（Pred）
Add as annotation -> 添加为标注
Mark reviewed -> 标记为已审查
Open Image -> 打开图像
```

预测任务状态使用 `formatRunStatus`，失败类型使用 `formatFailureType`；类别名、置信度数值和预测 ID 保持原样。

- [ ] **Step 4: 翻译实验看板和导出区域**

使用以下映射：

```text
Experiment Dashboard -> 实验看板
Run Comparison -> 任务对比
Run -> 任务
Status -> 状态
Model -> 模型
Box loss -> 边界框损失
Matched -> 匹配正确
False + -> 误报
False - -> 漏报
Best F1 -> 最佳 F1
Best Conf -> 最佳置信度
Artifact -> 产物
Class Outcomes -> 类别检测结果
Confusion Matrix -> 混淆矩阵
Threshold Scan -> 阈值扫描
Best threshold -> 最佳阈值
Precision -> 精确率
Recall -> 召回率
Deployment Artifacts -> 部署产物
Model Export -> 模型导出
.pt Weights -> .pt 权重
Export PT -> 导出 PT
Export ONNX -> 导出 ONNX
Export TENSORRT -> 导出 TensorRT
```

任务对比表中的状态调用 `formatRunStatus(row.status)`。`mAP50`、`F1`、`P`、`R`、模型文件名、路径和产物类型保持原样。

- [ ] **Step 5: 运行完整 App 测试**

Run: `cd frontend && npm test -- --run src/App.test.tsx`

Expected: App 测试全部通过。

- [ ] **Step 6: 提交训练与分析中文化**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat: localize training and analysis UI"
```

### Task 4: 修复路径溢出并验证完整中文界面

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 添加状态路径属性测试并确认失败**

在 App 初始渲染测试中加入：

```ts
const workspacePath = "/tmp/yolo-workspace";
const databasePath = "/tmp/yolo-workspace/app.db";
expect(screen.getByTitle(workspacePath)).toHaveTextContent(workspacePath);
expect(screen.getByTitle(databasePath)).toHaveTextContent(databasePath);
```

确保健康检查 mock 使用相同路径。运行：

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "renders local app"`

Expected: FAIL，路径元素尚无 `title` 属性。

- [ ] **Step 2: 为状态卡片增加可收缩内容容器和完整路径提示**

将 `StatusTile` 扩展为可选路径提示：

```tsx
function StatusTile(props: {
  icon: ReactNode;
  label: string;
  value: string;
  revealFullValue?: boolean;
}) {
  return (
    <div className="status-tile">
      {props.icon}
      <div className="status-tile-content">
        <span>{props.label}</span>
        <strong title={props.revealFullValue ? props.value : undefined}>{props.value}</strong>
      </div>
    </div>
  );
}
```

工作空间和数据库调用传入 `revealFullValue`，计算设备不传入。

- [ ] **Step 3: 修复 CSS 收缩与省略**

在 `frontend/src/styles.css` 中加入：

```css
.status-tile-content {
  min-width: 0;
  flex: 1;
}

.status-tile-content strong {
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
```

保留 `.status-tile { min-width: 0; }`，并将原 `.status-tile strong` 的重复省略属性合并到新选择器。

- [ ] **Step 4: 审计残留英文界面文案**

运行：

```bash
cd frontend
rg -n '>[[:space:]]*[A-Za-z][^<{]*<|aria-label="[A-Za-z]|placeholder="[A-Za-z]' src/App.tsx
```

逐项确认命中只属于以下允许范围：`YOLO Trainer`、技术缩写、模型/设备值、指标名称、文件格式或数据值。其余用户界面文案全部翻译。

- [ ] **Step 5: 运行完整自动化验证**

Run: `cd frontend && npm test -- --run`

Expected: 所有前端测试通过。

Run: `cd frontend && npm run build`

Expected: TypeScript 和 Vite 构建成功。

Run: `git diff --check`

Expected: 无输出，退出码为 0。

- [ ] **Step 6: 浏览器验证桌面与窄屏布局**

在 `http://127.0.0.1:5173/` 检查：

- 桌面宽度下三个状态卡片不溢出，长路径以省略号显示。
- 悬停工作空间和数据库路径可以看到完整路径。
- 窄屏下状态卡片换列后仍不溢出。
- 数据集导入、质量审查、训练、预测、导出和标注区域标题及控件均为中文。
- 中文按钮、标签和表头没有重叠或截断。

- [ ] **Step 7: 提交路径修复和最终验证调整**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "fix: contain long status paths"
```

## 最终验收

- [ ] `cd frontend && npm test -- --run` 全部通过。
- [ ] `cd frontend && npm run build` 通过。
- [ ] `git diff --check` 通过。
- [ ] 浏览器桌面与窄屏检查通过。
- [ ] `git status --short` 只保留用户原有的未跟踪文件或与本功能无关的改动。
