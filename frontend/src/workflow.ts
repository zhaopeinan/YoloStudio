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
