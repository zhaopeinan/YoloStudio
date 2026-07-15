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
