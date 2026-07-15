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
