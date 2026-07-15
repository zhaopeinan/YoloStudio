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
      "completed",
    ]);
    expect(steps.every((step) => step.availability === "available")).toBe(true);
  });

  it("解析有效步骤 hash 并拒绝未知值", () => {
    expect(parseWorkflowHash("#annotation")).toBe("annotation");
    expect(parseWorkflowHash("#unknown")).toBe("dataset");
    expect(parseWorkflowHash("")).toBe("dataset");
  });
});
