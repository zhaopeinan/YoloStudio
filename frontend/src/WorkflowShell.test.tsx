import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WorkflowShell } from "./WorkflowShell";
import { buildWorkflowSteps } from "./workflow";

const emptySteps = buildWorkflowSteps({
  hasDataset: false,
  classCount: 0,
  annotatedImageCount: 0,
  versionCount: 0,
  runCount: 0,
  predictionJobCount: 0,
  exportCount: 0,
});

describe("WorkflowShell", () => {
  it("展示步骤状态并把锁定点击交给调用方", () => {
    const onNavigate = vi.fn();
    render(
      <WorkflowShell
        currentStep="dataset"
        steps={emptySteps}
        navigationNotice={null}
        onNavigate={onNavigate}
      >
        <p>数据集页面内容</p>
      </WorkflowShell>,
    );

    const shell = screen.getByTestId("workflow-shell");
    expect(shell).toHaveClass("workflow-shell-top");
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();

    const navigation = screen.getByRole("navigation", { name: "工作流步骤" });
    expect(navigation).toHaveClass("workflow-navigation-horizontal");
    expect(navigation).toHaveTextContent("项目与数据");
    expect(navigation).not.toHaveTextContent("扫描、导入或加载数据集");
    expect(screen.getByRole("button", { name: /1 项目与数据/ })).toHaveAttribute(
      "aria-current",
      "step",
    );

    const lockedStep = screen.getByRole("button", { name: /2 类别管理/ });
    expect(lockedStep).toHaveAttribute("data-availability", "locked");
    expect(lockedStep).toBeEnabled();
    expect(lockedStep).toHaveTextContent("暂不可用");
    fireEvent.click(lockedStep);

    expect(onNavigate).toHaveBeenCalledWith("classes");
    expect(screen.getByText("数据集页面内容")).toBeInTheDocument();
  });

  it("通过移动步骤选择器导航并显示导航提示", () => {
    const onNavigate = vi.fn();
    render(
      <WorkflowShell
        currentStep="dataset"
        steps={emptySteps}
        navigationNotice="请先导入或加载数据集"
        onNavigate={onNavigate}
      >
        <p>页面内容</p>
      </WorkflowShell>,
    );

    expect(screen.getByRole("status")).toHaveTextContent("请先导入或加载数据集");
    fireEvent.change(screen.getByLabelText("当前步骤"), { target: { value: "classes" } });
    expect(onNavigate).toHaveBeenCalledWith("classes");
  });

  it("仅在非标注页展示前后步骤操作", () => {
    const onNavigate = vi.fn();
    const { rerender } = render(
      <WorkflowShell
        currentStep="classes"
        steps={emptySteps}
        navigationNotice={null}
        onNavigate={onNavigate}
      >
        <p>类别页面</p>
      </WorkflowShell>,
    );

    fireEvent.click(screen.getByRole("button", { name: "上一步：项目与数据" }));
    fireEvent.click(screen.getByRole("button", { name: "下一步：图像标注" }));
    expect(onNavigate).toHaveBeenNthCalledWith(1, "dataset");
    expect(onNavigate).toHaveBeenNthCalledWith(2, "annotation");

    rerender(
      <WorkflowShell
        currentStep="annotation"
        steps={emptySteps}
        navigationNotice={null}
        onNavigate={onNavigate}
      >
        <p>标注页面</p>
      </WorkflowShell>,
    );

    expect(screen.queryByRole("button", { name: /上一步：/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /下一步：/ })).not.toBeInTheDocument();
  });
});
