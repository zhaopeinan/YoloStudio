import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnnotationCanvasToolbar } from "./AnnotationCanvasToolbar";

describe("AnnotationCanvasToolbar", () => {
  it("展示倍率并触发缩放命令", () => {
    const onZoomOut = vi.fn();
    const onZoomIn = vi.fn();
    const onActualSize = vi.fn();
    const onFit = vi.fn();

    render(
      <AnnotationCanvasToolbar
        zoom={2.5}
        canZoomOut
        canZoomIn
        disabled={false}
        onZoomOut={onZoomOut}
        onZoomIn={onZoomIn}
        onActualSize={onActualSize}
        onFit={onFit}
      />,
    );

    expect(screen.getByRole("toolbar", { name: "画布缩放" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "当前倍率 250%" })).toHaveTextContent("250%");

    fireEvent.click(screen.getByRole("button", { name: "缩小图像" }));
    fireEvent.click(screen.getByRole("button", { name: "放大图像" }));
    fireEvent.click(screen.getByRole("button", { name: "显示原始大小" }));
    fireEvent.click(screen.getByRole("button", { name: "适应窗口" }));

    expect(onZoomOut).toHaveBeenCalledOnce();
    expect(onZoomIn).toHaveBeenCalledOnce();
    expect(onActualSize).toHaveBeenCalledOnce();
    expect(onFit).toHaveBeenCalledOnce();
  });

  it("双击倍率恢复适应窗口", () => {
    const onFit = vi.fn();

    render(
      <AnnotationCanvasToolbar
        zoom={1.25}
        canZoomOut
        canZoomIn
        disabled={false}
        onZoomOut={vi.fn()}
        onZoomIn={vi.fn()}
        onActualSize={vi.fn()}
        onFit={onFit}
      />,
    );

    fireEvent.doubleClick(screen.getByRole("button", { name: "当前倍率 125%" }));
    expect(onFit).toHaveBeenCalledOnce();
  });

  it("在倍率边界禁用对应按钮", () => {
    render(
      <AnnotationCanvasToolbar
        zoom={8}
        canZoomOut
        canZoomIn={false}
        disabled={false}
        onZoomOut={vi.fn()}
        onZoomIn={vi.fn()}
        onActualSize={vi.fn()}
        onFit={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "放大图像" })).toBeDisabled();
  });
});
