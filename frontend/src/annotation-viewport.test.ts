import { describe, expect, it } from "vitest";
import {
  clampManualZoom,
  constrainPan,
  fitViewport,
  nextZoomLevel,
  zoomAroundPoint,
} from "./annotation-viewport";

const image = { width: 2000, height: 1000 };
const viewport = { width: 1000, height: 700 };

describe("annotation viewport math", () => {
  it("限制手动倍率并选择相邻倍率", () => {
    expect(clampManualZoom(0.1)).toBe(0.25);
    expect(clampManualZoom(9)).toBe(8);
    expect(clampManualZoom(Number.NaN)).toBe(1);
    expect(nextZoomLevel(1, "in")).toBe(1.5);
    expect(nextZoomLevel(1, "out")).toBe(0.75);
    expect(nextZoomLevel(0.18, "in")).toBe(0.25);
  });

  it("计算适应窗口倍率和居中位置", () => {
    expect(fitViewport(image, viewport)).toEqual({
      zoom: 0.5,
      panX: 0,
      panY: 100,
      mode: "fit",
    });
    expect(
      fitViewport({ width: 8000, height: 4000 }, { width: 1000, height: 500 }).zoom,
    ).toBe(0.125);
  });

  it("缩放前后保持锚点对应同一图像坐标", () => {
    const current = { zoom: 1, panX: -300, panY: -100, mode: "manual" as const };
    const anchor = { x: 400, y: 250 };
    const before = {
      x: (anchor.x - current.panX) / current.zoom,
      y: (anchor.y - current.panY) / current.zoom,
    };
    const next = zoomAroundPoint(current, 2, anchor, image, viewport);
    expect((anchor.x - next.panX) / next.zoom).toBeCloseTo(before.x);
    expect((anchor.y - next.panY) / next.zoom).toBeCloseTo(before.y);
    expect(next.mode).toBe("manual");
  });

  it("约束大图平移并让小图居中", () => {
    expect(constrainPan(image, viewport, 1, { x: 200, y: -900 })).toEqual({
      x: 0,
      y: -300,
    });
    expect(
      constrainPan({ width: 400, height: 200 }, viewport, 1, { x: 0, y: 0 }),
    ).toEqual({ x: 300, y: 250 });
  });
});
