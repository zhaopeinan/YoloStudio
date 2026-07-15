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
    expect(
      countAdvancedImageFilters({
        platform: "vtol",
        class_id: "2",
        edge_tag: "occluded",
        failure_type: "all",
        altitude_min: "",
        altitude_max: "30",
      }),
    ).toBe(4);
  });
});
