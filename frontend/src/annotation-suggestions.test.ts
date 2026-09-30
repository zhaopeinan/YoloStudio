import { describe, expect, it } from "vitest";
import { suggestClassFromFilename } from "./annotation-suggestions";

const classes = [
  "fire_truck",
  "person_white",
  "prius_hybrid",
  "car_lexus",
  "prius_hybrid_camo",
  "suv_camo",
  "car_opel",
  "person_red",
].map((name, index) => ({
  id: index + 1,
  project_id: 1,
  name,
  color: "#ef4444",
  description: null,
  active: true,
  annotation_count: 0,
  version_count: 0,
}));

describe("filename class suggestions", () => {
  it("从文件名中的 firet 建议 fire_truck", () => {
    const result = suggestClassFromFilename("frames/001_firet_02.jpg", classes);
    expect(result.selected?.name).toBe("fire_truck");
    expect(result.token).toBe("firet");
  });

  it("支持大小写、空格和别名", () => {
    expect(suggestClassFromFilename("Lx suUV_001.JPG", classes).selected).toBeNull();
    expect(suggestClassFromFilename("Lx_001.JPG", classes).selected?.name).toBe("car_lexus");
    expect(suggestClassFromFilename("suUV_001.JPG", classes).selected?.name).toBe("suv_camo");
    expect(suggestClassFromFilename("opel_001.JPG", classes).selected?.name).toBe("car_opel");
    expect(suggestClassFromFilename("person_red_001.JPG", classes).selected?.name).toBe(
      "person_red",
    );
  });

  it("按 h/a 姿态命名优先匹配完整类别名", () => {
    const result = suggestClassFromFilename(
      "projects/1/datasets/17/images/prius_hybrid_camo_h10_a036.jpg",
      classes,
    );
    expect(result.selected?.name).toBe("prius_hybrid_camo");
    expect(result.token).toBe("prius_hybrid_camo");
    expect(result.candidates.map((item) => item.name)).toEqual(["prius_hybrid_camo"]);
  });

  it("父类名是子类前缀时优先更长的类别", () => {
    const result = suggestClassFromFilename("frames/prius_hybrid_camo_001.jpg", classes);
    expect(result.selected?.name).toBe("prius_hybrid_camo");
  });

  it("候选不唯一时不擅自选择", () => {
    const result = suggestClassFromFilename("prius_001.jpg", classes);
    expect(result.selected).toBeNull();
    expect(result.candidates.map((item) => item.name)).toEqual([
      "prius_hybrid",
      "prius_hybrid_camo",
    ]);

    const personResult = suggestClassFromFilename("person_001.jpg", classes);
    expect(personResult.selected).toBeNull();
    expect(personResult.candidates.map((item) => item.name)).toEqual([
      "person_white",
      "person_red",
    ]);
  });

  it("无法识别时不切换类别", () => {
    const result = suggestClassFromFilename("unknown_001.jpg", classes);
    expect(result).toEqual({ selected: null, candidates: [], token: null });
  });
});
