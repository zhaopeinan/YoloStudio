import { X } from "lucide-react";
import type { ProjectClass } from "./api";
import type { AdvancedImageFilters } from "./annotation-workbench";

type FilterKey = keyof AdvancedImageFilters;

type AnnotationFilterDrawerProps = {
  filters: AdvancedImageFilters;
  classes: ProjectClass[];
  failureOptions: Array<{ value: string; label: string }>;
  onChange: (key: FilterKey, value: string) => void;
  onReset: () => void;
  onApply: () => void;
  onClose: () => void;
};

export function AnnotationFilterDrawer({
  filters,
  classes,
  failureOptions,
  onChange,
  onReset,
  onApply,
  onClose,
}: AnnotationFilterDrawerProps) {
  return (
    <section className="annotation-filter-drawer" aria-label="图像高级筛选">
      <header>
        <strong>高级筛选</strong>
        <button type="button" className="icon-button" aria-label="关闭高级筛选" onClick={onClose}>
          <X size={18} />
        </button>
      </header>

      <label htmlFor="advanced-filter-platform">平台</label>
      <input
        id="advanced-filter-platform"
        value={filters.platform}
        onChange={(event) => onChange("platform", event.target.value)}
        placeholder="iris"
      />

      <label htmlFor="advanced-filter-class">类别</label>
      <select
        id="advanced-filter-class"
        value={filters.class_id}
        onChange={(event) => onChange("class_id", event.target.value)}
      >
        <option value="">全部类别</option>
        {classes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select>

      <label htmlFor="advanced-filter-result">识别结果</label>
      <select
        id="advanced-filter-result"
        value={filters.failure_type}
        onChange={(event) => onChange("failure_type", event.target.value)}
      >
        {failureOptions.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
      </select>

      <label htmlFor="advanced-filter-edge-tag">边缘案例标签</label>
      <input
        id="advanced-filter-edge-tag"
        value={filters.edge_tag}
        onChange={(event) => onChange("edge_tag", event.target.value)}
        placeholder="occluded"
      />

      <div className="range-row">
        <label htmlFor="advanced-filter-altitude-min">
          最低高度
          <input
            id="advanced-filter-altitude-min"
            type="number"
            value={filters.altitude_min}
            onChange={(event) => onChange("altitude_min", event.target.value)}
          />
        </label>
        <label htmlFor="advanced-filter-altitude-max">
          最高高度
          <input
            id="advanced-filter-altitude-max"
            type="number"
            value={filters.altitude_max}
            onChange={(event) => onChange("altitude_max", event.target.value)}
          />
        </label>
      </div>

      <footer>
        <button type="button" className="secondary-button" onClick={onReset}>清除筛选</button>
        <button type="button" onClick={onApply}>应用筛选</button>
      </footer>
    </section>
  );
}
