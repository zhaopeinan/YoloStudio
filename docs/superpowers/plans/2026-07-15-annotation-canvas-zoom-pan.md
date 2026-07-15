# YOLO Trainer 标注画布缩放与平移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为标注画布增加适应窗口、`25%–800%` 手动缩放、光标中心滚轮缩放和空格加左键平移，并保证缩放后的标注坐标不漂移。

**Architecture:** 新增纯函数模块负责倍率、适应窗口、锚点缩放和平移约束，`App` 持有当前图像视口状态并继续复用现有 SVG 画框逻辑。图像和 SVG 位于同一个变换层；现有 `getBoundingClientRect()` 坐标换算通过回归测试确认在缩放和平移后仍返回相同归一化坐标。

**Tech Stack:** React 18、TypeScript、Vite、Vitest、Testing Library、lucide-react、SVG、CSS Transform、ResizeObserver

---

## 文件结构

- Create: `frontend/src/annotation-viewport.ts`：倍率限制、适应窗口、锚点缩放和平移约束纯函数。
- Create: `frontend/src/annotation-viewport.test.ts`：视口数学单元测试。
- Create: `frontend/src/AnnotationCanvasToolbar.tsx`：缩小、倍率、放大、`1:1` 和适应窗口控件。
- Create: `frontend/src/AnnotationCanvasToolbar.test.tsx`：工具栏可访问性、状态和回调测试。
- Modify: `frontend/src/App.tsx`：视口状态、图像尺寸、统一变换层、ResizeObserver、滚轮和空格平移。
- Modify: `frontend/src/App.test.tsx`：切图重置、缩放后画框、移动、调整和输入焦点测试。
- Modify: `frontend/src/styles.css`：画布工具栏、变换层、抓手状态和移动端缩放布局。

### Task 1: 建立视口数学模型

**Files:**
- Create: `frontend/src/annotation-viewport.ts`
- Create: `frontend/src/annotation-viewport.test.ts`

- [ ] **Step 1: 编写视口纯函数失败测试**

创建 `frontend/src/annotation-viewport.test.ts`：

```ts
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
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `cd frontend && npm test -- --run src/annotation-viewport.test.ts`

Expected: FAIL，无法找到 `./annotation-viewport`。

- [ ] **Step 3: 实现视口纯函数**

创建 `frontend/src/annotation-viewport.ts`：

```ts
export const manualMinZoom = 0.25;
export const manualMaxZoom = 8;
export const zoomLevels = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8] as const;

export type Size = { width: number; height: number };
export type Point = { x: number; y: number };
export type CanvasViewport = {
  zoom: number;
  panX: number;
  panY: number;
  mode: "fit" | "manual";
};

function validSize(size: Size): boolean {
  return Number.isFinite(size.width) && Number.isFinite(size.height) && size.width > 0 && size.height > 0;
}

export function clampManualZoom(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(manualMaxZoom, Math.max(manualMinZoom, value));
}

export function nextZoomLevel(current: number, direction: "in" | "out"): number {
  if (direction === "in") {
    return zoomLevels.find((level) => level > current + 0.001) ?? manualMaxZoom;
  }
  return [...zoomLevels].reverse().find((level) => level < current - 0.001) ?? manualMinZoom;
}

export function constrainPan(
  image: Size,
  viewport: Size,
  zoom: number,
  pan: Point,
): Point {
  if (!validSize(image) || !validSize(viewport) || !Number.isFinite(zoom) || zoom <= 0) {
    return { x: 0, y: 0 };
  }
  const scaledWidth = image.width * zoom;
  const scaledHeight = image.height * zoom;
  const x = scaledWidth <= viewport.width
    ? (viewport.width - scaledWidth) / 2
    : Math.min(0, Math.max(viewport.width - scaledWidth, pan.x));
  const y = scaledHeight <= viewport.height
    ? (viewport.height - scaledHeight) / 2
    : Math.min(0, Math.max(viewport.height - scaledHeight, pan.y));
  return { x, y };
}

export function fitViewport(image: Size, viewport: Size): CanvasViewport {
  if (!validSize(image) || !validSize(viewport)) {
    return { zoom: 1, panX: 0, panY: 0, mode: "fit" };
  }
  const zoom = Math.min(viewport.width / image.width, viewport.height / image.height, 1);
  const pan = constrainPan(image, viewport, zoom, { x: 0, y: 0 });
  return { zoom, panX: pan.x, panY: pan.y, mode: "fit" };
}

export function zoomAroundPoint(
  current: CanvasViewport,
  requestedZoom: number,
  anchor: Point,
  image: Size,
  viewport: Size,
): CanvasViewport {
  const zoom = clampManualZoom(requestedZoom);
  const imagePointX = (anchor.x - current.panX) / current.zoom;
  const imagePointY = (anchor.y - current.panY) / current.zoom;
  const pan = constrainPan(image, viewport, zoom, {
    x: anchor.x - imagePointX * zoom,
    y: anchor.y - imagePointY * zoom,
  });
  return { zoom, panX: pan.x, panY: pan.y, mode: "manual" };
}
```

- [ ] **Step 4: 运行纯函数测试并提交**

Run: `cd frontend && npm test -- --run src/annotation-viewport.test.ts`

Expected: 4 tests passed。

```bash
git add frontend/src/annotation-viewport.ts frontend/src/annotation-viewport.test.ts
git commit -m "feat: model annotation viewport transforms"
```

### Task 2: 创建画布缩放工具栏

**Files:**
- Create: `frontend/src/AnnotationCanvasToolbar.tsx`
- Create: `frontend/src/AnnotationCanvasToolbar.test.tsx`

- [ ] **Step 1: 编写工具栏失败测试**

创建 `frontend/src/AnnotationCanvasToolbar.test.tsx`：

```tsx
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
```

- [ ] **Step 2: 运行测试并确认组件不存在**

Run: `cd frontend && npm test -- --run src/AnnotationCanvasToolbar.test.tsx`

Expected: FAIL，无法找到 `./AnnotationCanvasToolbar`。

- [ ] **Step 3: 实现工具栏组件**

创建 `frontend/src/AnnotationCanvasToolbar.tsx`：

```tsx
import { Maximize2, Minus, Plus, Scan } from "lucide-react";

type AnnotationCanvasToolbarProps = {
  zoom: number;
  canZoomOut: boolean;
  canZoomIn: boolean;
  disabled: boolean;
  onZoomOut: () => void;
  onZoomIn: () => void;
  onActualSize: () => void;
  onFit: () => void;
};

export function AnnotationCanvasToolbar(props: AnnotationCanvasToolbarProps) {
  const percentage = Math.round(props.zoom * 100);
  return (
    <div className="annotation-canvas-toolbar" role="toolbar" aria-label="画布缩放">
      <button type="button" className="icon-button" aria-label="缩小图像" title="缩小图像" disabled={props.disabled || !props.canZoomOut} onClick={props.onZoomOut}>
        <Minus size={16} />
      </button>
      <button type="button" className="canvas-zoom-value" aria-label={`当前倍率 ${percentage}%`} title="双击恢复适应窗口" disabled={props.disabled} onDoubleClick={props.onFit}>
        {percentage}%
      </button>
      <button type="button" className="icon-button" aria-label="放大图像" title="放大图像" disabled={props.disabled || !props.canZoomIn} onClick={props.onZoomIn}>
        <Plus size={16} />
      </button>
      <button type="button" className="icon-button" aria-label="显示原始大小" title="显示原始大小" disabled={props.disabled} onClick={props.onActualSize}>
        <Scan size={16} />
      </button>
      <button type="button" className="icon-button" aria-label="适应窗口" title="适应窗口" disabled={props.disabled} onClick={props.onFit}>
        <Maximize2 size={16} />
      </button>
    </div>
  );
}
```

- [ ] **Step 4: 运行工具栏测试并提交**

Run: `cd frontend && npm test -- --run src/AnnotationCanvasToolbar.test.tsx`

Expected: 2 tests passed。

```bash
git add frontend/src/AnnotationCanvasToolbar.tsx frontend/src/AnnotationCanvasToolbar.test.tsx
git commit -m "feat: add annotation canvas zoom controls"
```

### Task 3: 接入适应窗口与统一变换层

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写切图重置和工具栏集成失败测试**

在 `frontend/src/App.test.tsx` 增加：

```tsx
it("切换图像后恢复适应窗口倍率", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");

  await user.click(screen.getByRole("button", { name: "显示原始大小" }));
  expect(screen.getByRole("button", { name: "当前倍率 100%" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "下一张图像" }));
  expect(screen.getByRole("button", { name: /当前倍率/ })).not.toHaveAccessibleName("当前倍率 100%");
});
```

测试中为画布视口元素模拟 `clientWidth=500`、`clientHeight=300`，图像 mock 尺寸为 `640×480`，因此适应倍率应为 `62.5%`，显示文案为 `63%`。

- [ ] **Step 2: 增加视口状态和尺寸引用**

在 `App.tsx` 导入新模块和组件，并新增：

```ts
const annotationViewportRef = useRef<HTMLDivElement | null>(null);
const [annotationViewportSize, setAnnotationViewportSize] = useState({ width: 0, height: 0 });
const [annotationImageSize, setAnnotationImageSize] = useState({ width: 0, height: 0 });
const [canvasViewport, setCanvasViewport] = useState<CanvasViewport>({
  zoom: 1,
  panX: 0,
  panY: 0,
  mode: "fit",
});
```

`annotationImageSize` 优先使用 `selectedImage.width`、`selectedImage.height`，缺失时由 `<img onLoad>` 的 `naturalWidth`、`naturalHeight` 更新。

- [ ] **Step 3: 监听画布尺寸并计算适应窗口**

新增 `ResizeObserver` effect：

```ts
useEffect(() => {
  const element = annotationViewportRef.current;
  if (!element) return;
  const updateSize = () => setAnnotationViewportSize({
    width: element.clientWidth,
    height: element.clientHeight,
  });
  updateSize();
  const observer = new ResizeObserver(updateSize);
  observer.observe(element);
  return () => observer.disconnect();
}, [selectedImageId, mobileAnnotationPane]);
```

新增 effect：切图时结束现有拖动并把 `mode` 重置为 `fit`；当图像尺寸或视口尺寸变化且当前模式为 `fit` 时调用 `fitViewport`。

- [ ] **Step 4: 重组中央画布标题和变换层**

标题右侧渲染 `AnnotationCanvasToolbar`。将现有 `.viewer-wrap` 改为：

```tsx
<div
  ref={annotationViewportRef}
  className={isCanvasPanning ? "annotation-canvas-viewport panning" : "annotation-canvas-viewport"}
>
  <div
    className="annotation-transform-layer"
    style={{
      width: annotationImageSize.width,
      height: annotationImageSize.height,
      transform: `translate(${canvasViewport.panX}px, ${canvasViewport.panY}px) scale(${canvasViewport.zoom})`,
      transformOrigin: "0 0",
    }}
  >
    <img
      src={selectedImage.image_url}
      alt={selectedImage.relative_path}
      width={annotationImageSize.width}
      height={annotationImageSize.height}
      onLoad={(event) => {
        if (!selectedImage.width || !selectedImage.height) {
          setAnnotationImageSize({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          });
        }
      }}
    />
    <svg
      ref={annotationCanvasRef}
      aria-label="标注画布"
      className={selectedClass ? "annotation-overlay drawable" : "annotation-overlay"}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
    >
      {showGroundTruthLayer
        ? annotations.map((annotation) => (
            <BoxRect
              key={annotation.local_id}
              annotation={annotation}
              color={resolveClassColor(annotation, classById)}
              selected={annotation.local_id === selectedAnnotationId}
              imageWidth={annotationImageSize.width}
              imageHeight={annotationImageSize.height}
              zoom={canvasViewport.zoom}
              onPointerDown={beginMoveAnnotation}
              onResizePointerDown={beginResizeAnnotation}
            />
          ))
        : null}
      {activeReview && activeReview.image.id === selectedImage.id && showPredictionLayer
        ? activeReview.predictions.map((prediction) => (
            <PredictionRect
              key={prediction.id}
              prediction={prediction}
              className={classById.get(prediction.class_id)?.name}
            />
          ))
        : null}
      {dragState ? <DragRect dragState={dragState} color={selectedClass?.color} /> : null}
    </svg>
  </div>
</div>
```

`BoxRect` 在 Task 4 增加 `imageWidth`、`imageHeight` 和 `zoom` props，其余三个 SVG 子节点继续调用现有标注与预测处理函数。

- [ ] **Step 5: 添加基础样式**

```css
.annotation-pane-heading-main {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.annotation-canvas-toolbar {
  display: grid;
  grid-template-columns: repeat(5, 34px);
  gap: 5px;
  align-items: center;
}

.canvas-zoom-value {
  width: 54px;
  min-width: 54px;
  min-height: 34px;
  padding: 0 4px;
  font-variant-numeric: tabular-nums;
}

.annotation-canvas-viewport {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: #111820;
}

.annotation-transform-layer {
  position: absolute;
  top: 0;
  left: 0;
  will-change: transform;
}

.annotation-transform-layer > img,
.annotation-transform-layer > .annotation-overlay {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
}
```

- [ ] **Step 6: 运行集成测试并提交**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "适应窗口倍率" && npm run build`

Expected: 目标测试与生产构建通过。

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: add zoomable annotation canvas viewport"
```

### Task 4: 实现按钮、滚轮和空格平移

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: 编写交互失败测试**

新增测试覆盖：

```tsx
it("滚轮缩放后画框仍保存相同归一化坐标", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: "加载数据集" }));
  await navigateToStep(user, "图像标注");
  const viewport = screen.getByLabelText("标注画布视口");
  const canvas = screen.getByLabelText("标注画布");
  Object.defineProperty(viewport, "clientWidth", { configurable: true, value: 640 });
  Object.defineProperty(viewport, "clientHeight", { configurable: true, value: 480 });
  Object.defineProperty(canvas, "getBoundingClientRect", {
    configurable: true,
    value: () => ({ left: 0, top: 0, width: 1280, height: 960, right: 1280, bottom: 960 }),
  });

  fireEvent.wheel(viewport, { clientX: 320, clientY: 240, deltaY: -120 });
  fireEvent(canvas, pointerEvent("pointerdown", 256, 192));
  fireEvent(canvas, pointerEvent("pointermove", 512, 384));
  fireEvent(canvas, pointerEvent("pointerup", 512, 384));
  await user.click(screen.getByRole("button", { name: "保存" }));

  expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
    10,
    expect.arrayContaining([
      expect.objectContaining({ x_center: 0.3, y_center: 0.3, width: 0.2, height: 0.2 }),
    ]),
  );
});
```

另增两个测试：输入“目标轨迹 ID”时按空格不进入平移；画布获得焦点后按空格并拖动，只更新视口 transform，不增加边界框且不显示“有未保存修改”。

- [ ] **Step 2: 实现工具栏命令**

按钮缩放以视口中心为锚点：

```ts
function setManualZoom(nextZoom: number) {
  const anchor = {
    x: annotationViewportSize.width / 2,
    y: annotationViewportSize.height / 2,
  };
  setCanvasViewport((current) =>
    zoomAroundPoint(current, nextZoom, anchor, annotationImageSize, annotationViewportSize),
  );
}
```

缩小和放大使用 `nextZoomLevel`；`1:1` 使用 `setManualZoom(1)`；适应窗口直接调用 `fitViewport`。

- [ ] **Step 3: 实现光标中心滚轮缩放**

在视口 `onWheel` 中：

```ts
function handleCanvasWheel(event: React.WheelEvent<HTMLDivElement>) {
  if (!selectedImage) return;
  event.preventDefault();
  const bounds = event.currentTarget.getBoundingClientRect();
  const anchor = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  const factor = Math.exp(-event.deltaY * 0.0015);
  setCanvasViewport((current) =>
    zoomAroundPoint(
      current,
      current.zoom * factor,
      anchor,
      annotationImageSize,
      annotationViewportSize,
    ),
  );
}
```

React 合成 `wheel` 监听若无法阻止页面滚动，则在 effect 中为视口注册 `{ passive: false }` 的原生 `wheel` 监听，并在 cleanup 中移除。

- [ ] **Step 4: 实现空格加左键平移**

新增 refs：

```ts
const spacePressedRef = useRef(false);
const panSessionRef = useRef<null | {
  pointerId: number;
  startClientX: number;
  startClientY: number;
  startPanX: number;
  startPanY: number;
}>(null);
const [isCanvasPanning, setIsCanvasPanning] = useState(false);
```

`keydown`、`keyup` 监听仅响应 `event.code === "Space"`，并使用以下判断忽略可编辑目标：

```ts
function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (
    target.matches("input, textarea, select, button") || target.isContentEditable
  );
}
```

视口 `onPointerDownCapture` 在空格按下且左键时创建 pan session、阻止事件传播并捕获指针。窗口 `pointermove` 根据起点差值调用 `constrainPan`；`pointerup`、`pointercancel`、`blur` 和空格抬起统一结束 session。

在 `handlePointerDown`、`beginMoveAnnotation` 和 `beginResizeAnnotation` 开头增加 `spacePressedRef.current` 判断，平移准备状态下不得进入标注交互。

- [ ] **Step 5: 稳定缩放控制点尺寸**

`BoxRect` 新增 `zoom` prop。控制点归一化尺寸使用图像像素与倍率反向补偿：

```ts
const handleWidth = Math.min(0.03, 10 / Math.max(1, imageWidth * zoom));
const handleHeight = Math.min(0.03, 10 / Math.max(1, imageHeight * zoom));
```

将 `imageWidth`、`imageHeight` 和 `canvasViewport.zoom` 传入 `BoxRect`，控制点始终约为 `10×10` 屏幕像素；框线继续使用固定像素 `strokeWidth` 和 `vectorEffect="non-scaling-stroke"`。

- [ ] **Step 6: 运行交互测试并提交**

Run: `cd frontend && npm test -- --run src/App.test.tsx -t "滚轮缩放|空格|平移"`

Expected: 新增交互测试全部通过。

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: add cursor zoom and spacebar panning"
```

### Task 5: 移动端适配与完整验收

**Files:**
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: 增加移动端结构测试**

断言“画布”标签下缩放工具栏仍存在，五个控件名称完整；`.annotation-canvas-viewport` 在移动断点应用 `touch-action: pan-x pan-y`，页面不渲染双指手势提示或控件。

- [ ] **Step 2: 添加移动端样式**

```css
@media (max-width: 820px) {
  .annotation-pane-heading-main {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .annotation-canvas-toolbar {
    width: 100%;
    grid-template-columns: 34px 54px 34px 34px 34px;
    overflow-x: auto;
  }

  .annotation-canvas-viewport {
    min-height: min(520px, calc(100vh - 360px));
    overflow: auto;
    touch-action: pan-x pan-y;
  }
}
```

移动端按钮缩放后，变换层外围尺寸占位元素使用 `annotationImageSize × zoom` 更新滚动范围；桌面端继续使用平移状态和隐藏溢出。

- [ ] **Step 3: 运行完整前端验证**

Run: `cd frontend && npm test -- --run && npm run build`

Expected: 所有测试通过，TypeScript 和 Vite 构建成功。

- [ ] **Step 4: 浏览器验证桌面倍率**

在 `http://127.0.0.1:5173/#annotation` 依次验证：

- 适应窗口完整显示图像。
- `1:1` 显示原始像素大小。
- `400%`、`800%` 下框线清晰，控制点约 `10px`。
- 滚轮缩放时光标指向位置稳定。
- 空格加左键平移，松开空格后恢复画框。
- 缩放后新建、移动、缩放边界框与目标对齐。
- 缩放和平移不改变保存状态。

- [ ] **Step 5: 浏览器验证 `390×844`**

切换到“画布”标签，验证五个缩放控件可达、按钮不溢出、放大后画布可滚动、页面级 `scrollWidth` 等于 `clientWidth`。

- [ ] **Step 6: 运行后端回归和差异检查**

Run: `cd backend && .venv/bin/pytest`

Expected: 现有后端测试全部通过。

Run: `git diff --check && git status --short`

Expected: 无空白错误，只剩项目原有未跟踪文件。

- [ ] **Step 7: 提交浏览器验收修复**

若 Step 4 或 Step 5 产生代码修复，使用以下提交命令：

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css frontend/src/AnnotationCanvasToolbar.tsx frontend/src/AnnotationCanvasToolbar.test.tsx frontend/src/annotation-viewport.ts frontend/src/annotation-viewport.test.ts
git commit -m "fix: polish annotation canvas zoom usability"
```
