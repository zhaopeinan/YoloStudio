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
  return (
    Number.isFinite(size.width) &&
    Number.isFinite(size.height) &&
    size.width > 0 &&
    size.height > 0
  );
}

export function clampManualZoom(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(manualMaxZoom, Math.max(manualMinZoom, value));
}

export function nextZoomLevel(current: number, direction: "in" | "out"): number {
  if (direction === "in") {
    return zoomLevels.find((level) => level > current + 0.001) ?? manualMaxZoom;
  }
  return (
    [...zoomLevels].reverse().find((level) => level < current - 0.001) ?? manualMinZoom
  );
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
  const x =
    scaledWidth <= viewport.width
      ? (viewport.width - scaledWidth) / 2
      : Math.min(0, Math.max(viewport.width - scaledWidth, pan.x));
  const y =
    scaledHeight <= viewport.height
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
