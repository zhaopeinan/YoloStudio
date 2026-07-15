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

export function AnnotationCanvasToolbar({
  zoom,
  canZoomOut,
  canZoomIn,
  disabled,
  onZoomOut,
  onZoomIn,
  onActualSize,
  onFit,
}: AnnotationCanvasToolbarProps) {
  const percentage = Math.round(zoom * 100);

  return (
    <div className="annotation-canvas-toolbar" role="toolbar" aria-label="画布缩放">
      <button
        type="button"
        className="icon-button"
        aria-label="缩小图像"
        title="缩小图像"
        disabled={disabled || !canZoomOut}
        onClick={onZoomOut}
      >
        <Minus size={16} />
      </button>
      <button
        type="button"
        className="canvas-zoom-value"
        aria-label={`当前倍率 ${percentage}%`}
        title="双击恢复适应窗口"
        disabled={disabled}
        onDoubleClick={onFit}
      >
        {percentage}%
      </button>
      <button
        type="button"
        className="icon-button"
        aria-label="放大图像"
        title="放大图像"
        disabled={disabled || !canZoomIn}
        onClick={onZoomIn}
      >
        <Plus size={16} />
      </button>
      <button
        type="button"
        className="icon-button"
        aria-label="显示原始大小"
        title="显示原始大小"
        disabled={disabled}
        onClick={onActualSize}
      >
        <Scan size={16} />
      </button>
      <button
        type="button"
        className="icon-button"
        aria-label="适应窗口"
        title="适应窗口"
        disabled={disabled}
        onClick={onFit}
      >
        <Maximize2 size={16} />
      </button>
    </div>
  );
}
