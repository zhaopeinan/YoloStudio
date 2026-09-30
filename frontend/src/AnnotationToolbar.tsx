import { ChevronLeft, ChevronRight, MoreHorizontal, Save } from "lucide-react";
import { useState } from "react";
import type { ProjectClass } from "./api";
import type { FilenameSuggestion } from "./annotation-suggestions";

type AnnotationToolbarProps = {
  classes: ProjectClass[];
  selectedClassId: number | null;
  selectedImageIndex: number;
  imageCount: number;
  annotationsDirty: boolean;
  isSaving: boolean;
  error: string | null;
  canGoPrevious: boolean;
  canGoNext: boolean;
  showReviewLayers: boolean;
  showGroundTruth: boolean;
  showPrediction: boolean;
  onClassChange: (classId: number) => void;
  onPrevious: () => void;
  onNext: () => void;
  onSave: () => void;
  onSaveAndNext: () => void;
  canCopyPrevious: boolean;
  canCopyNext: boolean;
  onCopyPrevious: () => void;
  onCopyNext: () => void;
  onGroundTruthChange: (checked: boolean) => void;
  onPredictionChange: (checked: boolean) => void;
  annotationStatus: "unreviewed" | "annotated" | "negative";
  filenameSuggestion: FilenameSuggestion;
  onApplySuggestion: () => void;
  onConfirmNegative: () => void;
  canConfirmNegative: boolean;
};

export function AnnotationToolbar({
  classes,
  selectedClassId,
  selectedImageIndex,
  imageCount,
  annotationsDirty,
  isSaving,
  error,
  canGoPrevious,
  canGoNext,
  showReviewLayers,
  showGroundTruth,
  showPrediction,
  onClassChange,
  onPrevious,
  onNext,
  onSave,
  onSaveAndNext,
  canCopyPrevious,
  canCopyNext,
  onCopyPrevious,
  onCopyNext,
  onGroundTruthChange,
  onPredictionChange,
  annotationStatus,
  filenameSuggestion,
  onApplySuggestion,
  onConfirmNegative,
  canConfirmNegative,
}: AnnotationToolbarProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const saveStatus = error
    ? error
    : isSaving
      ? "保存中"
      : annotationsDirty
        ? "有未保存修改"
        : "已保存";

  return (
    <section className="panel annotation-toolbar" aria-label="标注工具栏">
      <div className="annotation-toolbar-class">
        <label htmlFor="annotation-active-class">当前绘制类别</label>
        <select
          id="annotation-active-class"
          value={selectedClassId ?? ""}
          onChange={(event) => onClassChange(Number(event.target.value))}
          disabled={classes.length === 0}
        >
          {classes.length === 0 ? <option value="">暂无可用类别</option> : null}
          {classes.map((classItem) => (
            <option key={classItem.id} value={classItem.id}>
              {classItem.name}
            </option>
          ))}
        </select>
      </div>

      <div className="annotation-toolbar-suggestion" aria-label="文件名类别建议">
        <span>文件名建议</span>
        {filenameSuggestion.selected ? (
          <strong>{filenameSuggestion.selected.name}</strong>
        ) : filenameSuggestion.candidates.length > 1 ? (
          <strong>候选：{filenameSuggestion.candidates.map((item) => item.name).join("、")}</strong>
        ) : (
          <strong className="muted">未识别</strong>
        )}
        <button
          type="button"
          className="secondary-button"
          disabled={!filenameSuggestion.selected || filenameSuggestion.selected.id === selectedClassId}
          onClick={onApplySuggestion}
        >
          采用建议
        </button>
      </div>

      <div className="annotation-toolbar-navigation">
        <button
          type="button"
          className="icon-button"
          aria-label="上一张图像"
          title="上一张图像"
          disabled={!canGoPrevious}
          onClick={onPrevious}
        >
          <ChevronLeft size={18} />
        </button>
        <span>{selectedImageIndex >= 0 ? `${selectedImageIndex + 1} / ${imageCount}` : `0 / ${imageCount}`}</span>
        <button
          type="button"
          className="icon-button"
          aria-label="下一张图像"
          title="下一张图像"
          disabled={!canGoNext}
          onClick={onNext}
        >
          <ChevronRight size={18} />
        </button>
      </div>

      {showReviewLayers ? (
        <div className="layer-toggles" aria-label="标注审查图层">
          <label>
            <input
              type="checkbox"
              checked={showGroundTruth}
              onChange={(event) => onGroundTruthChange(event.target.checked)}
            />
            真实标注（GT）
          </label>
          <label>
            <input
              type="checkbox"
              checked={showPrediction}
              onChange={(event) => onPredictionChange(event.target.checked)}
            />
            模型预测（Pred）
          </label>
        </div>
      ) : null}

      <span className="annotation-status-label">{annotationStatusLabel(annotationStatus)}</span>
      <span className={error ? "annotation-save-status error" : "annotation-save-status"}>
        {saveStatus}
      </span>

      <div className="annotation-toolbar-actions">
        <button type="button" className="secondary-button" disabled={isSaving || selectedImageIndex < 0} onClick={onSave}>
          <Save size={16} />
          保存
        </button>
        <button
          type="button"
          className="secondary-button negative-button"
          disabled={isSaving || selectedImageIndex < 0 || !canConfirmNegative}
          onClick={onConfirmNegative}
        >
          确认无目标
        </button>
        <button type="button" disabled={isSaving || selectedImageIndex < 0} onClick={onSaveAndNext}>
          保存并下一张
        </button>
        <div className="annotation-toolbar-more">
          <button
            type="button"
            className="icon-button"
            aria-label="更多标注操作"
            aria-expanded={isMenuOpen}
            onClick={() => setIsMenuOpen((open) => !open)}
          >
            <MoreHorizontal size={18} />
          </button>
          {isMenuOpen ? (
            <div className="annotation-command-menu" role="menu">
              <button type="button" role="menuitem" disabled={!canCopyPrevious} onClick={() => { onCopyPrevious(); setIsMenuOpen(false); }}>
                复制上一张标注
              </button>
              <button type="button" role="menuitem" disabled={!canCopyNext} onClick={() => { onCopyNext(); setIsMenuOpen(false); }}>
                复制下一张标注
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function annotationStatusLabel(status: "unreviewed" | "annotated" | "negative") {
  return {
    unreviewed: "待标注",
    annotated: "已标注",
    negative: "已确认无目标",
  }[status];
}
