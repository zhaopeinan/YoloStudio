import { Download, Film, Image as ImageIcon, LoaderCircle, Play, Trash2, Upload } from "lucide-react";
import type { CSSProperties, ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ImagePreviewResponse,
  PreviewJob,
  PreviewModelOption,
  PreviewVideo,
  ProjectClass,
  ProjectDatasetSummary,
  DatasetImage,
  TrainingRun,
} from "./api";
import {
  createVideoPreview,
  deletePreviewVideo,
  deleteVideoPreview,
  getVideoPreviewResultUrl,
  getVideoPreviewStatus,
  getVideoPreviewStreamUrl,
  listAllImages,
  listPreviewModels,
  listPreviewVideos,
  listVideoPreviewJobs,
  previewImage,
  uploadPreviewVideo,
  authedMediaUrl,
} from "./api";

type PreviewMode = "image" | "video";

type PreviewViewProps = {
  projectId: number | null;
  datasets: ProjectDatasetSummary[];
  runs: TrainingRun[];
  classes: ProjectClass[];
};

const supportedVideoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
const activeJobStatuses = new Set(["queued", "preparing", "running"]);

export function PreviewView({ projectId, datasets, runs, classes }: PreviewViewProps) {
  const [mode, setMode] = useState<PreviewMode>("image");
  const [models, setModels] = useState<PreviewModelOption[]>([]);
  const [modelRef, setModelRef] = useState("base:yolov8n.pt");
  const [datasetId, setDatasetId] = useState<number | "">("");
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [imageIndex, setImageIndex] = useState(0);
  const [jumpDraft, setJumpDraft] = useState("1");
  const [imagePreview, setImagePreview] = useState<ImagePreviewResponse | null>(null);
  const [threshold, setThreshold] = useState(0.25);
  const [showPredictions, setShowPredictions] = useState(true);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [imageLoading, setImageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videoJob, setVideoJob] = useState<PreviewJob | null>(null);
  const [libraryVideos, setLibraryVideos] = useState<PreviewVideo[]>([]);
  const [selectedLibraryVideoId, setSelectedLibraryVideoId] = useState<number | "">("");
  const [isUploadingVideo, setIsUploadingVideo] = useState(false);
  const [isSavingVideo, setIsSavingVideo] = useState(false);
  const [isStartingRetest, setIsStartingRetest] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoNotice, setVideoNotice] = useState<string | null>(null);
  const imageCache = useRef(new Map<string, ImagePreviewResponse>());

  const selectedImage = images[imageIndex] ?? null;
  const completedRuns = useMemo(
    () => runs.filter((run) => run.status === "completed"),
    [runs],
  );

  const refreshVideoLibrary = useCallback(async (project: number) => {
    const response = await listPreviewVideos(project);
    setLibraryVideos(response.items);
    setSelectedLibraryVideoId((current) => {
      if (current !== "" && response.items.some((item) => item.id === current)) {
        return current;
      }
      return response.items[0]?.id ?? "";
    });
  }, []);

  useEffect(() => {
    if (projectId === null) {
      setModels([]);
      return;
    }
    let cancelled = false;
    setError(null);
    void listPreviewModels(projectId)
      .then((response) => {
        if (cancelled) return;
        setModels(response.items);
        setModelRef((current) =>
          response.items.some((item) => item.model_ref === current)
            ? current
            : response.items[0]?.model_ref ?? "base:yolov8n.pt",
        );
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "模型列表加载失败");
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    if (projectId === null) {
      setLibraryVideos([]);
      setSelectedLibraryVideoId("");
      setVideoJob(null);
      return;
    }
    let cancelled = false;
    setVideoError(null);
    void (async () => {
      try {
        const [videos, jobs] = await Promise.all([
          listPreviewVideos(projectId),
          listVideoPreviewJobs(projectId),
        ]);
        if (cancelled) return;
        setLibraryVideos(videos.items);
        setSelectedLibraryVideoId(videos.items[0]?.id ?? "");
        const active = jobs.items.find((job) => activeJobStatuses.has(job.status));
        const latestCompleted = jobs.items.find((job) => job.status === "completed");
        setVideoJob(active ?? latestCompleted ?? null);
      } catch (reason: unknown) {
        if (!cancelled) {
          setVideoError(reason instanceof Error ? reason.message : "视频库加载失败");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    setDatasetId((current) => {
      if (datasets.some((dataset) => dataset.id === current)) return current;
      return datasets[0]?.id ?? "";
    });
  }, [datasets]);

  useEffect(() => {
    if (datasetId === "") {
      setImages([]);
      setImageIndex(0);
      setJumpDraft("1");
      setImagePreview(null);
      return;
    }
    let cancelled = false;
    setImageLoading(true);
    setError(null);
    imageCache.current.clear();
    void listAllImages(datasetId)
      .then((items) => {
        if (cancelled) return;
        setImages(items);
        setImageIndex(0);
        setJumpDraft("1");
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "数据集图像加载失败");
      })
      .finally(() => {
        if (!cancelled) setImageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  useEffect(() => {
    if (!selectedImage) {
      setImagePreview(null);
      return;
    }
    const cacheKey = `${selectedImage.id}:${modelRef}:${threshold.toFixed(2)}`;
    const cached = imageCache.current.get(cacheKey);
    if (cached) {
      setImagePreview(cached);
      return;
    }
    let cancelled = false;
    setImageLoading(true);
    setError(null);
    void previewImage(selectedImage.id, {
      model_ref: modelRef,
      confidence_threshold: threshold,
    })
      .then((response) => {
        if (cancelled) return;
        imageCache.current.set(cacheKey, response);
        setImagePreview(response);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setImagePreview(null);
          setError(reason instanceof Error ? reason.message : "图像预览失败");
        }
      })
      .finally(() => {
        if (!cancelled) setImageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelRef, selectedImage, threshold]);

  useEffect(() => {
    if (!videoJob || ["completed", "failed", "deleted"].includes(videoJob.status)) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await getVideoPreviewStatus(videoJob.id);
        if (!cancelled) setVideoJob(next);
      } catch (reason: unknown) {
        if (!cancelled) setVideoError(reason instanceof Error ? reason.message : "视频状态刷新失败");
      }
    };
    const timer = window.setInterval(() => void poll(), 500);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [videoJob?.id, videoJob?.status]);

  function handleModelChange(nextModel: string) {
    setModelRef(nextModel);
    imageCache.current.clear();
  }

  function handleDatasetChange(value: string) {
    setDatasetId(value ? Number(value) : "");
  }

  function changeImage(direction: -1 | 1) {
    setImageIndex((current) => {
      const next = Math.max(0, Math.min(images.length - 1, current + direction));
      setJumpDraft(String(next + 1));
      return next;
    });
  }

  function commitImageJump(rawValue: string) {
    if (images.length === 0) {
      setJumpDraft("1");
      return;
    }
    const parsed = Number.parseInt(rawValue.trim(), 10);
    if (!Number.isFinite(parsed)) {
      setJumpDraft(String(imageIndex + 1));
      return;
    }
    const nextIndex = Math.max(0, Math.min(images.length - 1, parsed - 1));
    setImageIndex(nextIndex);
    setJumpDraft(String(nextIndex + 1));
  }

  function validateVideoFile(file: File): string | null {
    const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
    if (!supportedVideoExtensions.includes(extension)) {
      return "仅支持 MP4、MOV、AVI、MKV 和 WEBM 视频";
    }
    return null;
  }

  async function handleSaveVideoOnly(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || projectId === null) return;
    const invalid = validateVideoFile(file);
    if (invalid) {
      setVideoError(invalid);
      return;
    }
    setVideoError(null);
    setVideoNotice(null);
    setIsSavingVideo(true);
    const form = new FormData();
    form.append("project_id", String(projectId));
    form.append("video", file);
    try {
      const saved = await uploadPreviewVideo(form);
      await refreshVideoLibrary(projectId);
      setSelectedLibraryVideoId(saved.id);
      setVideoNotice(`已保存到视频库：${saved.original_filename}`);
    } catch (reason: unknown) {
      setVideoError(reason instanceof Error ? reason.message : "视频上传失败");
    } finally {
      setIsSavingVideo(false);
    }
  }

  async function handleVideoUploadAndRun(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || projectId === null) return;
    const invalid = validateVideoFile(file);
    if (invalid) {
      setVideoError(invalid);
      return;
    }
    setVideoError(null);
    setVideoNotice(null);
    setIsUploadingVideo(true);
    const form = new FormData();
    form.append("project_id", String(projectId));
    form.append("model_ref", modelRef);
    form.append("confidence_threshold", String(threshold));
    form.append("frame_step", "1");
    form.append("video", file);
    try {
      const job = await createVideoPreview(form);
      setVideoJob(job);
      await refreshVideoLibrary(projectId);
      if (job.video_id) {
        setSelectedLibraryVideoId(job.video_id);
      }
    } catch (reason: unknown) {
      setVideoError(reason instanceof Error ? reason.message : "视频上传失败");
    } finally {
      setIsUploadingVideo(false);
    }
  }

  async function handleStartRetest() {
    if (projectId === null || selectedLibraryVideoId === "") return;
    setVideoError(null);
    setVideoNotice(null);
    setIsStartingRetest(true);
    const form = new FormData();
    form.append("project_id", String(projectId));
    form.append("model_ref", modelRef);
    form.append("confidence_threshold", String(threshold));
    form.append("frame_step", "1");
    form.append("video_id", String(selectedLibraryVideoId));
    try {
      setVideoJob(await createVideoPreview(form));
    } catch (reason: unknown) {
      setVideoError(reason instanceof Error ? reason.message : "回测启动失败");
    } finally {
      setIsStartingRetest(false);
    }
  }

  async function handleDeleteLibraryVideo(video: PreviewVideo) {
    if (!window.confirm(`确定删除视频「${video.original_filename}」？此操作不可恢复。`)) {
      return;
    }
    setVideoError(null);
    setVideoNotice(null);
    try {
      await deletePreviewVideo(video.id);
      if (projectId !== null) {
        await refreshVideoLibrary(projectId);
      }
      setVideoNotice(`已删除：${video.original_filename}`);
    } catch (reason: unknown) {
      setVideoError(reason instanceof Error ? reason.message : "删除视频失败");
    }
  }

  async function handleDeleteVideo() {
    if (!videoJob) return;
    try {
      await deleteVideoPreview(videoJob.id);
      setVideoJob(null);
      setVideoError(null);
      setVideoNotice("已删除本次推理结果（源视频仍保留在视频库）");
    } catch (reason: unknown) {
      setVideoError(reason instanceof Error ? reason.message : "删除视频结果失败");
    }
  }

  return (
    <section className="preview-page" aria-label="效果预览">
      <div className="preview-heading">
        <div>
          <p className="eyebrow">模型效果检查</p>
          <h2>效果预览</h2>
          <p className="preview-intro">选择模型和数据，逐张核对预测框；视频会边推理边播放并保存结果。</p>
        </div>
        <div className="preview-ready-summary">
          <strong>{completedRuns.length} 个已完成训练任务</strong>
          <span>{classes.length} 个项目类别</span>
        </div>
      </div>

      <div className="preview-mode-tabs" role="tablist" aria-label="预览模式">
        <button type="button" role="tab" aria-selected={mode === "image"} onClick={() => setMode("image")}>
          <ImageIcon size={17} /> 图片逐张预览
        </button>
        <button type="button" role="tab" aria-selected={mode === "video"} onClick={() => setMode("video")}>
          <Film size={17} /> 视频实时预览
        </button>
      </div>

      <div className="preview-control-bar">
        <label>
          预览模型
          <select aria-label="预览模型" value={modelRef} onChange={(event) => handleModelChange(event.target.value)}>
            {models.length === 0 ? <option value="base:yolov8n.pt">内置模型 · YOLOv8n</option> : null}
            {models.map((model) => <option key={model.model_ref} value={model.model_ref}>{model.label}</option>)}
          </select>
        </label>
        <label>
          置信度阈值
          <input
            aria-label="置信度阈值"
            type="number"
            min="0.01"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(event) => setThreshold(Math.max(0.01, Math.min(0.99, Number(event.target.value) || 0.01)))}
          />
        </label>
        {mode === "image" ? (
          <label>
            预览数据集
            <select aria-label="预览数据集" value={datasetId} onChange={(event) => handleDatasetChange(event.target.value)}>
              <option value="">请选择数据集</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}（{dataset.image_count} 张图像 · 已标注 {dataset.annotated_image_count ?? 0} 张）
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      {mode === "image" ? (
        <section className="preview-image-layout" aria-label="图片预览">
          <div className="panel preview-image-panel">
            <div className="preview-panel-heading">
              <div>
                <strong>{selectedImage?.relative_path ?? "尚未选择图像"}</strong>
                {images.length > 0 ? (
                  <label className="preview-jump">
                    第
                    <input
                      aria-label="跳转到第几张"
                      type="number"
                      min={1}
                      max={images.length}
                      inputMode="numeric"
                      value={jumpDraft}
                      onChange={(event) => setJumpDraft(event.target.value)}
                      onBlur={() => commitImageJump(jumpDraft)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          commitImageJump(jumpDraft);
                          (event.target as HTMLInputElement).blur();
                        }
                      }}
                    />
                    <span>/ {images.length} 张</span>
                  </label>
                ) : (
                  <span>请选择数据集</span>
                )}
              </div>
              <div className="preview-nav-actions">
                <button type="button" className="secondary-button" disabled={imageIndex <= 0} onClick={() => changeImage(-1)}>上一张</button>
                <button type="button" className="secondary-button" disabled={imageIndex >= images.length - 1} onClick={() => changeImage(1)}>下一张</button>
              </div>
            </div>
            {imageLoading ? <div className="preview-loading"><LoaderCircle className="spin" size={20} /> 正在加载预览</div> : null}
            {imagePreview ? (
              <div className="preview-canvas" style={{ aspectRatio: `${imagePreview.width} / ${imagePreview.height}` }}>
                <img src={authedMediaUrl(imagePreview.image_url)} alt={imagePreview.filename} />
                <div className="preview-box-layer" aria-label="预览边界框">
                  {showAnnotations ? imagePreview.annotations.map((box, index) => <PreviewBoxView key={`annotation-${index}`} box={box} kind="annotation" />) : null}
                  {showPredictions ? imagePreview.predictions.map((box, index) => <PreviewBoxView key={`prediction-${index}`} box={box} kind="prediction" />) : null}
                </div>
              </div>
            ) : !imageLoading ? <p className="empty-state preview-empty">选择数据集后将加载第一张图像。</p> : null}
          </div>
          <aside className="panel preview-side-panel">
            <div className="preview-side-heading"><strong>显示图层</strong><span>只读对比</span></div>
            <label className="preview-toggle"><input type="checkbox" checked={showPredictions} onChange={(event) => setShowPredictions(event.target.checked)} /> <span className="preview-swatch prediction-swatch" /> 模型预测框 <b>{imagePreview?.predictions.length ?? 0}</b></label>
            <label className="preview-toggle"><input type="checkbox" checked={showAnnotations} onChange={(event) => setShowAnnotations(event.target.checked)} /> <span className="preview-swatch annotation-swatch" /> 已有人工标注 <b>{imagePreview?.annotations.length ?? 0}</b></label>
            <div className="preview-legend-block"><span><i className="preview-swatch prediction-swatch" />橙色：模型预测</span><span><i className="preview-swatch annotation-swatch" />绿色：人工标注</span></div>
            <p className="preview-note">此页面不会修改原图和标注。调整阈值后可观察召回率与误报的变化。</p>
          </aside>
        </section>
      ) : (
        <section className="preview-video-panel" aria-label="视频预览">
          <div className="preview-video-layout">
            <div className="panel preview-video-library" aria-label="视频库">
              <div className="preview-panel-heading">
                <div>
                  <strong>视频库</strong>
                  <span>上传后保存在服务器，可反复选择回测</span>
                </div>
              </div>
              <div className="preview-video-library-actions">
                <label className="preview-upload-button">
                  <Upload size={17} /> {isSavingVideo ? "正在保存..." : "上传并保存"}
                  <input
                    aria-label="上传并保存视频"
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                    onChange={(event) => void handleSaveVideoOnly(event)}
                    disabled={isSavingVideo || projectId === null}
                  />
                </label>
                <label className="preview-upload-button secondary">
                  <Play size={17} /> {isUploadingVideo ? "正在上传..." : "上传并立即推理"}
                  <input
                    aria-label="上传视频并立即推理"
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                    onChange={(event) => void handleVideoUploadAndRun(event)}
                    disabled={isUploadingVideo || projectId === null}
                  />
                </label>
              </div>
              {libraryVideos.length === 0 ? (
                <p className="empty-state preview-empty">暂无已保存视频。</p>
              ) : (
                <table className="preview-video-table" aria-label="视频库列表">
                  <thead>
                    <tr>
                      <th scope="col">文件名</th>
                      <th scope="col">大小</th>
                      <th scope="col">上传时间</th>
                      <th scope="col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {libraryVideos.map((video) => (
                      <tr
                        key={video.id}
                        className={selectedLibraryVideoId === video.id ? "is-selected" : undefined}
                      >
                        <th scope="row">{video.original_filename}</th>
                        <td>{formatBytes(video.size_bytes)}</td>
                        <td>{formatDateTime(video.created_at)}</td>
                        <td>
                          <div className="preview-video-row-actions">
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() => setSelectedLibraryVideoId(video.id)}
                            >
                              选用
                            </button>
                            <button
                              type="button"
                              className="secondary-button danger-button"
                              aria-label={`删除视频 ${video.original_filename}`}
                              onClick={() => void handleDeleteLibraryVideo(video)}
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div className="panel preview-video-run">
              <div className="preview-panel-heading">
                <div>
                  <strong>回测推理</strong>
                  <span>选择视频库中的文件，配合上方模型与阈值开始回测</span>
                </div>
                {videoJob ? (
                  <span className={`preview-job-status ${videoJob.status}`}>
                    {formatPreviewStatus(videoJob.status)}
                  </span>
                ) : null}
              </div>
              <div className="preview-retest-controls">
                <label>
                  已保存视频
                  <select
                    aria-label="已保存视频"
                    value={selectedLibraryVideoId}
                    onChange={(event) =>
                      setSelectedLibraryVideoId(event.target.value ? Number(event.target.value) : "")
                    }
                  >
                    <option value="">请选择视频</option>
                    {libraryVideos.map((video) => (
                      <option key={video.id} value={video.id}>
                        {video.original_filename}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="primary-button"
                  disabled={
                    selectedLibraryVideoId === "" ||
                    isStartingRetest ||
                    projectId === null ||
                    (videoJob != null && activeJobStatuses.has(videoJob.status))
                  }
                  onClick={() => void handleStartRetest()}
                >
                  <Play size={16} />
                  {isStartingRetest ? "正在启动..." : "开始回测"}
                </button>
              </div>
              {videoNotice ? <div role="status" className="status-message success-message">{videoNotice}</div> : null}
              {videoError ? <div className="error-banner">{videoError}</div> : null}
              {videoJob ? (
                <div className="video-result-area">
                  {videoJob.status === "completed" && videoJob.result_url ? (
                    <video
                      key={`result-${videoJob.id}`}
                      className="preview-result-video"
                      controls
                      preload="auto"
                      src={authedMediaUrl(getVideoPreviewResultUrl(videoJob.id))}
                      onLoadedMetadata={(event) => {
                        const video = event.currentTarget;
                        if (Number.isFinite(video.duration) && video.duration > 0) {
                          // Stay on the final frame instead of the often-black start.
                          video.currentTime = Math.max(0, video.duration - 0.04);
                        }
                      }}
                    />
                  ) : videoJob.status !== "failed" ? (
                    <div className="preview-stream-frame">
                      <img
                        src={authedMediaUrl(getVideoPreviewStreamUrl(videoJob.id))}
                        alt="视频实时推理结果"
                      />
                    </div>
                  ) : null}
                  <div className="preview-job-progress">
                    <span>
                      {videoJob.status === "completed"
                        ? `结果已保存 · ${videoJob.source_filename}`
                        : videoJob.status === "failed"
                          ? videoJob.error_message
                          : "正在推理，请稍候..."}
                    </span>
                    <span>
                      {videoJob.total_frames > 0
                        ? `${videoJob.processed_frames} / ${videoJob.total_frames} 帧`
                        : "正在读取视频"}
                    </span>
                  </div>
                  {videoJob.status === "completed" ? (
                    <div className="preview-result-actions">
                      <a
                        className="secondary-button"
                        href={authedMediaUrl(getVideoPreviewResultUrl(videoJob.id))}
                        download
                      >
                        <Download size={16} /> 下载结果视频
                      </a>
                      <button
                        type="button"
                        className="secondary-button danger-button"
                        onClick={() => void handleDeleteVideo()}
                      >
                        <Trash2 size={16} /> 删除本次结果
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="empty-state preview-empty">选择已保存视频并开始回测，或上传并立即推理。</p>
              )}
            </div>
          </div>
        </section>
      )}
    </section>
  );
}

function PreviewBoxView({ box, kind }: { box: ImagePreviewResponse["predictions"][number]; kind: "annotation" | "prediction" }) {
  const boxColor = kind === "prediction" ? "#f59e0b" : "#22a06b";
  const style = {
    left: `${Math.max(0, (box.x_center - box.width / 2) * 100)}%`,
    top: `${Math.max(0, (box.y_center - box.height / 2) * 100)}%`,
    width: `${Math.min(100, box.width * 100)}%`,
    height: `${Math.min(100, box.height * 100)}%`,
    "--box-color": boxColor,
  } as CSSProperties;
  return <div className={`preview-box ${kind}`} style={style} aria-label={`${kind === "prediction" ? "预测" : "标注"}：${box.class_name}`}><span>{box.class_name}{box.confidence === null ? "" : ` ${box.confidence.toFixed(2)}`}</span></div>;
}

function formatPreviewStatus(status: string) {
  return ({ queued: "排队中", preparing: "准备中", running: "推理中", completed: "已完成", failed: "失败" } as Record<string, string>)[status] ?? status;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[index]}`;
}

function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
