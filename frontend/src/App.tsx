import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Database,
  Edit3,
  FolderSearch,
  HardDrive,
  Image as ImageIcon,
  Library,
  PackageCheck,
  Play,
  Radar,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Save,
  Share2,
  Tags,
  Trash2,
  Upload,
} from "lucide-react";
import type { FormEvent, PointerEvent, ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  Annotation,
  DatasetCoverageSummary,
  DatasetDimensionRefreshSummary,
  DatasetImageFilters,
  DatasetImage,
  DatasetImportResponse,
  DatasetQualityIssue,
  DatasetQualityIssueType,
  DatasetQualitySummary,
  DatasetScanSummary,
  DatasetVersion,
  ExportArtifact,
  ExportCapabilities,
  HealthResponse,
  ProjectExperimentSummary,
  ProjectClass,
  ProjectSummary,
  AnnotationWrite,
  ClassOutcome,
  ConfusionCell,
  MetricSeries,
  Prediction,
  PredictionFailureType,
  PredictionFilters,
  PredictionImageReview,
  PredictionJob,
  QualityTagApplySummary,
  RunExperimentSummary,
  TrainingAugmentationConfig,
  TrainingRunArtifactSummary,
  TrainingRun,
} from "./api";
import {
  applyQualityTags,
  cancelTrainingRun,
  createClass,
  createDatasetVersion,
  createPredictionJob,
  createPredictionThresholdScan,
  createRunExport,
  createTrainingRun,
  getAnnotations,
  getDatasetCoverage,
  getExportCapabilities,
  getHealth,
  getPredictionJobLogs,
  getPredictionImageReview,
  getProjectTrainingSummary,
  getQuality,
  getTrainingRunArtifacts,
  getTrainingRunLogs,
  getTrainingRunSummary,
  importDataset,
  listClasses,
  listDatasetVersions,
  listImages,
  listPredictionJobs,
  listPredictions,
  listProjects,
  listQualityIssues,
  listRunExports,
  listTrainingRuns,
  replaceAnnotations,
  refreshImageDimensions,
  scanDataset,
  updateClass,
} from "./api";
import {
  formatEdgeTag,
  formatFailureType,
  formatQualityIssueType,
} from "./localization";

const defaultDatasetPath = "/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip";
const defaultProjectName = "YOLO Trainer Project";
const defaultDatasetName = "image_dataset";
const defaultClassColor = "#ef4444";
const imagePageSize = 50;
const monitorRefreshMs = 2500;
const activeRunStatuses = new Set(["queued", "preparing", "running"]);
const activePredictionStatuses = new Set(["queued", "running"]);
const qualityIssueTypeOptions: Array<{ value: DatasetQualityIssueType; label: string }> = [
  { value: "all", label: "全部问题" },
  { value: "unannotated_image", label: "未标注图像" },
  { value: "tiny_box", label: "极小边界框" },
  { value: "invalid_box", label: "无效边界框" },
  { value: "duplicate_box", label: "重复边界框" },
  { value: "missing_metadata", label: "缺少元数据" },
  { value: "missing_image_dimensions", label: "缺少图像尺寸" },
  { value: "unknown_class_reference", label: "未知类别引用" },
];
const predictionFailureOptions: Array<{ value: PredictionFailureType; label: string }> = [
  { value: "all", label: "All" },
  { value: "matched", label: "Matched" },
  { value: "false_positive", label: "False positive" },
  { value: "false_negative", label: "False negative" },
  { value: "class_confusion", label: "Class confusion" },
];
const edgeTagPresets = [
  "occluded",
  "camouflaged",
  "low_light",
  "small",
  "dense",
  "hard_negative",
];
const qualityAutoTagIssueTypes = new Set<DatasetQualityIssueType>([
  "all",
  "tiny_box",
  "invalid_box",
  "duplicate_box",
]);
const defaultAugmentation: TrainingAugmentationConfig = {
  mosaic: 1,
  mixup: 0,
  copy_paste: 0,
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  translate: 0.1,
  scale: 0.5,
  fliplr: 0.5,
  erasing: 0.4,
  gridmask: false,
};

function defaultPredictionFilters() {
  return {
    failure_type: "all" as PredictionFailureType,
    class_id: "",
    confidence_min: "",
    confidence_max: "",
    platform: "",
    altitude_min: "",
    altitude_max: "",
    timestamp_min: "",
    timestamp_max: "",
  };
}

type DraftBox = Annotation & {
  local_id: string;
};

type DragState = {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
};

type BoxMoveState = {
  localId: string;
  startX: number;
  startY: number;
  originalX: number;
  originalY: number;
};

type BoxResizeHandle = "top-left" | "top-right" | "bottom-left" | "bottom-right";

type BoxResizeState = {
  localId: string;
  handle: BoxResizeHandle;
  originalLeft: number;
  originalTop: number;
  originalRight: number;
  originalBottom: number;
};

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [projectName, setProjectName] = useState(defaultProjectName);
  const [datasetName, setDatasetName] = useState(defaultDatasetName);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedSavedDataset, setSelectedSavedDataset] = useState("");
  const [savedDatasetError, setSavedDatasetError] = useState<string | null>(null);
  const [isLoadingSavedDataset, setIsLoadingSavedDataset] = useState(false);
  const [importedDataset, setImportedDataset] = useState<DatasetImportResponse | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [classes, setClasses] = useState<ProjectClass[]>([]);
  const [className, setClassName] = useState("");
  const [classColor, setClassColor] = useState(defaultClassColor);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [versionClassIds, setVersionClassIds] = useState<number[]>([]);
  const [classError, setClassError] = useState<string | null>(null);
  const [isCreatingClass, setIsCreatingClass] = useState(false);
  const [editingClassId, setEditingClassId] = useState<number | null>(null);
  const [classEditName, setClassEditName] = useState("");
  const [classEditColor, setClassEditColor] = useState(defaultClassColor);
  const [isUpdatingClass, setIsUpdatingClass] = useState(false);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [imagePage, setImagePage] = useState({ limit: imagePageSize, offset: 0, total: 0 });
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [imageFilters, setImageFilters] = useState({
    platform: "",
    label_status: "all" as "all" | "annotated" | "unannotated",
    class_id: "",
    edge_tag: "",
    failure_type: "all" as PredictionFailureType,
    altitude_min: "",
    altitude_max: "",
  });
  const [imageFilterError, setImageFilterError] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<DraftBox[]>([]);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [boxMoveState, setBoxMoveState] = useState<BoxMoveState | null>(null);
  const boxMoveStateRef = useRef<BoxMoveState | null>(null);
  const [boxResizeState, setBoxResizeState] = useState<BoxResizeState | null>(null);
  const boxResizeStateRef = useRef<BoxResizeState | null>(null);
  const annotationCanvasRef = useRef<SVGSVGElement | null>(null);
  const stopBoxInteractionTrackingRef = useRef<(() => void) | null>(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);
  const [quality, setQuality] = useState<DatasetQualitySummary | null>(null);
  const [coverage, setCoverage] = useState<DatasetCoverageSummary | null>(null);
  const [qualityIssues, setQualityIssues] = useState<DatasetQualityIssue[]>([]);
  const [qualityIssueType, setQualityIssueType] = useState<DatasetQualityIssueType>("all");
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [isLoadingQuality, setIsLoadingQuality] = useState(false);
  const [isApplyingQualityTags, setIsApplyingQualityTags] = useState(false);
  const [qualityTagSummary, setQualityTagSummary] = useState<QualityTagApplySummary | null>(null);
  const [dimensionRefresh, setDimensionRefresh] =
    useState<DatasetDimensionRefreshSummary | null>(null);
  const [isRefreshingDimensions, setIsRefreshingDimensions] = useState(false);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [versionName, setVersionName] = useState("");
  const [versionError, setVersionError] = useState<string | null>(null);
  const [isCreatingVersion, setIsCreatingVersion] = useState(false);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [runLogs, setRunLogs] = useState<Record<number, string>>({});
  const [runArtifacts, setRunArtifacts] = useState<Record<number, TrainingRunArtifactSummary>>({});
  const [runSummary, setRunSummary] = useState<RunExperimentSummary | null>(null);
  const [projectExperimentSummary, setProjectExperimentSummary] =
    useState<ProjectExperimentSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [trainingNotice, setTrainingNotice] = useState<string | null>(null);
  const [isStartingRun, setIsStartingRun] = useState(false);
  const [cancellingRunId, setCancellingRunId] = useState<number | null>(null);
  const [trainingModel, setTrainingModel] = useState("yolov8n.pt");
  const [trainingEpochs, setTrainingEpochs] = useState(50);
  const [trainingImageSize, setTrainingImageSize] = useState(640);
  const [trainingBatchSize, setTrainingBatchSize] = useState(8);
  const [trainingDevice, setTrainingDevice] = useState("");
  const [augmentationPreset, setAugmentationPreset] = useState("balanced");
  const [augmentation, setAugmentation] = useState<TrainingAugmentationConfig>(defaultAugmentation);
  const [trainingTta, setTrainingTta] = useState(false);
  const [thresholdScan, setThresholdScan] = useState(false);
  const [predictionJobs, setPredictionJobs] = useState<PredictionJob[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionLogs, setPredictionLogs] = useState<Record<number, string>>({});
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isCreatingPrediction, setIsCreatingPrediction] = useState(false);
  const [isCreatingThresholdScan, setIsCreatingThresholdScan] = useState(false);
  const [predictionScope, setPredictionScope] = useState("all");
  const [predictionConfidence, setPredictionConfidence] = useState(0.25);
  const [useImageFiltersForPrediction, setUseImageFiltersForPrediction] = useState(false);
  const [predictionThresholds, setPredictionThresholds] = useState(
    "0.15, 0.25, 0.35, 0.5, 0.65",
  );
  const [predictionFilters, setPredictionFilters] = useState(defaultPredictionFilters);
  const [activeReview, setActiveReview] = useState<PredictionImageReview | null>(null);
  const [showGroundTruthLayer, setShowGroundTruthLayer] = useState(true);
  const [showPredictionLayer, setShowPredictionLayer] = useState(true);
  const [exportCapabilities, setExportCapabilities] = useState<ExportCapabilities | null>(null);
  const [exports, setExports] = useState<ExportArtifact[]>([]);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isCreatingExport, setIsCreatingExport] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error: Error) => setHealthError(error.message));
    void refreshProjects();
  }, []);

  const totalGroupImages = useMemo(
    () => scan?.groups.reduce((total, group) => total + group.image_count, 0) ?? 0,
    [scan],
  );

  const selectedImage = useMemo(
    () => images.find((image) => image.id === selectedImageId) ?? null,
    [images, selectedImageId],
  );

  const selectedClass = useMemo(
    () => classes.find((classItem) => classItem.id === selectedClassId) ?? null,
    [classes, selectedClassId],
  );

  const annotationReadinessSteps = useMemo(
    () => [
      { label: "数据集已加载", complete: Boolean(importedDataset) },
      { label: "类别库", complete: classes.length > 0 },
      { label: "已选择图像", complete: Boolean(selectedImage) },
      { label: "已选择类别", complete: Boolean(selectedClass) },
    ],
    [classes.length, importedDataset, selectedClass, selectedImage],
  );
  const annotationReady = annotationReadinessSteps.every((step) => step.complete);

  const classById = useMemo(() => {
    return new Map(classes.map((classItem) => [classItem.id, classItem]));
  }, [classes]);

  const hasActiveRun = useMemo(() => runs.some((run) => isActiveRun(run.status)), [runs]);
  const hasActivePredictionJob = useMemo(
    () => predictionJobs.some((job) => isActivePredictionJob(job.status)),
    [predictionJobs],
  );
  const activeRunIds = useMemo(
    () => runs.filter((run) => isActiveRun(run.status)).map((run) => run.id),
    [runs],
  );
  const activePredictionJobIds = useMemo(
    () => predictionJobs.filter((job) => isActivePredictionJob(job.status)).map((job) => job.id),
    [predictionJobs],
  );
  const latestRunId = runs[0]?.id ?? null;
  const latestRun = runs[0] ?? null;
  const imagePageStart = imagePage.total === 0 ? 0 : imagePage.offset + 1;
  const imagePageEnd = Math.min(imagePage.offset + images.length, imagePage.total);
  const canPageImagesPrevious = imagePage.offset > 0;
  const canPageImagesNext = imagePage.offset + imagePage.limit < imagePage.total;
  const hasApplicableQualityIssues = qualityIssues.some(
    (issue) =>
      qualityAutoTagIssueTypes.has(issue.issue_type) && issue.annotation_id,
  );
  const canApplyQualityTags =
    Boolean(importedDataset) &&
    qualityAutoTagIssueTypes.has(qualityIssueType) &&
    hasApplicableQualityIssues &&
    !isApplyingQualityTags;
  const activePredictionImageFilters = useMemo(
    () =>
      useImageFiltersForPrediction
        ? activeImageFilterRequest(toImageFilterRequest(imageFilters))
        : undefined,
    [imageFilters, useImageFiltersForPrediction],
  );
  const predictionImageFilterSummary = useMemo(
    () =>
      activePredictionImageFilters
        ? formatImageFilterSummary(activePredictionImageFilters, classById)
        : "All images in selected scope",
    [activePredictionImageFilters, classById],
  );

  useEffect(() => {
    if (!selectedImageId) {
      setAnnotations([]);
      setSelectedAnnotationId(null);
      setDragState(null);
      setActiveBoxMoveState(null);
      return;
    }
    if (activeReview?.image.id === selectedImageId) {
      return;
    }

    setAnnotationError(null);
    setActiveReview(null);
    void loadAnnotations(selectedImageId);
  }, [activeReview?.image.id, selectedImageId]);

  useEffect(() => {
    if (!importedDataset || !hasActiveRun) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshTrainingRuns(importedDataset.project_id);
      activeRunIds.forEach((runId) => {
        void loadRunLogs(runId);
      });
    }, monitorRefreshMs);

    return () => window.clearInterval(interval);
  }, [activeRunIds, hasActiveRun, importedDataset]);

  useEffect(() => {
    if (!latestRunId || !hasActivePredictionJob) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshPredictionJobs(latestRunId);
      activePredictionJobIds.forEach((jobId) => {
        void loadPredictionLogs(jobId);
      });
    }, monitorRefreshMs);

    return () => window.clearInterval(interval);
  }, [activePredictionJobIds, hasActivePredictionJob, latestRunId]);

  useEffect(() => {
    if (!latestRunId) {
      setExportCapabilities(null);
      setExports([]);
      setRunSummary(null);
      setProjectExperimentSummary(null);
      return;
    }

    void refreshExports(latestRunId);
    void refreshRunSummary(latestRunId);
  }, [latestRunId]);

  async function handleScan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsScanning(true);
    setScanError(null);

    try {
      setScan(await scanDataset(datasetPath));
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "数据集扫描失败");
    } finally {
      setIsScanning(false);
    }
  }

  async function handleImportDataset() {
    setIsImporting(true);
    setImportError(null);
    setClassError(null);
    setAnnotationError(null);
    setQualityError(null);
    setVersionError(null);
    setQualityIssues([]);
    setCoverage(null);
    setDimensionRefresh(null);

    try {
      const imported = await importDataset(datasetPath, projectName.trim(), datasetName.trim());
      await loadDatasetWorkspace(imported);
      await refreshProjects();
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "数据集导入失败");
    } finally {
      setIsImporting(false);
    }
  }

  async function refreshProjects() {
    try {
      const response = await listProjects();
      setProjects(response.items);
      setSavedDatasetError(null);
      setSelectedSavedDataset((current) => {
        if (current && savedDatasetFromValue(response.items, current)) {
          return current;
        }
        return firstSavedDatasetValue(response.items);
      });
    } catch (error) {
      setSavedDatasetError(error instanceof Error ? error.message : "已保存数据集加载失败");
    }
  }

  async function loadDatasetWorkspace(dataset: DatasetImportResponse) {
    setImportedDataset(dataset);
    setProjectName(dataset.project_name);
    setDatasetName(dataset.dataset_name);

    const [
      classResponse,
      imageResponse,
      qualityResponse,
      coverageResponse,
      qualityIssueResponse,
      versionResponse,
    ] = await Promise.all([
      listClasses(dataset.project_id),
      listImages(dataset.dataset_id, {}, { limit: imagePageSize, offset: 0 }),
      getQuality(dataset.dataset_id),
      getDatasetCoverage(dataset.dataset_id),
      listQualityIssues(dataset.dataset_id, qualityIssueType),
      listDatasetVersions(dataset.dataset_id),
    ]);
    const runResponse = await listTrainingRuns(dataset.project_id);

    setClasses(classResponse.items);
    setSelectedClassId(classResponse.items[0]?.id ?? null);
    setVersionClassIds(classResponse.items.map((classItem) => classItem.id));
    setImages(imageResponse.items);
    setImagePage({
      limit: imageResponse.limit,
      offset: imageResponse.offset,
      total: imageResponse.total,
    });
    setSelectedImageId(imageResponse.items[0]?.id ?? null);
    setAnnotations([]);
    setActiveReview(null);
    setQuality(qualityResponse);
    setCoverage(coverageResponse);
    setQualityIssues(qualityIssueResponse.items);
    setDimensionRefresh(null);
    setVersions(versionResponse.items);
    setRuns(runResponse.items);
    await refreshRunArtifacts(runResponse.items.map((run) => run.id));
    setPredictionJobs([]);
    setPredictions([]);
    setPredictionLogs({});
    setExportCapabilities(null);
    setExports([]);
    if (runResponse.items[0]) {
      await refreshRunSummary(runResponse.items[0].id);
      await refreshProjectTrainingSummary(dataset.project_id);
      await refreshPredictionJobs(runResponse.items[0].id);
    } else {
      setRunSummary(null);
      setProjectExperimentSummary(null);
    }
  }

  async function handleLoadSavedDataset() {
    const saved = savedDatasetFromValue(projects, selectedSavedDataset);
    if (!saved) {
      return;
    }

    setIsLoadingSavedDataset(true);
    setSavedDatasetError(null);
    try {
      await loadDatasetWorkspace({
        project_id: saved.project.id,
        dataset_id: saved.dataset.id,
        project_name: saved.project.name,
        dataset_name: saved.dataset.name,
        image_count: saved.dataset.image_count,
        groups: [],
      });
    } catch (error) {
      setSavedDatasetError(error instanceof Error ? error.message : "数据集加载失败");
    } finally {
      setIsLoadingSavedDataset(false);
    }
  }

  async function refreshTrainingPrep(datasetId = importedDataset?.dataset_id) {
    if (!datasetId) {
      return;
    }

    setIsLoadingQuality(true);
    setQualityError(null);

    try {
      const [qualityResponse, coverageResponse, qualityIssueResponse, versionResponse] = await Promise.all([
        getQuality(datasetId),
        getDatasetCoverage(datasetId),
        listQualityIssues(datasetId, qualityIssueType),
        listDatasetVersions(datasetId),
      ]);
      setQuality(qualityResponse);
      setCoverage(coverageResponse);
      setQualityIssues(qualityIssueResponse.items);
      setVersions(versionResponse.items);
      if (importedDataset) {
        const runResponse = await listTrainingRuns(importedDataset.project_id);
        setRuns(runResponse.items);
        await refreshRunArtifacts(runResponse.items.map((run) => run.id));
      }
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "质量信息刷新失败");
    } finally {
      setIsLoadingQuality(false);
    }
  }

  async function refreshQualityIssueSamples(
    issueType: DatasetQualityIssueType,
    datasetId = importedDataset?.dataset_id,
  ) {
    if (!datasetId) {
      return;
    }

    setQualityError(null);
    try {
      const response = await listQualityIssues(datasetId, issueType);
      setQualityIssues(response.items);
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "质量问题加载失败");
    }
  }

  async function loadAnnotations(imageId: number) {
    try {
      const response = await getAnnotations(imageId);
      setAnnotations(response.items.map(toDraftBox));
      setSelectedAnnotationId(null);
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "标注加载失败");
    }
  }

  async function refreshImages(
    datasetId = importedDataset?.dataset_id,
    offset = imagePage.offset,
  ) {
    if (!datasetId) {
      return;
    }

    setImageFilterError(null);
    try {
      const response = await listImages(datasetId, toImageFilterRequest(imageFilters), {
        limit: imagePageSize,
        offset,
      });
      setImages(response.items);
      setImagePage({
        limit: response.limit,
        offset: response.offset,
        total: response.total,
      });
      setSelectedImageId((current) => {
        if (response.items.some((image) => image.id === current)) {
          return current;
        }
        return response.items[0]?.id ?? null;
      });
    } catch (error) {
      setImageFilterError(error instanceof Error ? error.message : "图像筛选失败");
    }
  }

  async function handleApplyQualityTags() {
    if (!importedDataset) {
      return;
    }

    setIsApplyingQualityTags(true);
    setQualityError(null);
    setQualityTagSummary(null);

    try {
      const response = await applyQualityTags(importedDataset.dataset_id, qualityIssueType);
      setQualityTagSummary(response);
      await Promise.all([
        refreshTrainingPrep(importedDataset.dataset_id),
        refreshImages(importedDataset.dataset_id),
        selectedImageId ? loadAnnotations(selectedImageId) : Promise.resolve(),
      ]);
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "自动质量标签应用失败");
    } finally {
      setIsApplyingQualityTags(false);
    }
  }

  async function handleRefreshImageDimensions() {
    if (!importedDataset) {
      return;
    }

    setIsRefreshingDimensions(true);
    setQualityError(null);

    try {
      const response = await refreshImageDimensions(importedDataset.dataset_id);
      setDimensionRefresh(response);
      await Promise.all([
        refreshImages(importedDataset.dataset_id),
        refreshTrainingPrep(importedDataset.dataset_id),
      ]);
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "图像尺寸刷新失败");
    } finally {
      setIsRefreshingDimensions(false);
    }
  }

  function handleApplyImageFilters() {
    void refreshImages(importedDataset?.dataset_id, 0);
  }

  function handlePreviousImagePage() {
    const nextOffset = Math.max(0, imagePage.offset - imagePage.limit);
    void refreshImages(importedDataset?.dataset_id, nextOffset);
  }

  function handleNextImagePage() {
    const nextOffset = imagePage.offset + imagePage.limit;
    void refreshImages(importedDataset?.dataset_id, nextOffset);
  }

  function handleResetImageFilters() {
    const nextFilters = {
      platform: "",
      label_status: "all" as const,
      class_id: "",
      edge_tag: "",
      failure_type: "all" as const,
      altitude_min: "",
      altitude_max: "",
    };
    setImageFilters(nextFilters);
    if (importedDataset) {
      setImageFilterError(null);
      listImages(importedDataset.dataset_id, {}, { limit: imagePageSize, offset: 0 })
        .then((response) => {
          setImages(response.items);
          setImagePage({
            limit: response.limit,
            offset: response.offset,
            total: response.total,
          });
          setSelectedImageId(response.items[0]?.id ?? null);
        })
        .catch((error: Error) => setImageFilterError(error.message));
    }
  }

  function toggleVersionClass(classId: number) {
    setVersionClassIds((current) =>
      current.includes(classId)
        ? current.filter((existing) => existing !== classId)
        : [...current, classId],
    );
  }

  function setAugmentationValue(
    field: keyof Omit<TrainingAugmentationConfig, "gridmask">,
    value: number,
  ) {
    setAugmentation((current) => ({ ...current, [field]: value }));
  }

  function buildTrainingRunRequest(versionId: number) {
    return {
      version_id: versionId,
      model: trainingModel.trim() || "yolov8n.pt",
      epochs: trainingEpochs,
      image_size: trainingImageSize,
      batch_size: trainingBatchSize,
      device: trainingDevice.trim() || undefined,
      augmentation_preset: augmentationPreset,
      augmentation,
      tta: trainingTta,
      threshold_scan: thresholdScan,
    };
  }

  function requestFromRunConfig(versionId: number, run: TrainingRun) {
    const config = run.config;
    return {
      version_id: versionId,
      model: stringConfig(config, "model", trainingModel.trim() || "yolov8n.pt"),
      epochs: numberConfig(config, "epochs", trainingEpochs),
      image_size: numberConfig(config, "image_size", trainingImageSize),
      batch_size: numberConfig(config, "batch_size", trainingBatchSize),
      device: stringConfig(config, "device", trainingDevice.trim()) || undefined,
      augmentation_preset: stringConfig(config, "augmentation_preset", augmentationPreset),
      augmentation: augmentationConfig(config, augmentation),
      tta: booleanConfig(config, "tta", trainingTta),
      threshold_scan: booleanConfig(config, "threshold_scan", thresholdScan),
    };
  }

  function applyRunConfigToForm(run: TrainingRun) {
    const config = run.config;
    setTrainingModel(stringConfig(config, "model", "yolov8n.pt"));
    setTrainingEpochs(numberConfig(config, "epochs", 50));
    setTrainingImageSize(numberConfig(config, "image_size", 640));
    setTrainingBatchSize(numberConfig(config, "batch_size", 8));
    setTrainingDevice(stringConfig(config, "device", ""));
    setAugmentationPreset(stringConfig(config, "augmentation_preset", "balanced"));
    setAugmentation(augmentationConfig(config, defaultAugmentation));
    setTrainingTta(booleanConfig(config, "tta", false));
    setThresholdScan(booleanConfig(config, "threshold_scan", false));
    setTrainingNotice(`Loaded config from Run #${run.id}`);
    setTrainingError(null);
  }

  async function refreshTrainingRuns(projectId = importedDataset?.project_id) {
    if (!projectId) {
      return;
    }

    try {
      const runResponse = await listTrainingRuns(projectId);
      setRuns(runResponse.items);
      await refreshRunArtifacts(runResponse.items.map((run) => run.id));
      if (runResponse.items[0]) {
        await refreshRunSummary(runResponse.items[0].id);
        await refreshProjectTrainingSummary(projectId);
        await refreshPredictionJobs(runResponse.items[0].id);
      } else {
        setProjectExperimentSummary(null);
      }
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run refresh failed");
    }
  }

  async function refreshRunArtifacts(runIds: number[]) {
    if (runIds.length === 0) {
      setRunArtifacts({});
      return;
    }

    try {
      const summaries = await Promise.all(runIds.map((runId) => getTrainingRunArtifacts(runId)));
      setRunArtifacts(
        Object.fromEntries(summaries.map((summary) => [summary.run_id, summary])),
      );
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run artifacts failed to load");
    }
  }

  async function refreshProjectTrainingSummary(projectId = importedDataset?.project_id) {
    if (!projectId) {
      setProjectExperimentSummary(null);
      return;
    }

    try {
      setProjectExperimentSummary(await getProjectTrainingSummary(projectId));
      setSummaryError(null);
    } catch (error) {
      setSummaryError(error instanceof Error ? error.message : "Project summary failed to load");
    }
  }

  async function refreshPredictionJobs(runId = runs[0]?.id) {
    if (!runId) {
      setPredictionJobs([]);
      setPredictions([]);
      return;
    }

    try {
      const jobsResponse = await listPredictionJobs(runId);
      setPredictionJobs(jobsResponse.items);
      if (jobsResponse.items[0]) {
        await loadFilteredPredictions(jobsResponse.items[0].id);
        if (
          isActivePredictionJob(jobsResponse.items[0].status) ||
          predictionLogs[jobsResponse.items[0].id]
        ) {
          await loadPredictionLogs(jobsResponse.items[0].id);
        }
        await refreshRunSummary(runId);
      } else {
        setPredictions([]);
        await refreshRunSummary(runId);
      }
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction refresh failed");
    }
  }

  async function refreshExports(runId = runs[0]?.id) {
    if (!runId) {
      setExportCapabilities(null);
      setExports([]);
      return;
    }

    try {
      const [capabilitiesResponse, exportsResponse] = await Promise.all([
        getExportCapabilities(runId),
        listRunExports(runId),
      ]);
      setExportCapabilities(capabilitiesResponse);
      setExports(exportsResponse.items);
      setExportError(null);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "Export refresh failed");
    }
  }

  async function refreshRunSummary(runId = runs[0]?.id) {
    if (!runId) {
      setRunSummary(null);
      return;
    }

    try {
      setRunSummary(await getTrainingRunSummary(runId));
      setSummaryError(null);
    } catch (error) {
      setSummaryError(error instanceof Error ? error.message : "Run summary failed to load");
    }
  }

  async function loadFilteredPredictions(jobId: number) {
    const predictionsResponse = await listPredictions(
      jobId,
      toPredictionFilterRequest(predictionFilters),
    );
    setPredictions(predictionsResponse.items);
  }

  function handleApplyPredictionFilters() {
    const job = predictionJobs[0];
    if (!job) {
      return;
    }

    setPredictionError(null);
    loadFilteredPredictions(job.id).catch((error: Error) => setPredictionError(error.message));
  }

  function handleResetPredictionFilters() {
    const nextFilters = defaultPredictionFilters();
    setPredictionFilters(nextFilters);
    const job = predictionJobs[0];
    if (!job) {
      return;
    }

    setPredictionError(null);
    listPredictions(job.id)
      .then((response) => setPredictions(response.items))
      .catch((error: Error) => setPredictionError(error.message));
  }

  async function handleCreateClass(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!importedDataset || className.trim().length === 0) {
      return;
    }

    setIsCreatingClass(true);
    setClassError(null);

    try {
      const created = await createClass(importedDataset.project_id, {
        name: className.trim(),
        color: classColor,
      });
      setClasses((current) => [...current, created]);
      setVersionClassIds((current) => [...current, created.id]);
      setSelectedClassId(created.id);
      setClassName("");
      void refreshTrainingPrep();
    } catch (error) {
      setClassError(error instanceof Error ? error.message : "类别创建失败");
    } finally {
      setIsCreatingClass(false);
    }
  }

  function beginEditClass(classItem: ProjectClass) {
    setEditingClassId(classItem.id);
    setClassEditName(classItem.name);
    setClassEditColor(classItem.color || defaultClassColor);
    setClassError(null);
  }

  function cancelEditClass() {
    setEditingClassId(null);
    setClassEditName("");
    setClassEditColor(defaultClassColor);
  }

  async function handleUpdateClass(classId: number) {
    if (!importedDataset || classEditName.trim().length === 0) {
      return;
    }

    setIsUpdatingClass(true);
    setClassError(null);

    try {
      const updated = await updateClass(importedDataset.project_id, classId, {
        name: classEditName.trim(),
        color: classEditColor,
      });
      setClasses((current) =>
        current.map((classItem) => (classItem.id === updated.id ? updated : classItem)),
      );
      cancelEditClass();
      void refreshTrainingPrep();
    } catch (error) {
      setClassError(error instanceof Error ? error.message : "类别更新失败");
    } finally {
      setIsUpdatingClass(false);
    }
  }

  function setActiveBoxMoveState(nextState: BoxMoveState | null) {
    if (!nextState && stopBoxInteractionTrackingRef.current && !boxResizeStateRef.current) {
      stopBoxInteractionTrackingRef.current();
      stopBoxInteractionTrackingRef.current = null;
    }
    boxMoveStateRef.current = nextState;
    setBoxMoveState(nextState);
  }

  function setActiveBoxResizeState(nextState: BoxResizeState | null) {
    if (!nextState && stopBoxInteractionTrackingRef.current && !boxMoveStateRef.current) {
      stopBoxInteractionTrackingRef.current();
      stopBoxInteractionTrackingRef.current = null;
    }
    boxResizeStateRef.current = nextState;
    setBoxResizeState(nextState);
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if (!selectedClass || !selectedImage) {
      return;
    }

    const point = getRelativePoint(event);
    setDragState({
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
    });
    setPointerCaptureSafe(event.currentTarget, event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (boxMoveStateRef.current) {
      moveActiveAnnotation(getRelativePoint(event));
      return;
    }

    if (boxResizeStateRef.current) {
      resizeActiveAnnotation(getRelativePoint(event));
      return;
    }

    if (!dragState) {
      return;
    }

    const point = getRelativePoint(event);
    setDragState((current) =>
      current ? { ...current, currentX: point.x, currentY: point.y } : current,
    );
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    if (boxMoveStateRef.current) {
      setActiveBoxMoveState(null);
      releasePointerCaptureSafe(event.currentTarget, event.pointerId);
      return;
    }

    if (boxResizeStateRef.current) {
      setActiveBoxResizeState(null);
      releasePointerCaptureSafe(event.currentTarget, event.pointerId);
      return;
    }

    if (!dragState || !selectedClass) {
      setDragState(null);
      return;
    }

    const point = getRelativePoint(event);
    const box = rectangleToAnnotation(dragState.startX, dragState.startY, point.x, point.y);
    setDragState(null);

    if (!box) {
      return;
    }

    const localId = `draft-${Date.now()}`;
    setAnnotations((current) => [
      ...current,
      {
        ...box,
        class_id: selectedClass.id,
        class_name: selectedClass.name,
        class_color: selectedClass.color,
        local_id: localId,
        track_id: "",
        edge_tags: [],
      },
    ]);
    setSelectedAnnotationId(localId);
  }

  function handlePointerCancel() {
    setDragState(null);
    setActiveBoxMoveState(null);
    setActiveBoxResizeState(null);
  }

  function beginMoveAnnotation(
    event: PointerEvent<SVGElement>,
    annotation: DraftBox,
  ) {
    event.preventDefault();
    event.stopPropagation();
    const point = getRelativePoint(event);
    setSelectedAnnotationId(annotation.local_id);
    setDragState(null);
    setActiveBoxMoveState({
      localId: annotation.local_id,
      startX: point.x,
      startY: point.y,
      originalX: annotation.x_center,
      originalY: annotation.y_center,
    });
    setActiveBoxResizeState(null);
    startBoxInteractionWindowTracking();
    setPointerCaptureSafe(event.currentTarget.ownerSVGElement, event.pointerId);
  }

  function beginResizeAnnotation(
    event: PointerEvent<SVGElement>,
    annotation: DraftBox,
    handle: BoxResizeHandle,
  ) {
    event.preventDefault();
    event.stopPropagation();
    const left = annotation.x_center - annotation.width / 2;
    const top = annotation.y_center - annotation.height / 2;
    const right = annotation.x_center + annotation.width / 2;
    const bottom = annotation.y_center + annotation.height / 2;
    setSelectedAnnotationId(annotation.local_id);
    setDragState(null);
    setActiveBoxMoveState(null);
    setActiveBoxResizeState({
      localId: annotation.local_id,
      handle,
      originalLeft: left,
      originalTop: top,
      originalRight: right,
      originalBottom: bottom,
    });
    startBoxInteractionWindowTracking();
    setPointerCaptureSafe(event.currentTarget.ownerSVGElement, event.pointerId);
  }

  function moveActiveAnnotation(point: { x: number; y: number }) {
    const activeBoxMoveState = boxMoveStateRef.current;
    if (!activeBoxMoveState) {
      return;
    }
    const annotation = annotations.find((item) => item.local_id === activeBoxMoveState.localId);
    if (!annotation) {
      setActiveBoxMoveState(null);
      return;
    }
    const nextX = clampCenter(
      activeBoxMoveState.originalX + point.x - activeBoxMoveState.startX,
      annotation.width,
    );
    const nextY = clampCenter(
      activeBoxMoveState.originalY + point.y - activeBoxMoveState.startY,
      annotation.height,
    );
    updateAnnotation(activeBoxMoveState.localId, { x_center: nextX, y_center: nextY });
  }

  function resizeActiveAnnotation(point: { x: number; y: number }) {
    const activeBoxResizeState = boxResizeStateRef.current;
    if (!activeBoxResizeState) {
      return;
    }
    const annotation = annotations.find((item) => item.local_id === activeBoxResizeState.localId);
    if (!annotation) {
      setActiveBoxResizeState(null);
      return;
    }

    const resizingLeft = activeBoxResizeState.handle.endsWith("left");
    const resizingTop = activeBoxResizeState.handle.startsWith("top");
    const nextLeft = resizingLeft
      ? Math.min(point.x, activeBoxResizeState.originalRight - 0.001)
      : activeBoxResizeState.originalLeft;
    const nextRight = resizingLeft
      ? activeBoxResizeState.originalRight
      : Math.max(point.x, activeBoxResizeState.originalLeft + 0.001);
    const nextTop = resizingTop
      ? Math.min(point.y, activeBoxResizeState.originalBottom - 0.001)
      : activeBoxResizeState.originalTop;
    const nextBottom = resizingTop
      ? activeBoxResizeState.originalBottom
      : Math.max(point.y, activeBoxResizeState.originalTop + 0.001);
    const nextBox = edgesToAnnotation(nextLeft, nextTop, nextRight, nextBottom);

    updateAnnotation(activeBoxResizeState.localId, nextBox);
  }

  function startBoxInteractionWindowTracking() {
    if (stopBoxInteractionTrackingRef.current) {
      stopBoxInteractionTrackingRef.current();
    }

    function handleWindowPointerMove(event: globalThis.PointerEvent) {
      if (!annotationCanvasRef.current) {
        return;
      }
      const point = getRelativePointFromClient(
        annotationCanvasRef.current,
        event.clientX,
        event.clientY,
      );
      if (boxMoveStateRef.current) {
        moveActiveAnnotation(point);
      } else if (boxResizeStateRef.current) {
        resizeActiveAnnotation(point);
      }
    }

    function handleWindowPointerUp() {
      setActiveBoxMoveState(null);
      setActiveBoxResizeState(null);
    }

    window.addEventListener("pointermove", handleWindowPointerMove);
    window.addEventListener("pointerup", handleWindowPointerUp);
    stopBoxInteractionTrackingRef.current = () => {
      window.removeEventListener("pointermove", handleWindowPointerMove);
      window.removeEventListener("pointerup", handleWindowPointerUp);
    };
  }

  function updateAnnotation(localId: string, patch: Partial<DraftBox>) {
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.local_id === localId ? { ...annotation, ...patch } : annotation,
      ),
    );
  }

  function updateAnnotationGeometry(
    localId: string,
    field: keyof Pick<DraftBox, "x_center" | "y_center" | "width" | "height">,
    value: number,
  ) {
    const nextValue =
      field === "width" || field === "height"
        ? clampDimension(value)
        : clamp(value);
    updateAnnotation(localId, { [field]: nextValue });
  }

  function updateAnnotationClass(localId: string, classId: number) {
    const classInfo = classById.get(classId);
    if (!classInfo) {
      return;
    }

    updateAnnotation(localId, {
      class_id: classInfo.id,
      class_name: classInfo.name,
      class_color: classInfo.color,
    });
  }

  function deleteAnnotation(localId: string) {
    setAnnotations((current) => current.filter((annotation) => annotation.local_id !== localId));
    if (selectedAnnotationId === localId) {
      setSelectedAnnotationId(null);
    }
  }

  function nudgeAnnotation(localId: string, deltaX: number, deltaY: number) {
    setSelectedAnnotationId(localId);
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.local_id === localId
          ? {
              ...annotation,
              x_center: clampCenter(annotation.x_center + deltaX, annotation.width),
              y_center: clampCenter(annotation.y_center + deltaY, annotation.height),
            }
          : annotation,
      ),
    );
  }

  async function copyAdjacentAnnotations(direction: "previous" | "next") {
    if (!selectedImageId) {
      return;
    }

    const selectedIndex = images.findIndex((image) => image.id === selectedImageId);
    const adjacent = images[selectedIndex + (direction === "previous" ? -1 : 1)];
    if (!adjacent) {
      return;
    }

    setAnnotationError(null);
    try {
      const response = await getAnnotations(adjacent.id);
      setAnnotations(
        response.items.map((annotation, index) => ({
          ...toDraftBox(annotation, index),
          id: undefined,
          image_id: selectedImageId,
          local_id: `copy-${adjacent.id}-${Date.now()}-${index}`,
        })),
      );
      setSelectedAnnotationId(null);
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "复制标注失败");
    }
  }

  function addPredictionAsAnnotation(prediction: Prediction) {
    const classInfo = classById.get(prediction.class_id);
    setAnnotations((current) => {
      if (current.some((annotation) => annotation.local_id === predictionDraftId(prediction.id))) {
        return current;
      }

      return [...current, predictionToDraftBox(prediction, classInfo)];
    });
    setShowGroundTruthLayer(true);
  }

  function markFalseNegativeReviewed(prediction: Prediction) {
    if (!prediction.matched_annotation_id) {
      return;
    }

    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.id === prediction.matched_annotation_id
          ? {
              ...annotation,
              edge_tags: mergeTags(annotation.edge_tags, [
                "false_negative",
                "reviewed_prediction",
              ]),
            }
          : annotation,
      ),
    );
  }

  async function handleSaveAnnotations() {
    if (!selectedImageId) {
      return;
    }

    setIsSavingAnnotations(true);
    setAnnotationError(null);
    const selectedIndex = annotations.findIndex(
      (annotation) => annotation.local_id === selectedAnnotationId,
    );

    try {
      const response = await replaceAnnotations(selectedImageId, annotations.map(toAnnotationWrite));
      const nextAnnotations = response.items.map(toDraftBox);
      setAnnotations(nextAnnotations);
      setSelectedAnnotationId(
        selectedIndex >= 0
          ? nextAnnotations[selectedIndex]?.local_id ?? null
          : nextAnnotations[0]?.local_id ?? null,
      );
      setImages((current) =>
        current.map((image) =>
          image.id === selectedImageId
            ? { ...image, annotation_count: response.items.length }
            : image,
        ),
      );
      void refreshTrainingPrep();
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "保存标注失败");
    } finally {
      setIsSavingAnnotations(false);
    }
  }

  async function handleCreateDatasetVersion() {
    if (!importedDataset || !quality?.ready_for_training) {
      return;
    }

    setIsCreatingVersion(true);
    setVersionError(null);

    try {
      const created = await createDatasetVersion(
        importedDataset.dataset_id,
        versionName.trim() || undefined,
        versionClassIds,
      );
      setVersions((current) => [created, ...current]);
      setVersionName("");
      await refreshTrainingPrep(importedDataset.dataset_id);
    } catch (error) {
      setVersionError(error instanceof Error ? error.message : "数据集版本导出失败");
    } finally {
      setIsCreatingVersion(false);
    }
  }

  async function handleStartTrainingRun() {
    const version = versions[0];
    if (!version || !importedDataset) {
      return;
    }

    setIsStartingRun(true);
    setTrainingError(null);
    setTrainingNotice(null);

    try {
      const run = await createTrainingRun(buildTrainingRunRequest(version.id));
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      await refreshTrainingRuns(importedDataset.project_id);
      await refreshRunSummary(run.id);
      await loadRunLogs(run.id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Training run failed to start");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function handleRerunTrainingRun(sourceRun: TrainingRun) {
    const version = versions[0];
    if (!version || !importedDataset) {
      return;
    }

    setIsStartingRun(true);
    setTrainingError(null);
    setTrainingNotice(null);

    try {
      const run = await createTrainingRun(requestFromRunConfig(version.id, sourceRun));
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      setTrainingNotice(`Started rerun from Run #${sourceRun.id}`);
      await refreshTrainingRuns(importedDataset.project_id);
      await refreshRunSummary(run.id);
      await loadRunLogs(run.id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Training rerun failed to start");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function loadRunLogs(runId: number) {
    try {
      const response = await getTrainingRunLogs(runId);
      setRunLogs((current) => ({ ...current, [runId]: response.text }));
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run logs failed to load");
    }
  }

  async function handleCancelTrainingRun(runId: number) {
    setCancellingRunId(runId);
    setTrainingError(null);

    try {
      const run = await cancelTrainingRun(runId);
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      await loadRunLogs(run.id);
      if (importedDataset) {
        await refreshTrainingRuns(importedDataset.project_id);
      }
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Training run cancel failed");
    } finally {
      setCancellingRunId(null);
    }
  }

  async function handleCreatePredictionJob() {
    const run = runs[0];
    if (!run) {
      return;
    }

    setIsCreatingPrediction(true);
    setPredictionError(null);

    try {
      const job = await createPredictionJob(run.id, {
        image_scope: predictionScope,
        confidence_threshold: predictionConfidence,
        image_filters: activePredictionImageFilters,
      });
      setPredictionJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      await loadFilteredPredictions(job.id);
      await refreshRunSummary(run.id);
      await loadPredictionLogs(job.id);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction job failed");
    } finally {
      setIsCreatingPrediction(false);
    }
  }

  async function handleCreateThresholdScan() {
    const run = runs[0];
    if (!run) {
      return;
    }

    let thresholds: number[];
    try {
      thresholds = parseThresholdList(predictionThresholds);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Threshold scan input failed");
      return;
    }

    setIsCreatingThresholdScan(true);
    setPredictionError(null);

    try {
      const response = await createPredictionThresholdScan(run.id, {
        image_scope: predictionScope,
        thresholds,
        image_filters: activePredictionImageFilters,
      });
      setPredictionJobs((current) => mergePredictionJobs(response.items, current));
      const latestJob = response.items[response.items.length - 1];
      if (latestJob) {
        await loadFilteredPredictions(latestJob.id);
        await loadPredictionLogs(latestJob.id);
      }
      await refreshRunSummary(run.id);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Threshold scan failed");
    } finally {
      setIsCreatingThresholdScan(false);
    }
  }

  async function loadPredictionLogs(jobId: number) {
    try {
      const response = await getPredictionJobLogs(jobId);
      setPredictionLogs((current) => ({ ...current, [jobId]: response.text }));
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction logs failed to load");
    }
  }

  async function handleCreateExport(format: string) {
    const run = runs[0];
    if (!run) {
      return;
    }

    setIsCreatingExport(format);
    setExportError(null);

    try {
      const artifact = await createRunExport(run.id, format);
      setExports((current) => [artifact, ...current.filter((item) => item.id !== artifact.id)]);
      await refreshExports(run.id);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "Export failed");
    } finally {
      setIsCreatingExport(null);
    }
  }

  async function openPredictionImage(prediction: Prediction) {
    setPredictionError(null);

    try {
      const review = await getPredictionImageReview(prediction.job_id, prediction.image_id);
      const existingImage = images.find((image) => image.id === review.image.id);
      setActiveReview(review);
      setAnnotations(review.annotations.map(toDraftBox));
      setShowGroundTruthLayer(true);
      setShowPredictionLayer(true);
      if (!existingImage) {
        setImages((current) => [
          ...current,
          {
            id: review.image.id,
            relative_path: review.image.relative_path,
            width: null,
            height: null,
            platform: review.image.platform,
            altitude: review.image.altitude,
            timestamp: review.image.timestamp,
            annotation_count: review.annotations.length,
            image_url: review.image.image_url,
          },
        ]);
      }
      setSelectedImageId(review.image.id);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction review failed");
    }
  }

  function openQualityIssue(issue: DatasetQualityIssue) {
    setActiveReview(null);
    setShowGroundTruthLayer(true);
    setShowPredictionLayer(false);
    setImages((current) => {
      if (current.some((image) => image.id === issue.image_id)) {
        return current;
      }
      return [
        ...current,
        {
          id: issue.image_id,
          relative_path: issue.image_path,
          width: null,
          height: null,
          platform: null,
          altitude: null,
          timestamp: null,
          annotation_count: issue.issue_type === "unannotated_image" ? 0 : 1,
          image_url: issue.image_url,
        },
      ];
    });
    setSelectedImageId(issue.image_id);
  }

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="应用状态">
        <div>
          <p className="eyebrow">本地目标检测工作台</p>
          <h1>YOLO Trainer</h1>
        </div>
        <div className="device-pill">
          <Activity size={16} />
          <span>{health?.devices.selected ?? "连接中"}</span>
        </div>
      </section>

      <section className="status-grid" aria-label="后端详情">
        <StatusTile
          icon={<HardDrive size={20} />}
          label="工作空间"
          value={health?.workspace_root ?? "等待后端响应"}
        />
        <StatusTile
          icon={<Database size={20} />}
          label="数据库"
          value={health?.database_path ?? "SQLite 将在启动时初始化"}
        />
        <StatusTile
          icon={<Activity size={20} />}
          label="计算设备"
          value={health ? `${health.devices.available.length} 个可用` : "检测中"}
        />
      </section>

      {healthError ? <div className="error-banner">{healthError}</div> : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">数据集导入</p>
            <h2>扫描本地数据集</h2>
          </div>
          <FolderSearch size={22} />
        </div>

        <form className="scan-form" onSubmit={handleScan}>
          <label htmlFor="dataset-path">数据集路径</label>
          <div className="input-row">
            <input
              id="dataset-path"
              value={datasetPath}
              onChange={(event) => setDatasetPath(event.target.value)}
            />
            <button type="submit" disabled={isScanning || datasetPath.trim().length === 0}>
              {isScanning ? "扫描中" : "扫描数据集"}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={
                isImporting ||
                datasetPath.trim().length === 0 ||
                projectName.trim().length === 0 ||
                datasetName.trim().length === 0
              }
              onClick={handleImportDataset}
            >
              <Upload size={16} />
              {isImporting ? "导入中" : "导入数据集"}
            </button>
          </div>
          <div className="import-name-grid">
            <label htmlFor="project-name">
              项目名称
              <input
                id="project-name"
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
              />
            </label>
            <label htmlFor="dataset-name">
              数据集名称
              <input
                id="dataset-name"
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
              />
            </label>
          </div>
          <div className="saved-dataset-row" aria-label="已保存数据集加载器">
            <label htmlFor="saved-dataset">
              已保存数据集
              <select
                id="saved-dataset"
                value={selectedSavedDataset}
                onChange={(event) => setSelectedSavedDataset(event.target.value)}
              >
                {projects.length === 0 ? <option value="">暂无已保存数据集</option> : null}
                {projects.flatMap((project) =>
                  project.datasets.map((dataset) => (
                    <option
                      key={`${project.id}-${dataset.id}`}
                      value={savedDatasetValue(project.id, dataset.id)}
                    >
                      {project.name} / {dataset.name}（{dataset.image_count} 张图像）
                    </option>
                  )),
                )}
              </select>
            </label>
            <button
              type="button"
              className="secondary-button"
              disabled={!selectedSavedDataset || isLoadingSavedDataset}
              onClick={handleLoadSavedDataset}
            >
              {isLoadingSavedDataset ? "加载中" : "加载数据集"}
            </button>
          </div>
        </form>

        {scanError ? <div className="error-banner">{scanError}</div> : null}
        {importError ? <div className="error-banner">{importError}</div> : null}
        {savedDatasetError ? <div className="error-banner">{savedDatasetError}</div> : null}

        {scan ? (
          <div className="scan-results">
            <div className="metrics-row">
              <Metric label="图像" value={scan.total_images.toLocaleString()} />
              <Metric label="元数据行" value={scan.total_metadata_rows.toLocaleString()} />
              <Metric label="YOLO 标签" value={scan.total_yolo_labels.toLocaleString()} />
              <Metric label="类别配置" value={scan.has_data_yaml ? "已找到" : "缺失"} />
            </div>

            <div className="group-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>分组</th>
                    <th>图像数</th>
                    <th>元数据</th>
                    <th>高度</th>
                  </tr>
                </thead>
                <tbody>
                  {scan.groups.map((group) => (
                    <tr key={group.name}>
                      <td>{group.name}</td>
                      <td>{group.image_count}</td>
                      <td>{group.metadata_rows}</td>
                      <td>{formatAltitude(group.altitude_min, group.altitude_max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="summary-line">
              已分组图像：{totalGroupImages.toLocaleString()} 张，来源：{scan.archive_name}
            </div>

            {scan.warnings.length > 0 ? (
              <div className="warnings">
                <AlertTriangle size={18} />
                <div>
                  {scan.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {importedDataset ? (
          <div className="summary-line">
            已将 {importedDataset.image_count.toLocaleString()} 张图像导入{" "}
            {importedDataset.project_name} / {importedDataset.dataset_name}
          </div>
        ) : null}
      </section>

      <section className="prep-grid" aria-label="训练准备情况">
        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">训练准备</p>
              <h2>质量审查</h2>
            </div>
            <div className="panel-heading-actions">
              {importedDataset ? (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleRefreshImageDimensions}
                  disabled={isRefreshingDimensions}
                >
                  <ImageIcon size={16} />
                  {isRefreshingDimensions ? "刷新中" : "刷新图像尺寸"}
                </button>
              ) : null}
              {quality?.ready_for_training ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
            </div>
          </div>

          {quality ? (
            <>
              <div className="readiness-row">
                <strong>{quality.ready_for_training ? "可以导出" : "需要处理"}</strong>
                <span>{isLoadingQuality ? "刷新中" : `${quality.annotation_count} 个边界框`}</span>
              </div>

              <div className="metrics-row quality-metrics" aria-label="质量指标">
                <Metric label="图像" value={quality.image_count.toLocaleString()} />
                <Metric
                  label="已标注"
                  value={quality.annotated_image_count.toLocaleString()}
                />
                <Metric label="类别" value={quality.class_count.toLocaleString()} />
                <Metric label="极小边界框" value={quality.tiny_box_count.toLocaleString()} />
                <Metric
                  label="重复边界框"
                  value={quality.duplicate_box_count.toLocaleString()}
                />
                <Metric
                  label="缺少元数据"
                  value={quality.missing_metadata_count.toLocaleString()}
                />
                <Metric
                  label="缺少图像尺寸"
                  value={quality.missing_image_dimensions_count.toLocaleString()}
                />
              </div>

              {quality.issues.length > 0 ? (
                <div className="issue-list">
                  {quality.issues.map((issue) => (
                    <p key={issue}>{issue}</p>
                  ))}
                </div>
              ) : (
                <p className="empty-state">未检测到阻塞训练的质量问题。</p>
              )}

              {dimensionRefresh ? (
                <p className="summary-line">{formatDimensionRefresh(dimensionRefresh)}</p>
              ) : null}

              <DatasetCoveragePanel coverage={coverage} />

              <label className="quality-issue-filter" htmlFor="quality-issue-type">
                质量问题类型
                <select
                  id="quality-issue-type"
                  value={qualityIssueType}
                  onChange={(event) => {
                    const nextType = event.target.value as DatasetQualityIssueType;
                    setQualityIssueType(nextType);
                    void refreshQualityIssueSamples(nextType);
                  }}
                >
                  {qualityIssueTypeOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>

              <div className="quality-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!canApplyQualityTags}
                  onClick={handleApplyQualityTags}
                >
                  <Tags size={16} />
                  {isApplyingQualityTags ? "正在应用标签" : "应用自动标签"}
                </button>
                {qualityTagSummary ? (
                  <span className="summary-line">{formatQualityTagSummary(qualityTagSummary)}</span>
                ) : null}
              </div>

              {qualityIssues.length > 0 ? (
                <div className="quality-issue-list" aria-label="质量问题样本">
                  {qualityIssues.map((issue) => (
                    <div
                      className={`quality-issue-row ${issue.severity}`}
                      key={`${issue.issue_type}-${issue.image_id}-${issue.annotation_id ?? "image"}`}
                    >
                      <div>
                        <strong>{formatQualityIssueType(issue.issue_type)}</strong>
                        <span>{issue.message}</span>
                        <small>
                          {issue.image_path}
                          {issue.class_name ? ` | ${issue.class_name}` : ""}
                        </small>
                      </div>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => openQualityIssue(issue)}
                      >
                        打开问题图像
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-state">所选质量问题类型暂无样本。</p>
              )}
            </>
          ) : (
            <p className="empty-state">导入数据集后可计算标注质量和导出准备情况。</p>
          )}

          {qualityError ? <div className="error-banner">{qualityError}</div> : null}
        </section>

        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">冻结数据集</p>
              <h2>版本导出</h2>
            </div>
            <PackageCheck size={20} />
          </div>

          <div className="version-controls">
            <label htmlFor="version-name">版本名称</label>
            <div className="input-row">
              <input
                id="version-name"
                value={versionName}
                disabled={!importedDataset}
                onChange={(event) => setVersionName(event.target.value)}
                placeholder="mvp-quality-pass"
              />
              <button
                type="button"
                disabled={!importedDataset || !quality?.ready_for_training || isCreatingVersion}
                onClick={handleCreateDatasetVersion}
              >
                {isCreatingVersion ? "导出中" : "创建数据集版本"}
              </button>
            </div>
            <div className="subset-controls" aria-label="版本类别子集">
              <span>类别子集</span>
              <div className="subset-grid">
                {classes.length === 0 ? (
                  <p className="empty-state">冻结类别子集前，请先创建项目类别。</p>
                ) : (
                  classes.map((classItem) => (
                    <label key={classItem.id}>
                      <input
                        type="checkbox"
                        checked={versionClassIds.includes(classItem.id)}
                        onChange={() => toggleVersionClass(classItem.id)}
                        disabled={!importedDataset}
                      />
                      <span style={{ background: classItem.color }} />
                      {classItem.name}
                    </label>
                  ))
                )}
              </div>
            </div>
          </div>

          {versionError ? <div className="error-banner">{versionError}</div> : null}

          <div className="version-list" aria-label="数据集版本">
            {versions.length === 0 ? (
              <p className="empty-state">导出的 YOLO 数据集版本将显示在这里。</p>
            ) : (
              versions.map((version) => (
                <div className="version-row" key={version.id}>
                  <div>
                    <strong>{version.name}</strong>
                    <span>{version.artifact_path}</span>
                  </div>
                  <span>
                    {version.split_counts.train}/{version.split_counts.val}/
                    {version.split_counts.test}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>
      </section>

      <section className="training-grid" aria-label="Training setup and runs">
        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Model Training</p>
              <h2>Training Setup</h2>
            </div>
            <Play size={20} />
          </div>

          <div className="training-form">
            <label htmlFor="training-model">Model preset or local weights</label>
            <input
              id="training-model"
              value={trainingModel}
              onChange={(event) => setTrainingModel(event.target.value)}
              disabled={versions.length === 0}
            />

            <div className="training-number-grid">
              <label htmlFor="training-epochs">
                Epochs
                <input
                  id="training-epochs"
                  type="number"
                  min={1}
                  max={1000}
                  value={trainingEpochs}
                  onChange={(event) => setTrainingEpochs(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
              <label htmlFor="training-imgsz">
                Image size
                <input
                  id="training-imgsz"
                  type="number"
                  min={32}
                  max={4096}
                  value={trainingImageSize}
                  onChange={(event) => setTrainingImageSize(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
              <label htmlFor="training-batch">
                Batch
                <input
                  id="training-batch"
                  type="number"
                  min={1}
                  max={256}
                  value={trainingBatchSize}
                  onChange={(event) => setTrainingBatchSize(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
            </div>

            <div className="training-number-grid">
              <label htmlFor="training-device">
                Device
                <input
                  id="training-device"
                  value={trainingDevice}
                  onChange={(event) => setTrainingDevice(event.target.value)}
                  placeholder={health?.devices.selected ?? "cpu"}
                  disabled={versions.length === 0}
                />
              </label>
            </div>

            <div className="augmentation-panel" aria-label="Augmentation strategy">
              <label htmlFor="augmentation-preset">
                Strategy name
                <input
                  id="augmentation-preset"
                  value={augmentationPreset}
                  onChange={(event) => setAugmentationPreset(event.target.value)}
                  disabled={versions.length === 0}
                />
              </label>
              <div className="augmentation-grid">
                <AugmentationNumber
                  id="aug-mosaic"
                  label="Mosaic"
                  value={augmentation.mosaic}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("mosaic", value)}
                />
                <AugmentationNumber
                  id="aug-mixup"
                  label="MixUp"
                  value={augmentation.mixup}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("mixup", value)}
                />
                <AugmentationNumber
                  id="aug-copy-paste"
                  label="Copy-Paste"
                  value={augmentation.copy_paste}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("copy_paste", value)}
                />
                <AugmentationNumber
                  id="aug-erasing"
                  label="Erasing"
                  value={augmentation.erasing}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("erasing", value)}
                />
                <AugmentationNumber
                  id="aug-scale"
                  label="Scale"
                  max={2}
                  value={augmentation.scale}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("scale", value)}
                />
                <AugmentationNumber
                  id="aug-fliplr"
                  label="Flip LR"
                  value={augmentation.fliplr}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("fliplr", value)}
                />
              </div>
            </div>

            <div className="toggle-row">
              <label>
                <input
                  type="checkbox"
                  checked={augmentation.gridmask}
                  onChange={(event) =>
                    setAugmentation((current) => ({
                      ...current,
                      gridmask: event.target.checked,
                    }))
                  }
                  disabled={versions.length === 0}
                />
                GridMask
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={trainingTta}
                  onChange={(event) => setTrainingTta(event.target.checked)}
                  disabled={versions.length === 0}
                />
                TTA
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={thresholdScan}
                  onChange={(event) => setThresholdScan(event.target.checked)}
                  disabled={versions.length === 0}
                />
                Auto threshold scan
              </label>
            </div>

            <button
              type="button"
              disabled={versions.length === 0 || isStartingRun}
              onClick={handleStartTrainingRun}
            >
              <Play size={16} />
              {isStartingRun ? "Starting" : "Start Training Run"}
            </button>
          </div>

          {trainingError ? <div className="error-banner">{trainingError}</div> : null}
          {trainingNotice ? <div className="success-banner">{trainingNotice}</div> : null}
        </section>

        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Experiments</p>
              <h2>Run History</h2>
              <span className="monitor-state">
                {hasActiveRun ? "Auto refresh on" : "Idle"}
              </span>
            </div>
            <Activity size={20} />
          </div>

          <div className="run-list" aria-label="Training runs">
            {runs.length === 0 ? (
              <p className="empty-state">Create a dataset version, then start a training run.</p>
            ) : (
              runs.map((run) => (
                <div className="run-row" key={run.id}>
                  <div className="run-row-heading">
                    <strong>Run #{run.id}</strong>
                    <span className={`run-status ${run.status}`}>{run.status}</span>
                  </div>
                  <span>{run.artifact_path}</span>
                  <span>
                    {String(run.config.model ?? "model")} | {String(run.config.epochs ?? "?")} epochs |{" "}
                    {run.device}
                  </span>
                  {Object.keys(run.latest_metrics).length > 0 ? (
                    <div className="metric-chips">
                      {Object.entries(run.latest_metrics).map(([name, value]) => (
                        <span key={name}>
                          {name}: {value.toFixed(3)}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {run.error_message ? <p className="run-error">{run.error_message}</p> : null}
                  <RunArtifactList summary={runArtifacts[run.id]} />
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => loadRunLogs(run.id)}
                  >
                    Load Logs
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => applyRunConfigToForm(run)}
                  >
                    Load Config
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={versions.length === 0 || hasActiveRun || isStartingRun}
                    onClick={() => handleRerunTrainingRun(run)}
                  >
                    Rerun
                  </button>
                  {isActiveRun(run.status) ? (
                    <button
                      type="button"
                      className="secondary-button danger-button"
                      disabled={cancellingRunId === run.id}
                      onClick={() => handleCancelTrainingRun(run.id)}
                    >
                      {cancellingRunId === run.id ? "Cancelling" : "Cancel Run"}
                    </button>
                  ) : null}
                  {runLogs[run.id] ? <pre className="log-preview">{runLogs[run.id]}</pre> : null}
                </div>
              ))
            )}
          </div>

          {summaryError ? <div className="error-banner">{summaryError}</div> : null}
          <ExperimentDashboard
            summary={runSummary}
            projectSummary={projectExperimentSummary}
          />
        </section>
      </section>

      <section className="panel prediction-panel" aria-label="Prediction analysis">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">Model Review</p>
            <h2>Prediction Analysis</h2>
            <span className="monitor-state">
              {hasActivePredictionJob ? "Auto refresh on" : "Idle"}
            </span>
          </div>
          <Radar size={20} />
        </div>

        <div className="prediction-controls">
          <label htmlFor="prediction-scope">Image scope</label>
          <input
            id="prediction-scope"
            value={predictionScope}
            onChange={(event) => setPredictionScope(event.target.value)}
            disabled={runs.length === 0}
          />
          <label htmlFor="prediction-confidence">Confidence threshold</label>
          <input
            id="prediction-confidence"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={predictionConfidence}
            onChange={(event) => setPredictionConfidence(Number(event.target.value))}
            disabled={runs.length === 0}
          />
          <label className="prediction-filter-toggle" htmlFor="use-image-filters-prediction">
            <input
              id="use-image-filters-prediction"
              type="checkbox"
              checked={useImageFiltersForPrediction}
              onChange={(event) => setUseImageFiltersForPrediction(event.target.checked)}
              disabled={runs.length === 0}
            />
            Use image filters
          </label>
          <span className="prediction-filter-summary">{predictionImageFilterSummary}</span>
          <button
            type="button"
            disabled={runs.length === 0 || isCreatingPrediction}
            onClick={handleCreatePredictionJob}
          >
            <Radar size={16} />
            {isCreatingPrediction ? "Running" : "Run Prediction Analysis"}
          </button>
          <label className="threshold-scan-field" htmlFor="prediction-thresholds">
            Scan thresholds
            <input
              id="prediction-thresholds"
              value={predictionThresholds}
              onChange={(event) => setPredictionThresholds(event.target.value)}
              disabled={runs.length === 0}
            />
          </label>
          <button
            type="button"
            className="secondary-button threshold-scan-button"
            disabled={runs.length === 0 || isCreatingThresholdScan}
            onClick={handleCreateThresholdScan}
          >
            <Radar size={16} />
            {isCreatingThresholdScan ? "Scanning" : "Run Threshold Scan"}
          </button>
        </div>

        <div className="prediction-filter-panel" aria-label="Prediction sample filters">
          <label htmlFor="prediction-filter-failure">
            Failure type
            <select
              id="prediction-filter-failure"
              value={predictionFilters.failure_type}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  failure_type: event.target.value as typeof predictionFilters.failure_type,
                }))
              }
              disabled={predictionJobs.length === 0}
            >
              {predictionFailureOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="prediction-filter-class">
            Prediction class
            <select
              id="prediction-filter-class"
              value={predictionFilters.class_id}
              onChange={(event) =>
                setPredictionFilters((current) => ({ ...current, class_id: event.target.value }))
              }
              disabled={predictionJobs.length === 0}
            >
              <option value="">All classes</option>
              {classes.map((classItem) => (
                <option key={classItem.id} value={classItem.id}>
                  {classItem.name}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="prediction-filter-conf-min">
            Min conf
            <input
              id="prediction-filter-conf-min"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={predictionFilters.confidence_min}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  confidence_min: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-conf-max">
            Max conf
            <input
              id="prediction-filter-conf-max"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={predictionFilters.confidence_max}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  confidence_max: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-platform">
            Prediction platform
            <input
              id="prediction-filter-platform"
              value={predictionFilters.platform}
              onChange={(event) =>
                setPredictionFilters((current) => ({ ...current, platform: event.target.value }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-alt-min">
            Min altitude
            <input
              id="prediction-filter-alt-min"
              type="number"
              value={predictionFilters.altitude_min}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  altitude_min: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-alt-max">
            Max altitude
            <input
              id="prediction-filter-alt-max"
              type="number"
              value={predictionFilters.altitude_max}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  altitude_max: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-time-min">
            Min time
            <input
              id="prediction-filter-time-min"
              type="number"
              value={predictionFilters.timestamp_min}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  timestamp_min: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <label htmlFor="prediction-filter-time-max">
            Max time
            <input
              id="prediction-filter-time-max"
              type="number"
              value={predictionFilters.timestamp_max}
              onChange={(event) =>
                setPredictionFilters((current) => ({
                  ...current,
                  timestamp_max: event.target.value,
                }))
              }
              disabled={predictionJobs.length === 0}
            />
          </label>
          <div className="prediction-filter-actions">
            <button
              type="button"
              disabled={predictionJobs.length === 0}
              onClick={handleApplyPredictionFilters}
            >
              Apply Sample Filters
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={predictionJobs.length === 0}
              onClick={handleResetPredictionFilters}
            >
              Reset
            </button>
          </div>
        </div>

        {predictionError ? <div className="error-banner">{predictionError}</div> : null}

        {predictionJobs[0] ? (
          <div className="prediction-summary">
            <Metric label="Images" value={predictionJobs[0].image_count.toLocaleString()} />
            <Metric label="Matched" value={predictionJobs[0].matched_count.toLocaleString()} />
            <Metric label="False +" value={predictionJobs[0].false_positive_count.toLocaleString()} />
            <Metric label="False -" value={predictionJobs[0].false_negative_count.toLocaleString()} />
            <Metric label="Confused" value={predictionJobs[0].class_confusion_count.toLocaleString()} />
          </div>
        ) : (
          <p className="empty-state">Start a prediction job from a completed or failed run to review outputs.</p>
        )}

        <div className="prediction-layout">
          <div className="prediction-list" aria-label="Prediction samples">
            {predictions.length === 0 ? (
              <p className="empty-state">Prediction and failure samples will appear here.</p>
            ) : (
              predictions.slice(0, 20).map((prediction) => (
                <div className="prediction-row" key={prediction.id}>
                  <div>
                    <strong>{prediction.failure_type}</strong>
                    <span>Image #{prediction.image_id} | Class #{prediction.class_id}</span>
                  </div>
                  <span>{prediction.confidence.toFixed(2)}</span>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => openPredictionImage(prediction)}
                  >
                    Open Image
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="prediction-jobs" aria-label="Prediction jobs">
            {predictionJobs.length === 0 ? null : (
              predictionJobs.map((job) => (
                <div className="run-row" key={job.id}>
                  <div className="run-row-heading">
                    <strong>Prediction #{job.id}</strong>
                    <span className={`run-status ${job.status}`}>{job.status}</span>
                  </div>
                  <span>{job.image_filters ? formatImageFilterSummary(job.image_filters, classById) : "All images in selected scope"}</span>
                  <span>{job.artifact_path}</span>
                  {job.error_message ? <p className="run-error">{job.error_message}</p> : null}
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => loadPredictionLogs(job.id)}
                  >
                    Load Prediction Logs
                  </button>
                  {predictionLogs[job.id] ? (
                    <pre className="log-preview">{predictionLogs[job.id]}</pre>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="panel export-panel" aria-label="Model export">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">Deployment Artifacts</p>
            <h2>Model Export</h2>
          </div>
          <Share2 size={20} />
        </div>

        {exportError ? <div className="error-banner">{exportError}</div> : null}

        <div className="export-grid">
          <ExportOption
            title=".pt Weights"
            format="pt"
            enabled={Boolean(latestRun && latestRun.status === "completed" && exportCapabilities?.pt_available)}
            reason={exportCapabilities?.reasons.pt}
            isCreating={isCreatingExport === "pt"}
            onCreate={handleCreateExport}
          />
          <ExportOption
            title="ONNX"
            format="onnx"
            enabled={Boolean(
              latestRun && latestRun.status === "completed" && exportCapabilities?.onnx_available,
            )}
            reason={exportCapabilities?.reasons.onnx}
            isCreating={isCreatingExport === "onnx"}
            onCreate={handleCreateExport}
          />
          <ExportOption
            title="TensorRT"
            format="tensorrt"
            enabled={Boolean(
              latestRun &&
                latestRun.status === "completed" &&
                exportCapabilities?.tensorrt_available,
            )}
            reason={exportCapabilities?.reasons.tensorrt}
            isCreating={isCreatingExport === "tensorrt"}
            onCreate={handleCreateExport}
          />
        </div>

        {exportCapabilities?.weights_path ? (
          <p className="export-source">Source weights: {exportCapabilities.weights_path}</p>
        ) : (
          <p className="empty-state">
            Complete a training run with `ultralytics/weights/best.pt` to enable export.
          </p>
        )}

        <div className="export-list" aria-label="Export artifacts">
          {exports.length === 0 ? (
            <p className="empty-state">Exported model artifacts will appear here.</p>
          ) : (
            exports.map((artifact) => (
              <div className="export-row" key={artifact.id}>
                <div>
                  <strong>{artifact.format.toUpperCase()} export #{artifact.id}</strong>
                  <span>{artifact.artifact_path || "No artifact path yet"}</span>
                </div>
                <span className={`run-status ${artifact.status}`}>{artifact.status}</span>
                {artifact.error_message ? (
                  <p className="run-error">{artifact.error_message}</p>
                ) : null}
              </div>
            ))
          )}
        </div>
      </section>

      <section className="workbench-grid" aria-label="标注工作台">
        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">项目标签</p>
              <h2>类别库</h2>
            </div>
            <Library size={20} />
          </div>

          <form className="class-form" onSubmit={handleCreateClass}>
            <label htmlFor="class-name">类别名称</label>
            <input
              id="class-name"
              value={className}
              disabled={!importedDataset}
              onChange={(event) => setClassName(event.target.value)}
              placeholder="target"
            />
            <label htmlFor="class-color">类别颜色</label>
            <div className="color-row">
              <input
                id="class-color"
                type="color"
                value={classColor}
                disabled={!importedDataset}
                onChange={(event) => setClassColor(event.target.value)}
                aria-label="类别颜色"
              />
              <button
                type="submit"
                disabled={!importedDataset || isCreatingClass || className.trim().length === 0}
              >
                创建类别
              </button>
            </div>
          </form>

          {classError ? <div className="error-banner">{classError}</div> : null}

          <div className="class-list" aria-label="可用类别">
            {classes.length === 0 ? (
              <p className="empty-state">请先导入数据集，再创建类别以绘制边界框。</p>
            ) : (
              classes.map((classItem) =>
                editingClassId === classItem.id ? (
                  <div className="class-edit-row" key={classItem.id}>
                    <input
                      value={classEditName}
                      onChange={(event) => setClassEditName(event.target.value)}
                      aria-label={`编辑类别名称 ${classItem.name}`}
                    />
                    <input
                      type="color"
                      value={classEditColor}
                      onChange={(event) => setClassEditColor(event.target.value)}
                      aria-label={`编辑类别颜色 ${classItem.name}`}
                    />
                    <div className="class-row-actions">
                      <button
                        type="button"
                        onClick={() => handleUpdateClass(classItem.id)}
                        disabled={isUpdatingClass || classEditName.trim().length === 0}
                      >
                        <Save size={15} />
                        保存
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={cancelEditClass}
                        disabled={isUpdatingClass}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="class-row" key={classItem.id}>
                    <button
                      type="button"
                      className={
                        classItem.id === selectedClassId ? "class-chip selected" : "class-chip"
                      }
                      onClick={() => setSelectedClassId(classItem.id)}
                    >
                      <span style={{ background: classItem.color }} />
                      {classItem.name}
                    </button>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`编辑 ${classItem.name}`}
                      title={`编辑 ${classItem.name}`}
                      onClick={() => beginEditClass(classItem)}
                    >
                      <Edit3 size={16} />
                    </button>
                  </div>
                ),
              )
            )}
          </div>
        </aside>

        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">数据集图像</p>
              <h2>图像浏览器</h2>
            </div>
            <ImageIcon size={20} />
          </div>

          <div className="image-filter-panel" aria-label="图像筛选器">
            <label htmlFor="filter-platform">平台</label>
            <input
              id="filter-platform"
              value={imageFilters.platform}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({ ...current, platform: event.target.value }))
              }
              placeholder="iris"
            />
            <label htmlFor="filter-label-status">标注状态</label>
            <select
              id="filter-label-status"
              value={imageFilters.label_status}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({
                  ...current,
                  label_status: event.target.value as "all" | "annotated" | "unannotated",
                }))
              }
            >
              <option value="all">全部</option>
              <option value="annotated">已标注</option>
              <option value="unannotated">未标注</option>
            </select>
            <label htmlFor="filter-class">类别</label>
            <select
              id="filter-class"
              value={imageFilters.class_id}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({ ...current, class_id: event.target.value }))
              }
            >
              <option value="">全部类别</option>
              {classes.map((classItem) => (
                <option key={classItem.id} value={classItem.id}>
                  {classItem.name}
                </option>
              ))}
            </select>
            <label htmlFor="filter-failure-type">识别结果</label>
            <select
              id="filter-failure-type"
              value={imageFilters.failure_type}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({
                  ...current,
                  failure_type: event.target.value as typeof imageFilters.failure_type,
                }))
              }
            >
              {predictionFailureOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.value === "all" ? "全部识别结果" : formatFailureType(option.value)}
                </option>
              ))}
            </select>
            <label htmlFor="filter-edge-tag">边缘案例标签</label>
            <input
              id="filter-edge-tag"
              value={imageFilters.edge_tag}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({ ...current, edge_tag: event.target.value }))
              }
              placeholder="occluded"
            />
            <div className="range-row">
              <label htmlFor="filter-altitude-min">
                最低高度
                <input
                  id="filter-altitude-min"
                  type="number"
                  value={imageFilters.altitude_min}
                  disabled={!importedDataset}
                  onChange={(event) =>
                    setImageFilters((current) => ({
                      ...current,
                      altitude_min: event.target.value,
                    }))
                  }
                />
              </label>
              <label htmlFor="filter-altitude-max">
                最高高度
                <input
                  id="filter-altitude-max"
                  type="number"
                  value={imageFilters.altitude_max}
                  disabled={!importedDataset}
                  onChange={(event) =>
                    setImageFilters((current) => ({
                      ...current,
                      altitude_max: event.target.value,
                    }))
                  }
                />
              </label>
            </div>
            <div className="filter-actions">
              <button type="button" disabled={!importedDataset} onClick={handleApplyImageFilters}>
                应用筛选
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={!importedDataset}
                onClick={handleResetImageFilters}
              >
                重置
              </button>
            </div>
          </div>

          {imageFilterError ? <div className="error-banner">{imageFilterError}</div> : null}

          <div className="image-pager" aria-label="图像分页">
            <span>
              第 {imagePageStart}-{imagePageEnd} 张，共 {imagePage.total} 张
            </span>
            <div>
              <button
                type="button"
                className="secondary-button"
                disabled={!importedDataset || !canPageImagesPrevious}
                onClick={handlePreviousImagePage}
              >
                上一页
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={!importedDataset || !canPageImagesNext}
                onClick={handleNextImagePage}
              >
                下一页
              </button>
            </div>
          </div>

          <div className="image-list" aria-label="已导入图像">
            {images.length === 0 ? (
              <p className="empty-state">导入的图像将显示在这里。</p>
            ) : (
              images.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  className={image.id === selectedImageId ? "image-row selected" : "image-row"}
                  onClick={() => {
                    setActiveReview(null);
                    setSelectedImageId(image.id);
                  }}
                >
                  <strong>{image.relative_path}</strong>
                  <span>
                    {image.platform ?? "未知平台"} | {formatImageAltitude(image.altitude)} |{" "}
                    {image.annotation_count} 个边界框
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="panel annotation-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">标注与审查</p>
              <h2>标注</h2>
            </div>
            <Box size={20} />
          </div>

          <div className="annotation-readiness" aria-label="标注就绪状态">
            {annotationReadinessSteps.map((step) => (
              <span
                key={step.label}
                className={step.complete ? "readiness-step complete" : "readiness-step"}
              >
                {step.complete ? "就绪" : "待完成"} | {step.label}
              </span>
            ))}
          </div>

          {selectedImage ? (
            <div className="annotation-layout">
              <div className="viewer-wrap">
                <div className="image-stage">
                  <img src={selectedImage.image_url} alt={selectedImage.relative_path} />
                  <svg
                    ref={annotationCanvasRef}
                    aria-label="标注画布"
                    className={selectedClass ? "annotation-overlay drawable" : "annotation-overlay"}
                    viewBox="0 0 1 1"
                    preserveAspectRatio="none"
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerCancel={() => setDragState(null)}
                  >
                    {showGroundTruthLayer
                      ? annotations.map((annotation) => (
                          <BoxRect
                            key={annotation.local_id}
                            annotation={annotation}
                            color={resolveClassColor(annotation, classById)}
                            selected={annotation.local_id === selectedAnnotationId}
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

              <div className="box-list">
                <div className="box-list-heading">
                  <strong>边界框</strong>
                  <div className="box-list-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => copyAdjacentAnnotations("previous")}
                      disabled={images.findIndex((image) => image.id === selectedImage.id) <= 0}
                    >
                      复制上一张
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => copyAdjacentAnnotations("next")}
                      disabled={
                        images.findIndex((image) => image.id === selectedImage.id) >=
                        images.length - 1
                      }
                    >
                      复制下一张
                    </button>
                    <button
                      type="button"
                      onClick={handleSaveAnnotations}
                      disabled={isSavingAnnotations || !selectedImage}
                    >
                      <Save size={16} />
                      {isSavingAnnotations ? "保存中" : "保存标注"}
                    </button>
                  </div>
                </div>

                {annotationError ? <div className="error-banner">{annotationError}</div> : null}

                {activeReview && activeReview.image.id === selectedImage.id ? (
                  <div className="review-banner">
                    <div className="review-banner-heading">
                      <div>
                        <strong>预测结果叠加</strong>
                        <span>
                          匹配正确 {activeReview.counts.matched ?? 0} | 误报{" "}
                          {activeReview.counts.false_positive ?? 0} | 漏报{" "}
                          {activeReview.counts.false_negative ?? 0} | 类别混淆{" "}
                          {activeReview.counts.class_confusion ?? 0}
                        </span>
                      </div>
                      <div className="layer-toggles" aria-label="标注审查图层">
                        <label>
                          <input
                            type="checkbox"
                            checked={showGroundTruthLayer}
                            onChange={(event) => setShowGroundTruthLayer(event.target.checked)}
                          />
                          真实标注（GT）
                        </label>
                        <label>
                          <input
                            type="checkbox"
                            checked={showPredictionLayer}
                            onChange={(event) => setShowPredictionLayer(event.target.checked)}
                          />
                          模型预测（Pred）
                        </label>
                      </div>
                    </div>
                    <div className="review-legend" aria-label="预测结果图例">
                      <span className="legend matched">匹配正确</span>
                      <span className="legend false-positive">误报</span>
                      <span className="legend false-negative">漏报</span>
                    </div>
                    <div className="review-actions" aria-label="预测修正操作">
                      {activeReview.predictions
                        .filter((prediction) => prediction.failure_type !== "matched")
                        .map((prediction) => (
                          <div className="review-action-row" key={prediction.id}>
                            <div>
                              <strong>{formatFailureType(prediction.failure_type)}</strong>
                              <span>
                                类别 #{prediction.class_id} | 置信度 {prediction.confidence.toFixed(2)}
                              </span>
                            </div>
                            {prediction.failure_type === "false_positive" ? (
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => addPredictionAsAnnotation(prediction)}
                                disabled={annotations.some(
                                  (annotation) =>
                                    annotation.local_id === predictionDraftId(prediction.id),
                                )}
                              >
                                添加为标注
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => markFalseNegativeReviewed(prediction)}
                                disabled={!prediction.matched_annotation_id}
                              >
                                标记为已审查
                              </button>
                            )}
                          </div>
                        ))}
                    </div>
                  </div>
                ) : null}

                {annotations.length === 0 ? (
                  <p className="empty-state">
                    {annotationReady
                      ? "在图像上拖动以添加边界框。"
                      : annotationGuidance(importedDataset, classes.length, selectedImage, selectedClass)}
                  </p>
                ) : (
                  annotations.map((annotation, index) => (
                    <div
                      className={
                        annotation.local_id === selectedAnnotationId
                          ? "box-editor selected"
                          : "box-editor"
                      }
                      key={annotation.local_id}
                      onClick={() => setSelectedAnnotationId(annotation.local_id)}
                    >
                      <div className="box-editor-title">
                        <span
                          style={{ background: resolveClassColor(annotation, classById) }}
                        />
                        <strong>
                          {classById.get(annotation.class_id)?.name ??
                            annotation.class_name ??
                            `类别 ${annotation.class_id}`}
                        </strong>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`删除边界框 ${index + 1}`}
                          onClick={() => deleteAnnotation(annotation.local_id)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                      <div className="nudge-controls" aria-label={`移动边界框 ${index + 1}`}>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`向左移动边界框 ${index + 1}`}
                          title="向左移动"
                          onClick={() => nudgeAnnotation(annotation.local_id, -0.01, 0)}
                        >
                          <ArrowLeft size={15} />
                        </button>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`向上移动边界框 ${index + 1}`}
                          title="向上移动"
                          onClick={() => nudgeAnnotation(annotation.local_id, 0, -0.01)}
                        >
                          <ArrowUp size={15} />
                        </button>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`向下移动边界框 ${index + 1}`}
                          title="向下移动"
                          onClick={() => nudgeAnnotation(annotation.local_id, 0, 0.01)}
                        >
                          <ArrowDown size={15} />
                        </button>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`向右移动边界框 ${index + 1}`}
                          title="向右移动"
                          onClick={() => nudgeAnnotation(annotation.local_id, 0.01, 0)}
                        >
                          <ArrowRight size={15} />
                        </button>
                      </div>

                      <label
                        className="box-class-control"
                        htmlFor={`box-class-${annotation.local_id}`}
                      >
                        类别
                        <select
                          id={`box-class-${annotation.local_id}`}
                          aria-label={`边界框 ${index + 1} 类别`}
                          value={annotation.class_id}
                          onChange={(event) =>
                            updateAnnotationClass(annotation.local_id, Number(event.target.value))
                          }
                          disabled={classes.length === 0}
                        >
                          {classes.map((classItem) => (
                            <option key={classItem.id} value={classItem.id}>
                              {classItem.name}
                            </option>
                          ))}
                        </select>
                      </label>

                      <div className="geometry-grid" aria-label={`边界框 ${index + 1} 几何参数`}>
                        <label htmlFor={`box-x-${annotation.local_id}`}>
                          X
                          <input
                            id={`box-x-${annotation.local_id}`}
                            type="number"
                            min={0}
                            max={1}
                            step={0.001}
                            value={formatGeometryValue(annotation.x_center)}
                            onChange={(event) =>
                              updateAnnotationGeometry(
                                annotation.local_id,
                                "x_center",
                                Number(event.target.value),
                              )
                            }
                          />
                        </label>
                        <label htmlFor={`box-y-${annotation.local_id}`}>
                          Y
                          <input
                            id={`box-y-${annotation.local_id}`}
                            type="number"
                            min={0}
                            max={1}
                            step={0.001}
                            value={formatGeometryValue(annotation.y_center)}
                            onChange={(event) =>
                              updateAnnotationGeometry(
                                annotation.local_id,
                                "y_center",
                                Number(event.target.value),
                              )
                            }
                          />
                        </label>
                        <label htmlFor={`box-w-${annotation.local_id}`}>
                          W
                          <input
                            id={`box-w-${annotation.local_id}`}
                            type="number"
                            min={0.001}
                            max={1}
                            step={0.001}
                            value={formatGeometryValue(annotation.width)}
                            onChange={(event) =>
                              updateAnnotationGeometry(
                                annotation.local_id,
                                "width",
                                Number(event.target.value),
                              )
                            }
                          />
                        </label>
                        <label htmlFor={`box-h-${annotation.local_id}`}>
                          H
                          <input
                            id={`box-h-${annotation.local_id}`}
                            type="number"
                            min={0.001}
                            max={1}
                            step={0.001}
                            value={formatGeometryValue(annotation.height)}
                            onChange={(event) =>
                              updateAnnotationGeometry(
                                annotation.local_id,
                                "height",
                                Number(event.target.value),
                              )
                            }
                          />
                        </label>
                      </div>

                      <label htmlFor={`track-${annotation.local_id}`}>目标轨迹 ID</label>
                      <input
                        id={`track-${annotation.local_id}`}
                        value={annotation.track_id ?? ""}
                        onChange={(event) =>
                          updateAnnotation(annotation.local_id, { track_id: event.target.value })
                        }
                      />

                      <label htmlFor={`tags-${annotation.local_id}`}>边缘案例标签</label>
                      <input
                        id={`tags-${annotation.local_id}`}
                        value={(annotation.edge_tags ?? []).join(", ")}
                        onChange={(event) =>
                          updateAnnotation(annotation.local_id, {
                            edge_tags: parseTags(event.target.value),
                          })
                        }
                        placeholder="occluded, small"
                      />
                      <div className="edge-tag-presets" aria-label={`边界框 ${index + 1} 边缘案例标签预设`}>
                        {edgeTagPresets.map((tag) => {
                          const isSelected = (annotation.edge_tags ?? []).includes(tag);
                          return (
                            <button
                              type="button"
                              key={tag}
                              className={isSelected ? "tag-chip selected" : "tag-chip"}
                              aria-pressed={isSelected}
                              onClick={() =>
                                updateAnnotation(annotation.local_id, {
                                  edge_tags: toggleTag(annotation.edge_tags, tag),
                                })
                              }
                            >
                              {formatEdgeTag(tag)}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <p className="empty-state">
              {annotationGuidance(importedDataset, classes.length, selectedImage, selectedClass)}
            </p>
          )}
        </section>
      </section>
    </main>
  );
}

function formatAltitude(minimum: number | null, maximum: number | null) {
  if (minimum === null || maximum === null) {
    return "暂无数据";
  }

  return `${minimum.toFixed(1)}-${maximum.toFixed(1)}m`;
}

function formatImageAltitude(altitude: number | null) {
  return altitude === null ? "高度未知" : `${altitude.toFixed(1)}m`;
}

function formatGeometryValue(value: number) {
  return Number(value.toFixed(4));
}

function savedDatasetValue(projectId: number, datasetId: number) {
  return `${projectId}:${datasetId}`;
}

function firstSavedDatasetValue(projects: ProjectSummary[]) {
  const project = projects.find((item) => item.datasets.length > 0);
  const dataset = project?.datasets[0];
  return project && dataset ? savedDatasetValue(project.id, dataset.id) : "";
}

function savedDatasetFromValue(projects: ProjectSummary[], value: string) {
  const [projectId, datasetId] = value.split(":").map((part) => Number(part));
  if (!projectId || !datasetId) {
    return null;
  }
  const project = projects.find((item) => item.id === projectId);
  const dataset = project?.datasets.find((item) => item.id === datasetId);
  return project && dataset ? { project, dataset } : null;
}

function toImageFilterRequest(filters: {
  platform: string;
  label_status: "all" | "annotated" | "unannotated";
  class_id: string;
  edge_tag: string;
  failure_type: PredictionFailureType;
  altitude_min: string;
  altitude_max: string;
}) {
  return {
    platform: filters.platform.trim() || undefined,
    label_status: filters.label_status,
    class_id: filters.class_id ? Number(filters.class_id) : undefined,
    edge_tag: filters.edge_tag.trim() || undefined,
    failure_type: filters.failure_type,
    altitude_min: filters.altitude_min === "" ? undefined : Number(filters.altitude_min),
    altitude_max: filters.altitude_max === "" ? undefined : Number(filters.altitude_max),
  };
}

function activeImageFilterRequest(filters: DatasetImageFilters): DatasetImageFilters | undefined {
  if (
    !filters.platform &&
    (filters.label_status ?? "all") === "all" &&
    filters.class_id === undefined &&
    !filters.edge_tag &&
    (filters.failure_type ?? "all") === "all" &&
    filters.altitude_min === undefined &&
    filters.altitude_max === undefined
  ) {
    return undefined;
  }
  return filters;
}

function formatImageFilterSummary(
  filters: DatasetImageFilters,
  classById: Map<number, ProjectClass>,
) {
  const parts: string[] = [];
  if (filters.platform) {
    parts.push(`平台 ${filters.platform}`);
  }
  if (filters.label_status && filters.label_status !== "all") {
    parts.push(filters.label_status === "annotated" ? "已标注" : "未标注");
  }
  if (filters.class_id !== undefined) {
    parts.push(`类别 ${classById.get(filters.class_id)?.name ?? filters.class_id}`);
  }
  if (filters.edge_tag) {
    parts.push(`标签 ${formatEdgeTag(filters.edge_tag)}`);
  }
  if (filters.failure_type && filters.failure_type !== "all") {
    parts.push(formatFailureType(filters.failure_type));
  }
  if (filters.altitude_min !== undefined || filters.altitude_max !== undefined) {
    const minimum = filters.altitude_min ?? 0;
    const maximum = filters.altitude_max ?? "最大值";
    parts.push(`高度 ${minimum}-${maximum}`);
  }
  return parts.length > 0
    ? `图像筛选：${parts.join(" | ")}`
    : "所选范围内的全部图像";
}

function toPredictionFilterRequest(filters: ReturnType<typeof defaultPredictionFilters>): PredictionFilters {
  return {
    failure_type: filters.failure_type,
    class_id: filters.class_id ? Number(filters.class_id) : undefined,
    confidence_min: filters.confidence_min === "" ? undefined : Number(filters.confidence_min),
    confidence_max: filters.confidence_max === "" ? undefined : Number(filters.confidence_max),
    platform: filters.platform.trim() || undefined,
    altitude_min: filters.altitude_min === "" ? undefined : Number(filters.altitude_min),
    altitude_max: filters.altitude_max === "" ? undefined : Number(filters.altitude_max),
    timestamp_min: filters.timestamp_min === "" ? undefined : Number(filters.timestamp_min),
    timestamp_max: filters.timestamp_max === "" ? undefined : Number(filters.timestamp_max),
  };
}

function stringConfig(config: Record<string, unknown>, key: string, fallback: string) {
  const value = config[key];
  return typeof value === "string" ? value : fallback;
}

function numberConfig(config: Record<string, unknown>, key: string, fallback: number) {
  const value = config[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function booleanConfig(config: Record<string, unknown>, key: string, fallback: boolean) {
  const value = config[key];
  return typeof value === "boolean" ? value : fallback;
}

function augmentationConfig(
  config: Record<string, unknown>,
  fallback: TrainingAugmentationConfig,
): TrainingAugmentationConfig {
  const raw = config.augmentation;
  const source =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, unknown>)
      : {};
  return {
    mosaic: numberConfig(source, "mosaic", fallback.mosaic),
    mixup: numberConfig(source, "mixup", fallback.mixup),
    copy_paste: numberConfig(source, "copy_paste", fallback.copy_paste),
    hsv_h: numberConfig(source, "hsv_h", fallback.hsv_h),
    hsv_s: numberConfig(source, "hsv_s", fallback.hsv_s),
    hsv_v: numberConfig(source, "hsv_v", fallback.hsv_v),
    translate: numberConfig(source, "translate", fallback.translate),
    scale: numberConfig(source, "scale", fallback.scale),
    fliplr: numberConfig(source, "fliplr", fallback.fliplr),
    erasing: numberConfig(source, "erasing", fallback.erasing),
    gridmask: booleanConfig(source, "gridmask", fallback.gridmask),
  };
}

function parseTags(value: string) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function parseThresholdList(value: string) {
  const parts = value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) {
    throw new Error("Enter at least one confidence threshold.");
  }
  if (parts.length > 20) {
    throw new Error("Threshold scan supports up to 20 values.");
  }

  const thresholds = parts.map((part) => Number(part));
  if (thresholds.some((threshold) => Number.isNaN(threshold))) {
    throw new Error("Threshold scan values must be numbers.");
  }
  if (thresholds.some((threshold) => threshold < 0 || threshold > 1)) {
    throw new Error("Threshold scan values must be between 0 and 1.");
  }

  return Array.from(new Set(thresholds.map((threshold) => Number(threshold.toFixed(4))))).sort(
    (left, right) => left - right,
  );
}

function mergePredictionJobs(incoming: PredictionJob[], existing: PredictionJob[]) {
  const byId = new Map<number, PredictionJob>();
  [...incoming, ...existing].forEach((job) => byId.set(job.id, job));
  return Array.from(byId.values()).sort((left, right) => right.id - left.id);
}

function toDraftBox(annotation: Annotation, index: number): DraftBox {
  return {
    ...annotation,
    local_id: annotation.id ? `annotation-${annotation.id}` : `loaded-${index}`,
    track_id: annotation.track_id ?? "",
    edge_tags: annotation.edge_tags ?? [],
  };
}

function predictionDraftId(predictionId: number) {
  return `prediction-${predictionId}`;
}

function predictionToDraftBox(
  prediction: Prediction,
  classInfo: ProjectClass | undefined,
): DraftBox {
  return {
    class_id: prediction.class_id,
    class_name: classInfo?.name,
    class_color: classInfo?.color,
    x_center: prediction.x_center,
    y_center: prediction.y_center,
    width: prediction.width,
    height: prediction.height,
    local_id: predictionDraftId(prediction.id),
    track_id: "",
    edge_tags: ["false_positive", "reviewed_prediction"],
  };
}

function toAnnotationWrite(annotation: DraftBox): AnnotationWrite {
  return {
    class_id: annotation.class_id,
    x_center: roundGeometry(annotation.x_center),
    y_center: roundGeometry(annotation.y_center),
    width: roundGeometry(annotation.width),
    height: roundGeometry(annotation.height),
    track_id: annotation.track_id?.trim() ? annotation.track_id.trim() : null,
    edge_tags: annotation.edge_tags ?? [],
  };
}

function getRelativePoint(event: PointerEvent<SVGElement>) {
  const svg = event.currentTarget.ownerSVGElement ?? event.currentTarget;
  return getRelativePointFromClient(svg, event.clientX, event.clientY);
}

function getRelativePointFromClient(svg: SVGElement, rawClientX: number, rawClientY: number) {
  const rect = svg.getBoundingClientRect();
  const clientX = Number.isFinite(rawClientX) ? rawClientX : rect.left;
  const clientY = Number.isFinite(rawClientY) ? rawClientY : rect.top;
  return {
    x: clamp((clientX - rect.left) / rect.width),
    y: clamp((clientY - rect.top) / rect.height),
  };
}

function rectangleToAnnotation(startX: number, startY: number, endX: number, endY: number) {
  const left = Math.min(startX, endX);
  const right = Math.max(startX, endX);
  const top = Math.min(startY, endY);
  const bottom = Math.max(startY, endY);
  const width = right - left;
  const height = bottom - top;

  if (width < 0.005 || height < 0.005) {
    return null;
  }

  return {
    x_center: left + width / 2,
    y_center: top + height / 2,
    width,
    height,
  };
}

function edgesToAnnotation(left: number, top: number, right: number, bottom: number) {
  const clampedLeft = clamp(left);
  const clampedTop = clamp(top);
  const clampedRight = clamp(right);
  const clampedBottom = clamp(bottom);
  const width = clampDimension(clampedRight - clampedLeft);
  const height = clampDimension(clampedBottom - clampedTop);
  return {
    x_center: clampCenter(clampedLeft + width / 2, width),
    y_center: clampCenter(clampedTop + height / 2, height),
    width,
    height,
  };
}

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

function clampDimension(value: number) {
  return Math.min(1, Math.max(0.001, value));
}

function clampCenter(value: number, size: number) {
  const halfSize = Math.min(0.5, Math.max(0, size / 2));
  return Math.min(1 - halfSize, Math.max(halfSize, value));
}

function roundGeometry(value: number) {
  return Number(value.toFixed(6));
}

function setPointerCaptureSafe(element: Element | null, pointerId: number) {
  if (
    element instanceof SVGElement &&
    typeof element.setPointerCapture === "function"
  ) {
    element.setPointerCapture(pointerId);
  }
}

function releasePointerCaptureSafe(element: Element, pointerId: number) {
  if (
    element instanceof SVGElement &&
    typeof element.hasPointerCapture === "function" &&
    typeof element.releasePointerCapture === "function" &&
    element.hasPointerCapture(pointerId)
  ) {
    element.releasePointerCapture(pointerId);
  }
}

function mergeTags(existing: string[] | undefined, tags: string[]) {
  return Array.from(new Set([...(existing ?? []), ...tags]));
}

function toggleTag(existing: string[] | undefined, tag: string) {
  const tags = existing ?? [];
  if (tags.includes(tag)) {
    return tags.filter((item) => item !== tag);
  }
  return [...tags, tag];
}

function isActiveRun(status: string) {
  return activeRunStatuses.has(status);
}

function isActivePredictionJob(status: string) {
  return activePredictionStatuses.has(status);
}

function resolveClassColor(annotation: DraftBox, classById: Map<number, ProjectClass>) {
  return classById.get(annotation.class_id)?.color ?? annotation.class_color ?? defaultClassColor;
}

function BoxRect(props: {
  annotation: DraftBox;
  color: string;
  selected: boolean;
  onPointerDown: (event: PointerEvent<SVGElement>, annotation: DraftBox) => void;
  onResizePointerDown: (
    event: PointerEvent<SVGElement>,
    annotation: DraftBox,
    handle: BoxResizeHandle,
  ) => void;
}) {
  const { annotation, color, selected, onPointerDown, onResizePointerDown } = props;
  const left = annotation.x_center - annotation.width / 2;
  const top = annotation.y_center - annotation.height / 2;
  const right = annotation.x_center + annotation.width / 2;
  const bottom = annotation.y_center + annotation.height / 2;
  const handles: Array<{ handle: BoxResizeHandle; x: number; y: number }> = [
    { handle: "top-left", x: left, y: top },
    { handle: "top-right", x: right, y: top },
    { handle: "bottom-left", x: left, y: bottom },
    { handle: "bottom-right", x: right, y: bottom },
  ];

  return (
    <g
      className={selected ? "annotation-box selected" : "annotation-box"}
      onPointerDown={(event) => onPointerDown(event, annotation)}
      role="button"
      aria-label={`选择边界框 ${annotation.class_name ?? annotation.class_id}`}
      tabIndex={0}
      data-testid={`annotation-box-${annotation.local_id}`}
    >
      <rect
        x={left}
        y={top}
        width={annotation.width}
        height={annotation.height}
        fill="transparent"
        stroke="transparent"
        strokeWidth={0.025}
        vectorEffect="non-scaling-stroke"
        onPointerDown={(event) => onPointerDown(event, annotation)}
      />
      <rect
        x={left}
        y={top}
        width={annotation.width}
        height={annotation.height}
        fill={selected ? "rgba(31, 111, 120, 0.08)" : "transparent"}
        stroke={color}
        strokeWidth={selected ? 0.007 : 0.004}
        vectorEffect="non-scaling-stroke"
        onPointerDown={(event) => onPointerDown(event, annotation)}
      />
      {selected
        ? handles.map((handle) => (
            <rect
              key={handle.handle}
              className={`resize-handle ${handle.handle}`}
              x={handle.x - 0.012}
              y={handle.y - 0.012}
              width={0.024}
              height={0.024}
              rx={0.004}
              fill="#ffffff"
              stroke={color}
              strokeWidth={0.004}
              vectorEffect="non-scaling-stroke"
              role="button"
              aria-label={`调整边界框 ${annotation.class_name ?? annotation.class_id} ${handle.handle.replace(
                "-",
                " ",
              )}`}
              tabIndex={0}
              data-testid={`resize-handle-${annotation.local_id}-${handle.handle}`}
              onPointerDown={(event) => onResizePointerDown(event, annotation, handle.handle)}
            />
          ))
        : null}
    </g>
  );
}

function DragRect(props: { dragState: DragState; color?: string }) {
  const { dragState, color = defaultClassColor } = props;
  const draft = rectangleToAnnotation(
    dragState.startX,
    dragState.startY,
    dragState.currentX,
    dragState.currentY,
  );

  if (!draft) {
    return null;
  }

  return (
    <rect
      x={draft.x_center - draft.width / 2}
      y={draft.y_center - draft.height / 2}
      width={draft.width}
      height={draft.height}
      fill="rgba(255, 255, 255, 0.16)"
      stroke={color}
      strokeDasharray="0.018 0.012"
      strokeWidth={0.004}
      vectorEffect="non-scaling-stroke"
    />
  );
}

function PredictionRect(props: { prediction: Prediction; className?: string }) {
  const { prediction, className } = props;
  const color =
    prediction.failure_type === "matched"
      ? "#16a34a"
      : prediction.failure_type === "false_negative"
        ? "#f97316"
        : prediction.failure_type === "class_confusion"
          ? "#eab308"
          : "#dc2626";
  const left = prediction.x_center - prediction.width / 2;
  const top = prediction.y_center - prediction.height / 2;
  const label = `${formatFailureType(prediction.failure_type)} | ${
    className ?? `类别 ${prediction.class_id}`
  }${prediction.failure_type === "false_negative" ? "" : ` ${prediction.confidence.toFixed(2)}`}`;
  const labelX = clamp(left);
  const labelY = clamp(top - 0.018);

  return (
    <g>
      <title>{label}</title>
      <rect
        x={left}
        y={top}
        width={prediction.width}
        height={prediction.height}
        fill="transparent"
        stroke={color}
        strokeDasharray={prediction.failure_type === "matched" ? "0" : "0.02 0.012"}
        strokeWidth={0.006}
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={labelX}
        y={labelY}
        fill={color}
        fontSize={0.022}
        fontWeight={900}
        paintOrder="stroke"
        stroke="#111820"
        strokeWidth={0.006}
      >
        {label}
      </text>
    </g>
  );
}

function StatusTile(props: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="status-tile">
      {props.icon}
      <div>
        <span>{props.label}</span>
        <strong>{props.value}</strong>
      </div>
    </div>
  );
}

function ExportOption(props: {
  title: string;
  format: string;
  enabled: boolean;
  reason?: string;
  isCreating: boolean;
  onCreate: (format: string) => void;
}) {
  const { title, format, enabled, reason, isCreating, onCreate } = props;
  return (
    <div className="export-option">
      <div>
        <strong>{title}</strong>
        <span>{enabled ? "Ready" : reason ?? "Waiting for a completed run"}</span>
      </div>
      <button
        type="button"
        className="secondary-button"
        disabled={!enabled || isCreating}
        onClick={() => onCreate(format)}
      >
        {isCreating ? "Exporting" : `Export ${format.toUpperCase()}`}
      </button>
    </div>
  );
}

function RunArtifactList(props: { summary?: TrainingRunArtifactSummary }) {
  const { summary } = props;
  const visibleItems = summary?.items.slice(0, 8) ?? [];

  return (
    <div className="run-artifacts" aria-label="Run artifacts">
      <div className="run-artifacts-heading">
        <strong>Run Artifacts</strong>
        <span>{summary ? `${summary.total_count} files` : "Loading"}</span>
      </div>
      {!summary ? (
        <p className="empty-state">Artifacts pending.</p>
      ) : visibleItems.length === 0 ? (
        <p className="empty-state">No files have been written yet.</p>
      ) : (
        <>
          <div className="artifact-list">
            {visibleItems.map((item) => (
              <div className="artifact-item" key={item.relative_path}>
                <span className="artifact-category">{item.category}</span>
                <span className="artifact-path" title={item.relative_path}>
                  {item.relative_path}
                </span>
                <span className="artifact-size">{formatBytes(item.size_bytes)}</span>
              </div>
            ))}
          </div>
          {summary.total_count > visibleItems.length ? (
            <span className="artifact-more">+{summary.total_count - visibleItems.length} more</span>
          ) : null}
        </>
      )}
    </div>
  );
}

function AugmentationNumber(props: {
  id: string;
  label: string;
  value: number;
  disabled: boolean;
  onChange: (value: number) => void;
  max?: number;
}) {
  const { id, label, value, disabled, onChange, max = 1 } = props;
  return (
    <label htmlFor={id}>
      {label}
      <input
        id={id}
        type="number"
        min={0}
        max={max}
        step={0.05}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ExperimentDashboard(props: {
  summary: RunExperimentSummary | null;
  projectSummary: ProjectExperimentSummary | null;
}) {
  const { summary, projectSummary } = props;
  const hasData =
    summary &&
    (summary.metric_series.length > 0 ||
      summary.class_outcomes.length > 0 ||
      summary.confusion_matrix.length > 0 ||
      summary.threshold_scan.length > 0);
  const hasComparison = Boolean(projectSummary?.runs.length);

  return (
    <div className="experiment-dashboard" aria-label="Experiment dashboard">
      <div className="dashboard-heading">
        <div>
          <strong>Experiment Dashboard</strong>
          <span>
            {summary?.latest_prediction_job_id
              ? `Latest prediction #${summary.latest_prediction_job_id}`
              : "Waiting for run evidence"}
          </span>
        </div>
      </div>

      {!hasData ? (
        <p className="empty-state">Metrics, class outcomes, and threshold scans will appear here.</p>
      ) : (
        <>
          <RunComparisonTable rows={projectSummary?.runs ?? []} />

          <div className="curve-grid">
            {summary.metric_series.slice(0, 4).map((series) => (
              <MetricCurve key={series.name} series={series} />
            ))}
          </div>

          <div className="analysis-grid">
            <ClassOutcomeTable rows={summary.class_outcomes} />
            <ConfusionMatrix cells={summary.confusion_matrix} />
          </div>

          <ThresholdScanTable
            rows={summary.threshold_scan}
            recommendation={summary.threshold_recommendation}
          />
        </>
      )}
      {!hasData && hasComparison ? <RunComparisonTable rows={projectSummary?.runs ?? []} /> : null}
    </div>
  );
}

function RunComparisonTable(props: { rows: ProjectExperimentSummary["runs"] }) {
  return (
    <div className="analysis-panel run-comparison-panel">
      <strong>Run Comparison</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">Completed runs will appear here for comparison.</p>
      ) : (
        <div className="compact-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Model</th>
                <th>Epochs</th>
                <th>mAP50</th>
                <th>Box loss</th>
                <th>Matched</th>
                <th>False +</th>
                <th>False -</th>
                <th>Best F1</th>
                <th>Best Conf</th>
                <th>Artifact</th>
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row) => (
                <tr key={row.run_id}>
                  <td>#{row.run_id}</td>
                  <td>{row.status}</td>
                  <td title={row.model}>{row.model}</td>
                  <td>{row.epochs ?? "n/a"}</td>
                  <td>{row.map50 === null ? "n/a" : row.map50.toFixed(3)}</td>
                  <td>{row.box_loss === null ? "n/a" : row.box_loss.toFixed(3)}</td>
                  <td>{row.matched}</td>
                  <td>{row.false_positive}</td>
                  <td>{row.false_negative}</td>
                  <td>{row.best_f1 === null ? "n/a" : formatPercent(row.best_f1)}</td>
                  <td>{row.best_threshold === null ? "n/a" : row.best_threshold.toFixed(2)}</td>
                  <td title={row.artifact_path}>{formatArtifactTail(row.artifact_path)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MetricCurve(props: { series: MetricSeries }) {
  const { series } = props;
  const values = series.points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = series.points.map((point, index) => {
    const x = series.points.length <= 1 ? 50 : (index / (series.points.length - 1)) * 100;
    const y = 88 - ((point.value - min) / range) * 72;
    return `${x},${y}`;
  });

  return (
    <div className="curve-card">
      <div className="curve-title">
        <strong title={series.name}>{shortMetricName(series.name)}</strong>
        <span>{series.latest === null ? "n/a" : series.latest.toFixed(3)}</span>
      </div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <line x1="0" y1="88" x2="100" y2="88" />
        {points.length > 1 ? <polyline points={points.join(" ")} /> : null}
        {points.map((point, index) => {
          const [x, y] = point.split(",");
          return <circle key={`${point}-${index}`} cx={x} cy={y} r="2.3" />;
        })}
      </svg>
    </div>
  );
}

function ClassOutcomeTable(props: { rows: ClassOutcome[] }) {
  return (
    <div className="analysis-panel">
      <strong>Class Outcomes</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">Run prediction analysis to populate class outcomes.</p>
      ) : (
        <div className="compact-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Match</th>
                <th>False +</th>
                <th>False -</th>
                <th>Confused</th>
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row) => (
                <tr key={row.class_id}>
                  <td>{row.class_name}</td>
                  <td>{row.matched}</td>
                  <td>{row.false_positive}</td>
                  <td>{row.false_negative}</td>
                  <td>{row.class_confusion}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ConfusionMatrix(props: { cells: ConfusionCell[] }) {
  return (
    <div className="analysis-panel">
      <strong>Confusion Matrix</strong>
      {props.cells.length === 0 ? (
        <p className="empty-state">Matched predictions will populate the matrix.</p>
      ) : (
        <div className="matrix-list">
          {props.cells.map((cell) => (
            <div
              className="matrix-cell"
              key={`${cell.actual_class_id}-${cell.predicted_class_id}`}
            >
              <span>
                {`${cell.actual_class_name} -> ${cell.predicted_class_name}`}
              </span>
              <strong>{cell.count}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ThresholdScanTable(props: {
  rows: RunExperimentSummary["threshold_scan"];
  recommendation: RunExperimentSummary["threshold_recommendation"];
}) {
  return (
    <div className="analysis-panel threshold-panel">
      <strong>Threshold Scan</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">Run prediction jobs at different confidence thresholds.</p>
      ) : (
        <>
          {props.recommendation ? (
            <div className="threshold-recommendation">
              <span>Best threshold</span>
              <strong>{props.recommendation.confidence_threshold.toFixed(2)}</strong>
              <small>
                F1 {formatPercent(props.recommendation.f1)} | P{" "}
                {formatPercent(props.recommendation.precision)} | R{" "}
                {formatPercent(props.recommendation.recall)}
              </small>
            </div>
          ) : null}
          <div className="compact-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Conf</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Matched</th>
                  <th>False +</th>
                  <th>False -</th>
                  <th>Confused</th>
                </tr>
              </thead>
              <tbody>
                {props.rows.map((row) => (
                  <tr key={row.job_id}>
                    <td>{row.confidence_threshold.toFixed(2)}</td>
                    <td>{formatPercent(row.precision)}</td>
                    <td>{formatPercent(row.recall)}</td>
                    <td>{formatPercent(row.f1)}</td>
                    <td>{row.matched}</td>
                    <td>{row.false_positive}</td>
                    <td>{row.false_negative}</td>
                    <td>{row.class_confusion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function shortMetricName(name: string) {
  return name.replace("metrics/", "").replace("train/", "").replace("(B)", "");
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatArtifactTail(path: string) {
  const parts = path.split("/").filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

function formatBytes(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = size / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function formatDimensionRefresh(summary: DatasetDimensionRefreshSummary) {
  return `已扫描 ${summary.scanned_count} 张，已更新 ${summary.updated_count} 张，仍缺失 ${summary.missing_count} 张`;
}

function formatQualityTagSummary(summary: QualityTagApplySummary) {
  return `已向 ${summary.updated_annotation_count} 个标注应用 ${summary.applied_tag_count} 个标签`;
}

function annotationGuidance(
  dataset: DatasetImportResponse | null,
  classCount: number,
  selectedImage: DatasetImage | null,
  selectedClass: ProjectClass | null,
) {
  if (!dataset) {
    return "请加载或导入数据集后开始标注。";
  }
  if (classCount === 0) {
    return "绘制边界框前，请先创建项目类别。";
  }
  if (!selectedImage) {
    return "请在图像浏览器中选择一张图像后开始标注。";
  }
  if (!selectedClass) {
    return "绘制边界框前，请先选择类别。";
  }
  return "在图像上拖动以添加边界框。";
}

function DatasetCoveragePanel(props: { coverage: DatasetCoverageSummary | null }) {
  const { coverage } = props;
  return (
    <div className="coverage-panel" aria-label="数据集覆盖情况">
      <div className="coverage-heading">
        <strong>数据集覆盖情况</strong>
        {coverage ? (
          <span>
            {coverage.annotated_image_count}/{coverage.image_count} 张图像 |{" "}
            {coverage.annotation_count} 个边界框
          </span>
        ) : null}
      </div>
      {!coverage ? (
        <p className="empty-state">导入数据集后可查看场景与标签覆盖情况。</p>
      ) : (
        <div className="coverage-grid">
          <CoverageBucketList title="平台" rows={coverage.platforms} />
          <CoverageBucketList title="高度" rows={coverage.altitude_bands} />
          <ClassCoverageList rows={coverage.classes} />
          <EdgeTagCoverageList rows={coverage.edge_tags} />
        </div>
      )}
    </div>
  );
}

function CoverageBucketList(props: { title: string; rows: DatasetCoverageSummary["platforms"] }) {
  return (
    <div className="coverage-card">
      <strong>{props.title}</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">暂无样本。</p>
      ) : (
        props.rows.map((row) => (
          <div className="coverage-row" key={row.label}>
            <span>{row.label}</span>
            <strong>{row.annotation_count}</strong>
            <small>
              {row.annotated_image_count}/{row.image_count} 张图像
            </small>
          </div>
        ))
      )}
    </div>
  );
}

function ClassCoverageList(props: { rows: DatasetCoverageSummary["classes"] }) {
  return (
    <div className="coverage-card">
      <strong>类别</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">创建类别后可跟踪标签覆盖情况。</p>
      ) : (
        props.rows.map((row) => (
          <div className="coverage-row class-coverage-row" key={row.class_id}>
            <span>
              <i style={{ background: row.class_color }} />
              {row.class_name}
            </span>
            <strong>{row.annotation_count}</strong>
            <small>{row.image_count} 张图像</small>
          </div>
        ))
      )}
    </div>
  );
}

function EdgeTagCoverageList(props: { rows: DatasetCoverageSummary["edge_tags"] }) {
  return (
    <div className="coverage-card">
      <strong>边缘案例标签</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">可添加“遮挡”或“伪装”等标签。</p>
      ) : (
        props.rows.slice(0, 6).map((row) => (
          <div className="coverage-row" key={row.tag}>
            <span>{formatEdgeTag(row.tag)}</span>
            <strong>{row.annotation_count}</strong>
            <small>{row.image_count} 张图像</small>
          </div>
        ))
      )}
    </div>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}
