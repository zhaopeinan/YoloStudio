import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Database,
  Download,
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
  LogOut,
  Users,
} from "lucide-react";
import type { CSSProperties, ChangeEvent, FormEvent, PointerEvent, ReactNode } from "react";
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
  PredictionReviewAnnotation,
  QualityTagApplySummary,
  RunExperimentSummary,
  TrainingAugmentationConfig,
  TrainingModelOption,
  TrainingRunArtifactSummary,
  TrainingRun,
  ModelWeight,
  DatasetSourceOption,
  DetectedClassSuggestion,
} from "./api";
import {
  applyQualityTags,
  cancelTrainingRun,
  createClass,
  createDatasetVersion,
  deleteClass,
  deleteDatasetSource,
  deleteModelWeight,
  createPredictionJob,
  createRunExport,
  createTrainingRun,
  getAnnotations,
  getDatasetCoverage,
  getExportCapabilities,
  getExportDownloadUrl,
  getHealth,
  getPredictionJobLogs,
  getPredictionImageReview,
  getProjectTrainingSummary,
  getQuality,
  getRunBestWeightsUrl,
  getTrainingRunArtifacts,
  getTrainingRunLogs,
  getTrainingRunSummary,
  importDataset,
  listClasses,
  listDatasetSources,
  listDatasetVersions,
  listProjectVersions,
  listDetectedClasses,
  listImages,
  listModelWeights,
  listPredictionJobs,
  listPredictions,
  listProjects,
  listQualityIssues,
  listRunExports,
  listTrainingModels,
  listTrainingRuns,
  replaceAnnotations,
  refreshImageDimensions,
  scanDataset,
  setUnauthorizedHandler,
  updateClass,
  uploadDatasetSource,
  uploadModelWeight,
  authedMediaUrl,
} from "./api";
import {
  formatEdgeTag,
  formatFailureType,
  formatQualityIssueType,
  formatRunStatus,
} from "./localization";
import { WorkflowShell } from "./WorkflowShell";
import { AnnotationCanvasToolbar } from "./AnnotationCanvasToolbar";
import { AnnotationFilterDrawer } from "./AnnotationFilterDrawer";
import { AnnotationToolbar } from "./AnnotationToolbar";
import { suggestClassFromFilename, type FilenameSuggestion } from "./annotation-suggestions";
import { StorageManagementView } from "./StorageManagementView";
import { PreviewView } from "./PreviewView";
import { TrainingLiveMonitor } from "./TrainingLiveMonitor";
import { LoginView } from "./LoginView";
import { UserManagementView } from "./UserManagementView";
import { roleLabel, useAuth } from "./auth";
import {
  fitViewport,
  constrainPan,
  manualMaxZoom,
  manualMinZoom,
  nextZoomLevel,
  zoomAroundPoint,
  type CanvasViewport,
} from "./annotation-viewport";
import {
  countAdvancedImageFilters,
  findNextAnnotationImageId,
  selectInitialAnnotationImageId,
  type AdvancedImageFilters,
} from "./annotation-workbench";
import { buildWorkflowSteps, parseWorkflowHash, workflowStepOrder, type WorkflowStep } from "./workflow";

function resolveDefaultDatasetPath(): string {
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      return "/opt/YoloStudio/image_dataset.zip";
    }
  }
  return "/tmp/image_dataset.zip";
}

const defaultDatasetPath = resolveDefaultDatasetPath();
const defaultProjectName = "YOLO 目标检测项目";
const defaultDatasetName = "image_dataset";
const defaultClassColor = "#ef4444";
const compactAnnotationViewportQuery = "(max-width: 820px)";

function usesCompactAnnotationViewport(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia(compactAnnotationViewportQuery).matches;
}
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
  { value: "all", label: "全部" },
  { value: "matched", label: "匹配正确" },
  { value: "false_positive", label: "误报" },
  { value: "false_negative", label: "漏报" },
  { value: "class_confusion", label: "类别混淆" },
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
  };
}

function defaultImageFilters(): ImageFiltersState {
  return {
    platform: "",
    label_status: "all",
    class_id: "",
    edge_tag: "",
    failure_type: "all",
    altitude_min: "",
    altitude_max: "",
  };
}

type DraftBox = Annotation & {
  local_id: string;
};

type ImageFiltersState = Omit<AdvancedImageFilters, "failure_type"> & {
  failure_type: PredictionFailureType;
  label_status: "all" | "annotated" | "unannotated";
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

type CanvasPanState = {
  pointerId: number;
  startClientX: number;
  startClientY: number;
  originalPanX: number;
  originalPanY: number;
};

export default function App() {
  const { user, loading: authLoading, isAdmin, logout } = useAuth();
  const [showUserManagement, setShowUserManagement] = useState(false);
  const [currentStep, setCurrentStep] = useState<WorkflowStep>("dataset");
  const [datasetPageTab, setDatasetPageTab] = useState<"import" | "management">("import");
  const currentStepRef = useRef<WorkflowStep>("dataset");
  const workflowStepsRef = useRef<ReturnType<typeof buildWorkflowSteps>>([]);

  useEffect(() => {
    setUnauthorizedHandler(() => {
      logout();
    });
    return () => setUnauthorizedHandler(null);
  }, [logout]);

  useEffect(() => {
    if (!isAdmin && showUserManagement) {
      setShowUserManagement(false);
    }
    if (!isAdmin && (datasetPageTab === "management" || currentStep === "training" || currentStep === "preview" || currentStep === "evaluation")) {
      setDatasetPageTab("import");
      if (currentStep === "training" || currentStep === "preview" || currentStep === "evaluation") {
        setCurrentStep("dataset");
        window.history.replaceState(null, "", "#dataset");
      }
    }
  }, [isAdmin, showUserManagement, datasetPageTab, currentStep]);
  const skipNextStorageWorkspaceRefreshRef = useRef(false);
  const [navigationNotice, setNavigationNotice] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [datasetSourceOptions, setDatasetSourceOptions] = useState<DatasetSourceOption[]>([]);
  const [selectedDatasetSourceRef, setSelectedDatasetSourceRef] = useState("");
  const [isUploadingDatasetSource, setIsUploadingDatasetSource] = useState(false);
  const [deletingDatasetSourceId, setDeletingDatasetSourceId] = useState<number | null>(null);
  const [projectName, setProjectName] = useState(defaultProjectName);
  const [datasetName, setDatasetName] = useState(defaultDatasetName);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedSavedDataset, setSelectedSavedDataset] = useState("");
  const [savedDatasetError, setSavedDatasetError] = useState<string | null>(null);
  const [isLoadingSavedDataset, setIsLoadingSavedDataset] = useState(false);
  const [openingStorageDatasetId, setOpeningStorageDatasetId] = useState<number | null>(null);
  const [importedDataset, setImportedDataset] = useState<DatasetImportResponse | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [classes, setClasses] = useState<ProjectClass[]>([]);
  const [detectedClasses, setDetectedClasses] = useState<DetectedClassSuggestion[]>([]);
  const [selectedDetectedClassNames, setSelectedDetectedClassNames] = useState<string[]>([]);
  const [isLoadingDetectedClasses, setIsLoadingDetectedClasses] = useState(false);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [versionClassIds, setVersionClassIds] = useState<number[]>([]);
  const [classError, setClassError] = useState<string | null>(null);
  const [isEnablingDetectedClasses, setIsEnablingDetectedClasses] = useState(false);
  const [classDeleteTarget, setClassDeleteTarget] = useState<ProjectClass | null>(null);
  const [isDeletingClass, setIsDeletingClass] = useState(false);
  const [editingClassId, setEditingClassId] = useState<number | null>(null);
  const [classEditName, setClassEditName] = useState("");
  const [classEditColor, setClassEditColor] = useState(defaultClassColor);
  const [isUpdatingClass, setIsUpdatingClass] = useState(false);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [imagePage, setImagePage] = useState({ limit: imagePageSize, offset: 0, total: 0 });
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [pendingAnnotationImageId, setPendingAnnotationImageId] = useState<number | null>(null);
  const [annotationNotice, setAnnotationNotice] = useState<string | null>(null);
  const [mobileAnnotationPane, setMobileAnnotationPane] = useState<"canvas" | "images" | "properties">("canvas");
  const [imageFilters, setImageFilters] = useState<ImageFiltersState>(defaultImageFilters);
  const [imageFilenameSearch, setImageFilenameSearch] = useState("");
  const [isImageFilterDrawerOpen, setIsImageFilterDrawerOpen] = useState(false);
  const [imageFilterError, setImageFilterError] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<DraftBox[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<DatasetImage["annotation_status"]>("unreviewed");
  const [annotationsDirty, setAnnotationsDirty] = useState(false);
  const annotationsDirtyRef = useRef(false);
  const [pendingWorkflowStep, setPendingWorkflowStep] = useState<WorkflowStep | null>(null);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [boxMoveState, setBoxMoveState] = useState<BoxMoveState | null>(null);
  const boxMoveStateRef = useRef<BoxMoveState | null>(null);
  const [boxResizeState, setBoxResizeState] = useState<BoxResizeState | null>(null);
  const boxResizeStateRef = useRef<BoxResizeState | null>(null);
  const annotationCanvasRef = useRef<SVGSVGElement | null>(null);
  const annotationViewportRef = useRef<HTMLDivElement | null>(null);
  const [annotationViewportSize, setAnnotationViewportSize] = useState({ width: 0, height: 0 });
  const [annotationImageSize, setAnnotationImageSize] = useState({ width: 0, height: 0 });
  const [canvasViewport, setCanvasViewport] = useState<CanvasViewport>({
    zoom: 1,
    panX: 0,
    panY: 0,
    mode: "fit",
  });
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const isSpacePressedRef = useRef(false);
  const [canvasPanState, setCanvasPanState] = useState<CanvasPanState | null>(null);
  const canvasPanStateRef = useRef<CanvasPanState | null>(null);
  const stopBoxInteractionTrackingRef = useRef<(() => void) | null>(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);
  const [showAnnotationGeometry, setShowAnnotationGeometry] = useState(false);
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
  const [projectVersions, setProjectVersions] = useState<DatasetVersion[]>([]);
  const [selectedTrainingVersionId, setSelectedTrainingVersionId] = useState<number | null>(null);
  const [selectedEvalRunId, setSelectedEvalRunId] = useState<number | null>(null);
  const [selectedEvalVersionId, setSelectedEvalVersionId] = useState<number | null>(null);
  const [versionName, setVersionName] = useState("");
  const [versionImageScope, setVersionImageScope] = useState<"annotated" | "all">("annotated");
  const [versionDatasetIds, setVersionDatasetIds] = useState<number[]>([]);
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
  const [liveMonitorRunId, setLiveMonitorRunId] = useState<number | null>(null);
  const [trainingModel, setTrainingModel] = useState("base:yolov8n.pt");
  const [trainingModelOptions, setTrainingModelOptions] = useState<TrainingModelOption[]>([]);
  const [modelWeights, setModelWeights] = useState<ModelWeight[]>([]);
  const [isUploadingWeight, setIsUploadingWeight] = useState(false);
  const [deletingWeightId, setDeletingWeightId] = useState<number | null>(null);
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
  const [predictionScope, setPredictionScope] = useState("all");
  const [predictionConfidence, setPredictionConfidence] = useState(0.25);
  const [predictionFilters, setPredictionFilters] = useState(defaultPredictionFilters);
  const [showEvalAdvanced, setShowEvalAdvanced] = useState(false);
  const [activeReview, setActiveReview] = useState<PredictionImageReview | null>(null);
  const [evalReview, setEvalReview] = useState<PredictionImageReview | null>(null);
  const [evalReviewFocusId, setEvalReviewFocusId] = useState<number | null>(null);
  const [showGroundTruthLayer, setShowGroundTruthLayer] = useState(true);
  const [showPredictionLayer, setShowPredictionLayer] = useState(true);
  const [exportCapabilities, setExportCapabilities] = useState<ExportCapabilities | null>(null);
  const [exports, setExports] = useState<ExportArtifact[]>([]);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isCreatingExport, setIsCreatingExport] = useState<string | null>(null);

  useEffect(() => {
    if (!user) {
      return;
    }
    getHealth()
      .then(setHealth)
      .catch((error: Error) => setHealthError(error.message));
    void refreshProjects();
    if (isAdmin) {
      void refreshDatasetSources();
    }
  }, [user?.id, isAdmin]);

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

  const filenameSuggestion = useMemo<FilenameSuggestion>(
    () => suggestClassFromFilename(selectedImage?.relative_path ?? "", classes),
    [classes, selectedImage?.relative_path],
  );

  const annotationReady = Boolean(importedDataset && classes.length > 0 && selectedImage && selectedClass);
  const projectDatasets = useMemo(() => {
    if (!importedDataset) {
      return [];
    }
    const project = projects.find((item) => item.id === importedDataset.project_id);
    return project?.datasets ?? [];
  }, [importedDataset, projects]);
  const advancedImageFilters = useMemo<AdvancedImageFilters>(
    () => ({
      platform: imageFilters.platform,
      class_id: imageFilters.class_id,
      edge_tag: imageFilters.edge_tag,
      failure_type: imageFilters.failure_type,
      altitude_min: imageFilters.altitude_min,
      altitude_max: imageFilters.altitude_max,
    }),
    [imageFilters],
  );
  const advancedFilterCount = countAdvancedImageFilters(advancedImageFilters);
  const visibleImages = useMemo(() => {
    const query = imageFilenameSearch.trim().toLowerCase();
    return query
      ? images.filter((image) => image.relative_path.toLowerCase().includes(query))
      : images;
  }, [imageFilenameSearch, images]);

  const classById = useMemo(() => {
    return new Map(classes.map((classItem) => [classItem.id, classItem]));
  }, [classes]);

  const evalReviewIndex = useMemo(() => {
    if (evalReviewFocusId == null) {
      return -1;
    }
    return predictions.findIndex((prediction) => prediction.id === evalReviewFocusId);
  }, [evalReviewFocusId, predictions]);

  const evalFocusedPrediction = useMemo(() => {
    if (evalReviewIndex < 0) {
      return null;
    }
    return predictions[evalReviewIndex] ?? null;
  }, [evalReviewIndex, predictions]);

  const selectedAnnotation = useMemo(
    () => annotations.find((annotation) => annotation.local_id === selectedAnnotationId) ?? null,
    [annotations, selectedAnnotationId],
  );
  const selectedAnnotationIndex = selectedAnnotation
    ? annotations.findIndex((annotation) => annotation.local_id === selectedAnnotation.local_id)
    : -1;

  useEffect(() => {
    setShowAnnotationGeometry(false);
  }, [selectedAnnotationId, selectedImageId]);

  useEffect(() => {
    if (filenameSuggestion.selected) {
      setSelectedClassId(filenameSuggestion.selected.id);
    }
  }, [selectedImageId, filenameSuggestion.selected?.id]);

  useEffect(() => {
    const width = Number(selectedImage?.width);
    const height = Number(selectedImage?.height);
    setAnnotationStatus(
      selectedImage?.annotation_status
        ?? (selectedImage?.annotation_count ? "annotated" : "unreviewed"),
    );
    setAnnotationImageSize({
      width: Number.isFinite(width) && width > 0 ? width : 0,
      height: Number.isFinite(height) && height > 0 ? height : 0,
    });
    setCanvasViewport({ zoom: 1, panX: 0, panY: 0, mode: "fit" });
    handlePointerCancel();
  }, [selectedImageId]);

  useEffect(() => {
    const element = annotationViewportRef.current;
    if (!element) {
      return;
    }

    const updateSize = () => {
      setAnnotationViewportSize({
        width: element.clientWidth,
        height: element.clientHeight,
      });
    };
    updateSize();

    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, [currentStep, selectedImageId, mobileAnnotationPane]);

  useEffect(() => {
    const element = annotationViewportRef.current;
    if (!element) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      if (usesCompactAnnotationViewport()) {
        return;
      }
      event.preventDefault();
      if (
        event.deltaY === 0 ||
        annotationImageSize.width <= 0 ||
        annotationImageSize.height <= 0
      ) {
        return;
      }
      const rect = element.getBoundingClientRect();
      const viewportSize = {
        width: rect.width || annotationViewportSize.width,
        height: rect.height || annotationViewportSize.height,
      };
      if (viewportSize.width <= 0 || viewportSize.height <= 0) {
        return;
      }
      setCanvasViewport((current) =>
        zoomAroundPoint(
          current,
          current.zoom * Math.exp(-event.deltaY * 0.0015),
          { x: event.clientX - rect.left, y: event.clientY - rect.top },
          annotationImageSize,
          viewportSize,
        ),
      );
    };

    element.addEventListener("wheel", handleWheel, { passive: false });
    return () => element.removeEventListener("wheel", handleWheel);
  }, [annotationImageSize, annotationViewportSize, currentStep, mobileAnnotationPane, selectedImageId]);

  useEffect(() => {
    if (canvasViewport.mode !== "fit") {
      return;
    }
    setCanvasViewport(fitViewport(annotationImageSize, annotationViewportSize));
  }, [annotationImageSize, annotationViewportSize, canvasViewport.mode]);

  useEffect(() => {
    function clearPanMode() {
      handlePointerCancel();
      const activePan = canvasPanStateRef.current;
      if (activePan && annotationViewportRef.current) {
        releasePointerCaptureSafe(annotationViewportRef.current, activePan.pointerId);
      }
      isSpacePressedRef.current = false;
      setIsSpacePressed(false);
      canvasPanStateRef.current = null;
      setCanvasPanState(null);
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (
        usesCompactAnnotationViewport()
        || (event.key !== " " && event.code !== "Space")
        || isEditableEventTarget(event.target)
      ) {
        return;
      }
      event.preventDefault();
      if (!isSpacePressedRef.current) {
        handlePointerCancel();
        isSpacePressedRef.current = true;
        setIsSpacePressed(true);
      }
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.key === " " || event.code === "Space") {
        clearPanMode();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", clearPanMode);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", clearPanMode);
    };
  }, []);

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
  const selectedEvalRun = useMemo(
    () => runs.find((run) => run.id === selectedEvalRunId) ?? null,
    [runs, selectedEvalRunId],
  );
  const completedEvalRuns = useMemo(
    () => runs.filter((run) => run.status === "completed"),
    [runs],
  );
  const imagePageStart = imagePage.total === 0 ? 0 : imagePage.offset + 1;
  const imagePageEnd = Math.min(imagePage.offset + images.length, imagePage.total);
  const versionImageCount = versionImageScope === "annotated"
    ? quality?.annotated_image_count ?? 0
    : quality?.image_count ?? 0;
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
  const workflowSteps = useMemo(
    () =>
      buildWorkflowSteps(
        {
          hasDataset: Boolean(importedDataset),
          classCount: classes.length,
          annotatedImageCount: quality?.annotated_image_count ?? 0,
          versionCount: versions.length,
          runCount: runs.length,
          predictionJobCount: predictionJobs.length,
          exportCount: exports.length,
        },
        { isAdmin },
      ),
    [
      classes.length,
      exports.length,
      importedDataset,
      isAdmin,
      predictionJobs.length,
      quality,
      runs.length,
      versions.length,
    ],
  );
  const previewDatasets = useMemo(() => {
    if (!importedDataset) return [];
    const project = projects.find((candidate) => candidate.id === importedDataset.project_id);
    if (project) return project.datasets;
    return [{
      id: importedDataset.dataset_id,
      project_id: importedDataset.project_id,
      name: importedDataset.dataset_name,
      source_type: "imported",
      import_status: "imported",
      image_count: importedDataset.image_count,
      annotated_image_count: 0,
      annotation_count: 0,
    }];
  }, [importedDataset, projects]);

  currentStepRef.current = currentStep;
  workflowStepsRef.current = workflowSteps;
  annotationsDirtyRef.current = annotationsDirty;

  function markAnnotationsDirty() {
    annotationsDirtyRef.current = true;
    setAnnotationsDirty(true);
  }

  function clearAnnotationsDirty() {
    annotationsDirtyRef.current = false;
    setAnnotationsDirty(false);
  }

  function clearLoadedDatasetWorkspace() {
    skipNextStorageWorkspaceRefreshRef.current = true;
    stopBoxInteractionTrackingRef.current?.();
    stopBoxInteractionTrackingRef.current = null;
    boxMoveStateRef.current = null;
    boxResizeStateRef.current = null;
    canvasPanStateRef.current = null;
    isSpacePressedRef.current = false;
    setImportedDataset(null);
    setClasses([]);
    setDetectedClasses([]);
    setSelectedDetectedClassNames([]);
    setIsLoadingDetectedClasses(false);
    setSelectedClassId(null);
    setVersionClassIds([]);
    setClassError(null);
    setIsEnablingDetectedClasses(false);
    setClassDeleteTarget(null);
    setIsDeletingClass(false);
    setEditingClassId(null);
    setClassEditName("");
    setClassEditColor(defaultClassColor);
    setIsUpdatingClass(false);
    setImages([]);
    setImagePage({ limit: imagePageSize, offset: 0, total: 0 });
    setSelectedImageId(null);
    setPendingAnnotationImageId(null);
    setAnnotationNotice(null);
    setMobileAnnotationPane("canvas");
    setImageFilters(defaultImageFilters());
    setImageFilenameSearch("");
    setIsImageFilterDrawerOpen(false);
    setImageFilterError(null);
    setAnnotations([]);
    setAnnotationStatus("unreviewed");
    setSelectedAnnotationId(null);
    clearAnnotationsDirty();
    setPendingWorkflowStep(null);
    setAnnotationError(null);
    setIsSavingAnnotations(false);
    setDragState(null);
    setActiveBoxMoveState(null);
    setActiveBoxResizeState(null);
    setAnnotationViewportSize({ width: 0, height: 0 });
    setAnnotationImageSize({ width: 0, height: 0 });
    setCanvasViewport({ zoom: 1, panX: 0, panY: 0, mode: "fit" });
    setIsSpacePressed(false);
    setCanvasPanState(null);
    setShowAnnotationGeometry(false);
    setActiveReview(null);
    setQuality(null);
    setCoverage(null);
    setQualityIssues([]);
    setQualityIssueType("all");
    setQualityError(null);
    setIsLoadingQuality(false);
    setIsApplyingQualityTags(false);
    setQualityTagSummary(null);
    setDimensionRefresh(null);
    setIsRefreshingDimensions(false);
    setVersions([]);
    setSelectedTrainingVersionId(null);
    setVersionName("");
    setVersionImageScope("annotated");
    setVersionDatasetIds([]);
    setVersionError(null);
    setIsCreatingVersion(false);
    setRuns([]);
    setRunLogs({});
    setRunArtifacts({});
    setRunSummary(null);
    setProjectExperimentSummary(null);
    setSummaryError(null);
    setTrainingError(null);
    setTrainingNotice(null);
    setIsStartingRun(false);
    setCancellingRunId(null);
    setTrainingModel("yolov8n.pt");
    setTrainingEpochs(50);
    setTrainingImageSize(640);
    setTrainingBatchSize(8);
    setTrainingDevice("");
    setAugmentationPreset("balanced");
    setAugmentation(defaultAugmentation);
    setTrainingTta(false);
    setThresholdScan(false);
    setPredictionJobs([]);
    setPredictions([]);
    setEvalReview(null);
    setEvalReviewFocusId(null);
    setPredictionLogs({});
    setPredictionError(null);
    setIsCreatingPrediction(false);
    setPredictionScope("all");
    setPredictionConfidence(0.25);
    setPredictionFilters(defaultPredictionFilters());
    setShowEvalAdvanced(false);
    setSelectedEvalRunId(null);
    setSelectedEvalVersionId(null);
    setProjectVersions([]);
    setShowGroundTruthLayer(true);
    setShowPredictionLayer(true);
    setExportCapabilities(null);
    setExports([]);
    setExportError(null);
    setIsCreatingExport(null);
    setNavigationNotice(null);
  }

  function commitWorkflowStep(stepId: WorkflowStep) {
    setNavigationNotice(null);
    setPendingWorkflowStep(null);
    setCurrentStep(stepId);
    currentStepRef.current = stepId;
    window.history.replaceState(null, "", `#${stepId}`);
  }

  function requestWorkflowStep(stepId: WorkflowStep) {
    const step = workflowStepsRef.current.find((item) => item.id === stepId);
    if (!step || step.availability === "locked") {
      setNavigationNotice(step?.lockedReason ?? "当前步骤暂不可用");
      window.history.replaceState(null, "", `#${currentStepRef.current}`);
      return false;
    }
    if (
      currentStepRef.current === "annotation" &&
      stepId !== "annotation" &&
      annotationsDirtyRef.current
    ) {
      setNavigationNotice(null);
      setPendingWorkflowStep(stepId);
      window.history.replaceState(null, "", "#annotation");
      return false;
    }
    commitWorkflowStep(stepId);
    return true;
  }

  function handleWorkflowNavigate(stepId: WorkflowStep) {
    requestWorkflowStep(stepId);
  }

  function handleStayOnAnnotation() {
    setPendingWorkflowStep(null);
    window.history.replaceState(null, "", "#annotation");
  }

  function commitAnnotationImage(imageId: number) {
    setPendingAnnotationImageId(null);
    setAnnotationNotice(null);
    setActiveReview(null);
    setSelectedImageId(imageId);
    const nextImage = images.find((image) => image.id === imageId);
    setAnnotationStatus(
      nextImage?.annotation_status ?? (nextImage?.annotation_count ? "annotated" : "unreviewed"),
    );
  }

  function requestAnnotationImage(imageId: number) {
    if (imageId === selectedImageId) {
      return true;
    }
    if (annotationsDirtyRef.current) {
      setPendingAnnotationImageId(imageId);
      return false;
    }
    commitAnnotationImage(imageId);
    return true;
  }

  function handleStayOnCurrentImage() {
    setPendingAnnotationImageId(null);
  }

  function handleDiscardAnnotationsAndSwitch() {
    if (pendingAnnotationImageId === null) {
      return;
    }
    const targetImageId = pendingAnnotationImageId;
    setAnnotations([]);
    setSelectedAnnotationId(null);
    clearAnnotationsDirty();
    commitAnnotationImage(targetImageId);
  }

  async function handleDiscardAnnotationsAndLeave() {
    if (!pendingWorkflowStep) {
      return;
    }

    const targetStep = pendingWorkflowStep;
    if (selectedImageId) {
      try {
        const response = await getAnnotations(selectedImageId);
        const nextAnnotations = response.items.map(toDraftBox);
        setAnnotations(nextAnnotations);
        setAnnotationStatus(response.annotation_status ?? (nextAnnotations.length ? "annotated" : "unreviewed"));
        setSelectedAnnotationId(nextAnnotations[0]?.local_id ?? null);
        clearAnnotationsDirty();
      } catch (error) {
        setAnnotationError(error instanceof Error ? error.message : "标注加载失败");
        return;
      }
    } else {
      clearAnnotationsDirty();
    }
    commitWorkflowStep(targetStep);
  }

  useEffect(() => {
    function restoreWorkflowStepFromHash(initial = false) {
      const rawStep = window.location.hash.replace(/^#/, "");
      const requestedStep = parseWorkflowHash(window.location.hash);
      const isValidHash = workflowStepOrder.includes(rawStep as WorkflowStep);
      const step = workflowStepsRef.current.find((item) => item.id === requestedStep);

      if (!isValidHash) {
        requestWorkflowStep("dataset");
        return;
      }

      if (!step || step.availability === "locked") {
        setNavigationNotice(step?.lockedReason ?? "当前步骤暂不可用");
        const fallbackStep = initial ? "dataset" : currentStepRef.current;
        setCurrentStep(fallbackStep);
        currentStepRef.current = fallbackStep;
        window.history.replaceState(null, "", `#${fallbackStep}`);
        return;
      }

      requestWorkflowStep(requestedStep);
    }

    restoreWorkflowStepFromHash(true);
    const handleHashChange = () => restoreWorkflowStepFromHash(false);
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    if (!selectedImageId) {
      setAnnotations([]);
      clearAnnotationsDirty();
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
    if (!selectedEvalRunId || !hasActivePredictionJob) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshPredictionJobs(selectedEvalRunId);
      activePredictionJobIds.forEach((jobId) => {
        void loadPredictionLogs(jobId);
      });
    }, monitorRefreshMs);

    return () => window.clearInterval(interval);
  }, [activePredictionJobIds, hasActivePredictionJob, selectedEvalRunId]);

  useEffect(() => {
    if (!selectedEvalRunId) {
      return;
    }

    void refreshExports(selectedEvalRunId);
    void refreshRunSummary(selectedEvalRunId);
    void refreshPredictionJobs(selectedEvalRunId);
  }, [selectedEvalRunId]);

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

  function applyDatasetSourceOption(option: DatasetSourceOption | undefined) {
    if (!option) {
      setSelectedDatasetSourceRef("");
      return;
    }
    setSelectedDatasetSourceRef(option.source_ref);
    setDatasetPath(option.source_path);
    const stem = option.original_filename.replace(/\.zip$/i, "");
    if (stem) {
      setDatasetName(stem);
    }
  }

  async function refreshDatasetSources(preferredRef?: string) {
    try {
      const response = await listDatasetSources();
      setDatasetSourceOptions(response.items);
      const preferred =
        (preferredRef
          ? response.items.find((item) => item.source_ref === preferredRef)
          : undefined) ??
        response.items.find((item) => item.source_path === datasetPath) ??
        response.items.find((item) => item.source_ref === selectedDatasetSourceRef) ??
        response.items[0];
      applyDatasetSourceOption(preferred);
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "数据集来源列表加载失败");
    }
  }

  async function handleUploadDatasetSource(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setIsUploadingDatasetSource(true);
    setScanError(null);
    setImportError(null);
    try {
      const form = new FormData();
      form.append("archive", file);
      const saved = await uploadDatasetSource(form);
      await refreshDatasetSources(`upload:${saved.id}`);
      setScan(null);
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "数据集上传失败");
    } finally {
      setIsUploadingDatasetSource(false);
    }
  }

  async function handleDeleteDatasetSource(option: DatasetSourceOption) {
    if (option.source_id === null) {
      return;
    }
    if (!window.confirm(`确定删除已上传压缩包「${option.original_filename}」？`)) {
      return;
    }
    setDeletingDatasetSourceId(option.source_id);
    setScanError(null);
    try {
      await deleteDatasetSource(option.source_id);
      await refreshDatasetSources();
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "数据集删除失败");
    } finally {
      setDeletingDatasetSourceId(null);
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
      commitWorkflowStep("classes");
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

  async function handleStorageChanged() {
    const loadedDataset = importedDataset;
    await refreshProjects();
    if (skipNextStorageWorkspaceRefreshRef.current) {
      skipNextStorageWorkspaceRefreshRef.current = false;
      return;
    }
    if (loadedDataset) {
      await loadDatasetWorkspace(loadedDataset);
    }
  }

  async function loadDatasetWorkspace(dataset: DatasetImportResponse): Promise<{ classCount: number }> {
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
    setVersionDatasetIds([dataset.dataset_id]);
    await refreshDetectedClasses(dataset.dataset_id);
    setImages(imageResponse.items);
    setImagePage({
      limit: imageResponse.limit,
      offset: imageResponse.offset,
      total: imageResponse.total,
    });
    setSelectedImageId(selectInitialAnnotationImageId(imageResponse.items, null));
    setPendingAnnotationImageId(null);
    setAnnotationNotice(null);
    setAnnotations([]);
    clearAnnotationsDirty();
    setActiveReview(null);
    setQuality(qualityResponse);
    setCoverage(coverageResponse);
    setQualityIssues(qualityIssueResponse.items);
    setDimensionRefresh(null);
    setVersions(versionResponse.items);
    setSelectedTrainingVersionId((current) => {
      if (current && versionResponse.items.some((item) => item.id === current)) {
        return current;
      }
      return versionResponse.items[0]?.id ?? null;
    });
    setVersionImageScope("annotated");
    setRuns(runResponse.items);
    setRunLogs({});
    await refreshRunArtifacts(runResponse.items.map((run) => run.id));
    await refreshTrainingModels(dataset.project_id);
    setPredictionJobs([]);
    setPredictions([]);
    setEvalReview(null);
    setEvalReviewFocusId(null);
    setPredictionLogs({});
    setExportCapabilities(null);
    setExports([]);

    let projectVersionItems: DatasetVersion[] = versionResponse.items;
    try {
      const projectVersionResponse = await listProjectVersions(dataset.project_id);
      projectVersionItems = projectVersionResponse.items;
      setProjectVersions(projectVersionItems);
    } catch {
      setProjectVersions(versionResponse.items);
    }

    const defaultEvalRun =
      runResponse.items.find((run) => run.status === "completed") ?? runResponse.items[0] ?? null;
    setSelectedEvalRunId(defaultEvalRun?.id ?? null);
    setSelectedEvalVersionId(defaultEvalRun?.version_id ?? projectVersionItems[0]?.id ?? null);

    if (defaultEvalRun) {
      await refreshRunSummary(defaultEvalRun.id);
      await refreshProjectTrainingSummary(dataset.project_id);
      await refreshPredictionJobs(defaultEvalRun.id);
      await refreshExports(defaultEvalRun.id);
    } else {
      setRunSummary(null);
      setProjectExperimentSummary(null);
    }
    return { classCount: classResponse.items.length };
  }

  async function handleOpenDatasetFromStorage(dataset: {
    project_id: number;
    project_name: string;
    dataset_id: number;
    dataset_name: string;
    image_count: number;
  }) {
    setOpeningStorageDatasetId(dataset.dataset_id);
    setSavedDatasetError(null);
    try {
      const { classCount } = await loadDatasetWorkspace({
        project_id: dataset.project_id,
        dataset_id: dataset.dataset_id,
        project_name: dataset.project_name,
        dataset_name: dataset.dataset_name,
        image_count: dataset.image_count,
        groups: [],
      });
      setSelectedSavedDataset(`${dataset.project_id}:${dataset.dataset_id}`);
      commitWorkflowStep(classCount > 0 ? "annotation" : "classes");
    } catch (error) {
      setSavedDatasetError(error instanceof Error ? error.message : "数据集加载失败");
      setDatasetPageTab("import");
    } finally {
      setOpeningStorageDatasetId(null);
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
      commitWorkflowStep("classes");
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
      setSelectedTrainingVersionId((current) => {
        if (current && versionResponse.items.some((item) => item.id === current)) {
          return current;
        }
        return versionResponse.items[0]?.id ?? null;
      });
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
      const nextAnnotations = response.items.map(toDraftBox);
      setAnnotations(nextAnnotations);
      setAnnotationStatus(response.annotation_status ?? (nextAnnotations.length ? "annotated" : "unreviewed"));
      setSelectedAnnotationId(nextAnnotations[0]?.local_id ?? null);
      clearAnnotationsDirty();
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "标注加载失败");
    }
  }

  async function refreshImages(
    datasetId = importedDataset?.dataset_id,
    offset = imagePage.offset,
    filters: ImageFiltersState = imageFilters,
  ) {
    if (!datasetId) {
      return;
    }

    setImageFilterError(null);
    try {
      const response = await listImages(datasetId, toImageFilterRequest(filters), {
        limit: imagePageSize,
        offset,
      });
      setImages(response.items);
      setImagePage({
        limit: response.limit,
        offset: response.offset,
        total: response.total,
      });
      setSelectedImageId((current) => selectInitialAnnotationImageId(response.items, current));
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
    setIsImageFilterDrawerOpen(false);
  }

  function handleImageLabelStatusChange(labelStatus: ImageFiltersState["label_status"]) {
    const nextFilters = { ...imageFilters, label_status: labelStatus };
    setImageFilters(nextFilters);
    void refreshImages(importedDataset?.dataset_id, 0, nextFilters);
  }

  function handleAdvancedImageFilterChange(key: keyof AdvancedImageFilters, value: string) {
    setImageFilters((current) => ({ ...current, [key]: value }) as ImageFiltersState);
  }

  function handlePreviousImagePage() {
    const nextOffset = Math.max(0, imagePage.offset - imagePage.limit);
    void refreshImages(importedDataset?.dataset_id, nextOffset);
  }

  function handleNextImagePage() {
    const nextOffset = imagePage.offset + imagePage.limit;
    void refreshImages(importedDataset?.dataset_id, nextOffset);
  }

  function handleResetAdvancedImageFilters() {
    const nextFilters = {
      ...imageFilters,
      platform: "",
      class_id: "",
      edge_tag: "",
      failure_type: "all" as const,
      altitude_min: "",
      altitude_max: "",
    };
    setImageFilters(nextFilters);
    if (importedDataset) {
      setImageFilterError(null);
      listImages(importedDataset.dataset_id, toImageFilterRequest(nextFilters), { limit: imagePageSize, offset: 0 })
        .then((response) => {
          setImages(response.items);
          setImagePage({
            limit: response.limit,
            offset: response.offset,
            total: response.total,
          });
          setSelectedImageId((current) => selectInitialAnnotationImageId(response.items, current));
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
      model: trainingModel.trim() || "base:yolov8n.pt",
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
      model: stringConfig(config, "model", trainingModel.trim() || "base:yolov8n.pt"),
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
    setTrainingModel(stringConfig(config, "model", "base:yolov8n.pt"));
    setTrainingEpochs(numberConfig(config, "epochs", 50));
    setTrainingImageSize(numberConfig(config, "image_size", 640));
    setTrainingBatchSize(numberConfig(config, "batch_size", 8));
    setTrainingDevice(stringConfig(config, "device", ""));
    setAugmentationPreset(stringConfig(config, "augmentation_preset", "balanced"));
    setAugmentation(augmentationConfig(config, defaultAugmentation));
    setTrainingTta(booleanConfig(config, "tta", false));
    setThresholdScan(booleanConfig(config, "threshold_scan", false));
    setTrainingNotice(`已加载训练任务 #${run.id} 的配置`);
    setTrainingError(null);
  }

  async function refreshTrainingModels(projectId = importedDataset?.project_id) {
    if (!projectId) {
      return;
    }
    try {
      const [models, weights] = await Promise.all([
        listTrainingModels(projectId),
        listModelWeights(projectId),
      ]);
      setTrainingModelOptions(models.items);
      setModelWeights(weights.items);
      setTrainingModel((current) =>
        models.items.some((item) => item.model_ref === current) || current.trim() !== ""
          ? current
          : models.items[0]?.model_ref ?? "base:yolov8n.pt",
      );
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "模型列表加载失败");
    }
  }

  async function handleUploadModelWeight(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !importedDataset) {
      return;
    }
    setIsUploadingWeight(true);
    setTrainingError(null);
    setTrainingNotice(null);
    try {
      const form = new FormData();
      form.append("project_id", String(importedDataset.project_id));
      form.append("weight", file);
      const saved = await uploadModelWeight(form);
      await refreshTrainingModels(importedDataset.project_id);
      setTrainingModel(`weight:${saved.id}`);
      setTrainingNotice(`已上传权重：${saved.original_filename}`);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "权重上传失败");
    } finally {
      setIsUploadingWeight(false);
    }
  }

  async function handleDeleteModelWeight(weight: ModelWeight) {
    if (!importedDataset) {
      return;
    }
    if (!window.confirm(`确定删除上传权重「${weight.original_filename}」？`)) {
      return;
    }
    setDeletingWeightId(weight.id);
    setTrainingError(null);
    try {
      await deleteModelWeight(weight.id);
      if (trainingModel === `weight:${weight.id}`) {
        setTrainingModel("base:yolov8n.pt");
      }
      await refreshTrainingModels(importedDataset.project_id);
      setTrainingNotice(`已删除权重：${weight.original_filename}`);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "权重删除失败");
    } finally {
      setDeletingWeightId(null);
    }
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
      setTrainingError(error instanceof Error ? error.message : "训练任务刷新失败");
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
      setTrainingError(error instanceof Error ? error.message : "训练产物加载失败");
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
      setSummaryError(error instanceof Error ? error.message : "项目实验汇总加载失败");
    }
  }

  async function refreshPredictionJobs(runId = selectedEvalRunId ?? runs[0]?.id) {
    if (!runId) {
      setPredictionJobs([]);
      setPredictions([]);
      setEvalReview(null);
      setEvalReviewFocusId(null);
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
        setEvalReview(null);
        setEvalReviewFocusId(null);
        await refreshRunSummary(runId);
      }
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "预测任务刷新失败");
    }
  }

  async function refreshExports(runId = selectedEvalRunId ?? runs[0]?.id) {
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
      setExportError(error instanceof Error ? error.message : "导出记录刷新失败");
    }
  }

  async function refreshRunSummary(runId = selectedEvalRunId ?? runs[0]?.id) {
    if (!runId) {
      setRunSummary(null);
      return;
    }

    try {
      setRunSummary(await getTrainingRunSummary(runId));
      setSummaryError(null);
    } catch (error) {
      setSummaryError(error instanceof Error ? error.message : "训练任务汇总加载失败");
    }
  }

  async function loadFilteredPredictions(jobId: number) {
    const predictionsResponse = await listPredictions(
      jobId,
      toPredictionFilterRequest(predictionFilters),
    );
    setPredictions(predictionsResponse.items);
    setEvalReview(null);
    setEvalReviewFocusId(null);
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
      .then((response) => {
        setPredictions(response.items);
        setEvalReview(null);
        setEvalReviewFocusId(null);
      })
      .catch((error: Error) => setPredictionError(error.message));
  }

  async function refreshDetectedClasses(datasetId: number) {
    setIsLoadingDetectedClasses(true);
    setClassError(null);
    try {
      const response = await listDetectedClasses(datasetId);
      setDetectedClasses(response.items);
      // Default: all detected class names selected for confirmation.
      setSelectedDetectedClassNames(response.items.map((item) => item.name));
    } catch (error) {
      setDetectedClasses([]);
      setSelectedDetectedClassNames([]);
      setClassError(error instanceof Error ? error.message : "类别扫描失败");
    } finally {
      setIsLoadingDetectedClasses(false);
    }
  }

  async function handleEnableDetectedClasses() {
    if (
      !importedDataset
      || selectedDetectedClassNames.length === 0
      || isEnablingDetectedClasses
    ) {
      return;
    }

    setIsEnablingDetectedClasses(true);
    setClassError(null);
    const isFirstClass = classes.length === 0;
    const existingNames = new Set(classes.map((classItem) => classItem.name));
    const classesToCreate = detectedClasses.filter(
      (preset) => selectedDetectedClassNames.includes(preset.name) && !existingNames.has(preset.name),
    );
    const createdClasses: ProjectClass[] = [];

    try {
      for (const preset of classesToCreate) {
        const created = await createClass(importedDataset.project_id, {
          name: preset.name,
          color: preset.color,
        });
        createdClasses.push(created);
        setClasses((current) => [...current, created]);
        setVersionClassIds((current) => [...current, created.id]);
      }

      const firstAvailableClassId = classes[0]?.id ?? createdClasses[0]?.id ?? null;
      setSelectedClassId((current) => current ?? firstAvailableClassId);
      void refreshTrainingPrep();
      if (isFirstClass && createdClasses.length > 0) {
        commitWorkflowStep("annotation");
      }
    } catch (error) {
      setClassError(error instanceof Error ? error.message : "类别启用失败");
    } finally {
      setIsEnablingDetectedClasses(false);
    }
  }

  async function handleDeleteClass() {
    if (!importedDataset || !classDeleteTarget || isDeletingClass) {
      return;
    }

    setIsDeletingClass(true);
    setClassError(null);
    const deletedClassId = classDeleteTarget.id;
    const remainingClasses = classes.filter((classItem) => classItem.id !== deletedClassId);

    try {
      await deleteClass(importedDataset.project_id, deletedClassId);
      setClasses(remainingClasses);
      setVersionClassIds((current) => current.filter((classId) => classId !== deletedClassId));
      setSelectedClassId((current) => (current === deletedClassId ? remainingClasses[0]?.id ?? null : current));
      setSelectedDetectedClassNames((current) => current.filter((name) => name !== classDeleteTarget.name));
      setClassDeleteTarget(null);
      void refreshTrainingPrep();
    } catch (error) {
      setClassError(error instanceof Error ? error.message : "类别删除失败");
    } finally {
      setIsDeletingClass(false);
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

  function setCanvasZoom(nextZoom: number) {
    if (
      annotationImageSize.width <= 0 ||
      annotationImageSize.height <= 0 ||
      annotationViewportSize.width <= 0 ||
      annotationViewportSize.height <= 0
    ) {
      return;
    }
    setCanvasViewport((current) =>
      zoomAroundPoint(
        current,
        nextZoom,
        {
          x: annotationViewportSize.width / 2,
          y: annotationViewportSize.height / 2,
        },
        annotationImageSize,
        annotationViewportSize,
      ),
    );
  }

  function handleCanvasFit() {
    setCanvasViewport(fitViewport(annotationImageSize, annotationViewportSize));
  }

  function handleCanvasZoom(direction: "in" | "out") {
    setCanvasZoom(nextZoomLevel(canvasViewport.zoom, direction));
  }

  function handleViewportPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (usesCompactAnnotationViewport() || !isSpacePressedRef.current || event.button !== 0) {
      return;
    }
    event.preventDefault();
    handlePointerCancel();
    const nextPanState = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      originalPanX: canvasViewport.panX,
      originalPanY: canvasViewport.panY,
    };
    canvasPanStateRef.current = nextPanState;
    setCanvasPanState(nextPanState);
    setPointerCaptureSafe(event.currentTarget, event.pointerId);
  }

  function handleViewportPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (usesCompactAnnotationViewport()) {
      return;
    }
    const activePan = canvasPanStateRef.current;
    if (!activePan || activePan.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const deltaX = event.clientX - activePan.startClientX;
    const deltaY = event.clientY - activePan.startClientY;
    setCanvasViewport((current) => {
      const pan = constrainPan(annotationImageSize, annotationViewportSize, current.zoom, {
        x: activePan.originalPanX + deltaX,
        y: activePan.originalPanY + deltaY,
      });
      return {
        ...current,
        panX: pan.x,
        panY: pan.y,
        mode: "manual",
      };
    });
  }

  function endCanvasPan(element?: Element, pointerId?: number) {
    canvasPanStateRef.current = null;
    setCanvasPanState(null);
    if (element && pointerId !== undefined) {
      releasePointerCaptureSafe(element, pointerId);
    }
  }

  function handleViewportPointerUp(event: PointerEvent<HTMLDivElement>) {
    if (canvasPanStateRef.current?.pointerId === event.pointerId) {
      endCanvasPan(event.currentTarget, event.pointerId);
    }
  }

  function handleViewportPointerCancel(event: PointerEvent<HTMLDivElement>) {
    handlePointerCancel();
    endCanvasPan(event.currentTarget, event.pointerId);
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if (isSpacePressedRef.current || canvasPanStateRef.current || !selectedClass || !selectedImage) {
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
    if (isSpacePressedRef.current || canvasPanStateRef.current) {
      return;
    }
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
    if (isSpacePressedRef.current || canvasPanStateRef.current) {
      setDragState(null);
      return;
    }
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
    setAnnotationStatus("annotated");
    markAnnotationsDirty();
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
    if (isSpacePressedRef.current || canvasPanStateRef.current) {
      return;
    }
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
    if (isSpacePressedRef.current || canvasPanStateRef.current) {
      return;
    }
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
    markAnnotationsDirty();
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
    markAnnotationsDirty();
    setAnnotations((current) => current.filter((annotation) => annotation.local_id !== localId));
    if (annotations.length === 1) {
      setAnnotationStatus("unreviewed");
    }
    if (selectedAnnotationId === localId) {
      setSelectedAnnotationId(null);
    }
  }

  function nudgeAnnotation(localId: string, deltaX: number, deltaY: number) {
    markAnnotationsDirty();
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
      const nextAnnotations = response.items.map((annotation, index) => ({
          ...toDraftBox(annotation, index),
          id: undefined,
          image_id: selectedImageId,
          local_id: `copy-${adjacent.id}-${Date.now()}-${index}`,
        }));
      setAnnotations(nextAnnotations);
      setAnnotationStatus(nextAnnotations.length > 0 ? "annotated" : "unreviewed");
      markAnnotationsDirty();
      setSelectedAnnotationId(nextAnnotations[0]?.local_id ?? null);
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "复制标注失败");
    }
  }

  function addPredictionAsAnnotation(prediction: Prediction) {
    const classInfo = classById.get(prediction.class_id);
    const localId = predictionDraftId(prediction.id);
    setAnnotations((current) => {
      if (current.some((annotation) => annotation.local_id === localId)) {
        return current;
      }

      return [...current, predictionToDraftBox(prediction, classInfo)];
    });
    setAnnotationStatus("annotated");
    setSelectedAnnotationId(localId);
    markAnnotationsDirty();
    setShowGroundTruthLayer(true);
  }

  function markFalseNegativeReviewed(prediction: Prediction) {
    if (!prediction.matched_annotation_id) {
      return;
    }

    const matchedAnnotation = annotations.find(
      (annotation) => annotation.id === prediction.matched_annotation_id,
    );
    markAnnotationsDirty();
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
    if (matchedAnnotation) {
      setSelectedAnnotationId(matchedAnnotation.local_id);
    }
  }

  async function handleSaveAnnotations(): Promise<boolean> {
    if (!selectedImageId) {
      return false;
    }

    setIsSavingAnnotations(true);
    setAnnotationError(null);
    const selectedIndex = annotations.findIndex(
      (annotation) => annotation.local_id === selectedAnnotationId,
    );

    try {
      const requestedStatus = annotations.length > 0
        ? "annotated"
        : annotationStatus === "negative"
          ? "negative"
          : "unreviewed";
      const response = requestedStatus === "negative"
        ? await replaceAnnotations(selectedImageId, annotations.map(toAnnotationWrite), "negative")
        : await replaceAnnotations(selectedImageId, annotations.map(toAnnotationWrite));
      const nextAnnotations = response.items.map(toDraftBox);
      setAnnotations(nextAnnotations);
      clearAnnotationsDirty();
      setAnnotationStatus(response.annotation_status ?? requestedStatus);
      setSelectedAnnotationId(
        selectedIndex >= 0
          ? nextAnnotations[selectedIndex]?.local_id ?? null
          : nextAnnotations[0]?.local_id ?? null,
      );
      setImages((current) =>
        current.map((image) =>
          image.id === selectedImageId
            ? {
                ...image,
                annotation_count: response.items.length,
                annotation_status: response.annotation_status ?? requestedStatus,
              }
            : image,
        ),
      );
      await refreshProjects();
      void refreshTrainingPrep();
      return true;
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "保存标注失败");
      return false;
    } finally {
      setIsSavingAnnotations(false);
    }
  }

  async function handleSaveAndNextImage() {
    if (!selectedImageId) {
      return;
    }
    const currentImageId = selectedImageId;
    if (!(await handleSaveAnnotations())) {
      return;
    }
    const nextImageId = findNextAnnotationImageId(images, currentImageId);
    if (nextImageId !== null) {
      commitAnnotationImage(nextImageId);
    } else {
      setAnnotationNotice("已完成当前图像列表");
    }
  }

  async function handleConfirmNegative() {
    if (!selectedImageId || annotations.length > 0) {
      return;
    }

    setIsSavingAnnotations(true);
    setAnnotationError(null);
    try {
      const response = await replaceAnnotations(selectedImageId, [], "negative");
      setAnnotations([]);
      setAnnotationStatus(response.annotation_status ?? "negative");
      clearAnnotationsDirty();
      setSelectedAnnotationId(null);
      setImages((current) =>
        current.map((image) =>
          image.id === selectedImageId
            ? { ...image, annotation_count: 0, annotation_status: "negative" }
            : image,
        ),
      );
      setAnnotationNotice("已确认该图像无目标，已保存为负样本");
      void refreshTrainingPrep();
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "保存负样本失败");
    } finally {
      setIsSavingAnnotations(false);
    }
  }

  async function handleCreateDatasetVersion() {
    if (!importedDataset || versionDatasetIds.length === 0) {
      return;
    }
    if (versionDatasetIds.length === 1 && !quality?.ready_for_training) {
      return;
    }

    setIsCreatingVersion(true);
    setVersionError(null);

    try {
      const created = await createDatasetVersion(
        importedDataset.dataset_id,
        versionName.trim() || undefined,
        versionClassIds,
        versionImageScope,
        versionDatasetIds,
      );
      setVersions((current) => [created, ...current]);
      setSelectedTrainingVersionId(created.id);
      setVersionName("");
      await refreshTrainingPrep(importedDataset.dataset_id);
      await refreshProjects();
    } catch (error) {
      setVersionError(error instanceof Error ? error.message : "数据集版本导出失败");
    } finally {
      setIsCreatingVersion(false);
    }
  }

  function toggleVersionDataset(datasetId: number) {
    setVersionDatasetIds((current) => {
      if (current.includes(datasetId)) {
        if (importedDataset && datasetId === importedDataset.dataset_id) {
          return current;
        }
        if (current.length <= 1) {
          return current;
        }
        return current.filter((id) => id !== datasetId);
      }
      return [...current, datasetId];
    });
  }

  async function handleStartTrainingRun() {
    const version =
      versions.find((item) => item.id === selectedTrainingVersionId) ?? versions[0] ?? null;
    if (!version || !importedDataset) {
      return;
    }

    setIsStartingRun(true);
    setTrainingError(null);
    setTrainingNotice(null);

    try {
      const run = await createTrainingRun(buildTrainingRunRequest(version.id));
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      setTrainingNotice(`已使用版本「${version.name}」启动训练`);
      await refreshTrainingRuns(importedDataset.project_id);
      await refreshRunSummary(run.id);
      await loadRunLogs(run.id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "训练任务启动失败");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function handleRerunTrainingRun(sourceRun: TrainingRun) {
    const version =
      versions.find((item) => item.id === selectedTrainingVersionId) ?? versions[0] ?? null;
    if (!version || !importedDataset) {
      return;
    }

    setIsStartingRun(true);
    setTrainingError(null);
    setTrainingNotice(null);

    try {
      const run = await createTrainingRun(requestFromRunConfig(version.id, sourceRun));
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      setTrainingNotice(
        `已基于训练任务 #${sourceRun.id}，使用版本「${version.name}」开始重新训练`,
      );
      await refreshTrainingRuns(importedDataset.project_id);
      await refreshRunSummary(run.id);
      await loadRunLogs(run.id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "重新训练启动失败");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function loadRunLogs(runId: number) {
    try {
      const response = await getTrainingRunLogs(runId);
      setRunLogs((current) => ({ ...current, [runId]: response.text }));
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "训练日志加载失败");
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
      setTrainingError(error instanceof Error ? error.message : "取消训练失败");
    } finally {
      setCancellingRunId(null);
    }
  }

  async function handleCreatePredictionJob() {
    const run = selectedEvalRun;
    if (!run || run.status !== "completed") {
      setPredictionError("请选择已完成的训练任务作为评估权重");
      return;
    }
    if (!selectedEvalVersionId) {
      setPredictionError("请选择评估数据版本");
      return;
    }

    setIsCreatingPrediction(true);
    setPredictionError(null);

    try {
      const job = await createPredictionJob(run.id, {
        image_scope: predictionScope,
        confidence_threshold: predictionConfidence,
        version_id: selectedEvalVersionId,
      });
      setPredictionJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      await loadFilteredPredictions(job.id);
      await refreshRunSummary(run.id);
      await loadPredictionLogs(job.id);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "预测任务启动失败");
    } finally {
      setIsCreatingPrediction(false);
    }
  }

  async function loadPredictionLogs(jobId: number) {
    try {
      const response = await getPredictionJobLogs(jobId);
      setPredictionLogs((current) => ({ ...current, [jobId]: response.text }));
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "预测日志加载失败");
    }
  }

  async function handleCreateExport(format: string) {
    const run = selectedEvalRun;
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
      setExportError(error instanceof Error ? error.message : "模型导出失败");
    } finally {
      setIsCreatingExport(null);
    }
  }

  function handleSelectEvalRun(runId: number) {
    setSelectedEvalRunId(runId);
    const run = runs.find((item) => item.id === runId);
    if (run) {
      setSelectedEvalVersionId(run.version_id);
    }
  }
  async function openPredictionImage(prediction: Prediction) {
    setPredictionError(null);

    try {
      const review = await getPredictionImageReview(prediction.job_id, prediction.image_id);
      setEvalReview(review);
      setEvalReviewFocusId(prediction.id);
      setShowGroundTruthLayer(true);
      setShowPredictionLayer(true);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "预测结果审查加载失败");
    }
  }

  async function navigateEvalReview(delta: number) {
    if (predictions.length === 0 || evalReviewFocusId == null) {
      return;
    }
    const currentIndex = predictions.findIndex((item) => item.id === evalReviewFocusId);
    if (currentIndex < 0) {
      return;
    }
    const nextIndex = Math.min(predictions.length - 1, Math.max(0, currentIndex + delta));
    if (nextIndex === currentIndex) {
      return;
    }
    await openPredictionImage(predictions[nextIndex]);
  }

  function openEvalReviewInAnnotation() {
    if (!evalReview) {
      return;
    }

    const review = evalReview;
    const existingImage = images.find((image) => image.id === review.image.id);
    const reviewAnnotations = review.annotations.map(toDraftBox);
    setActiveReview(review);
    setAnnotations(reviewAnnotations);
    setSelectedAnnotationId(reviewAnnotations[0]?.local_id ?? null);
    clearAnnotationsDirty();
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
          annotation_status: review.annotations.length > 0 ? "annotated" : "unreviewed",
          image_url: review.image.image_url,
        },
      ]);
    }
    setSelectedImageId(review.image.id);
    commitWorkflowStep("annotation");
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
          annotation_status: issue.issue_type === "unannotated_image" ? "unreviewed" : "annotated",
          image_url: issue.image_url,
        },
      ];
    });
    setSelectedImageId(issue.image_id);
    commitWorkflowStep("annotation");
  }

  if (authLoading) {
    return (
      <main className="login-shell">
        <section className="login-card">
          <p className="muted">正在恢复登录状态…</p>
        </section>
      </main>
    );
  }

  if (!user) {
    return <LoginView />;
  }

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="应用状态">
        <div>
          <p className="eyebrow">本地目标检测工作台</p>
          <h1>YoloStudio</h1>
        </div>
        <div className="topbar-actions">
          <div className="device-pill">
            <Activity size={16} />
            <span>{health?.devices.selected ?? "连接中"}</span>
          </div>
          <div className="user-pill" aria-label="当前用户">
            <span>
              {user.username} · {roleLabel(user.role)}
            </span>
            {isAdmin ? (
              <button
                type="button"
                className="secondary-button"
                onClick={() => setShowUserManagement((current) => !current)}
              >
                <Users size={16} />
                {showUserManagement ? "返回工作流" : "用户管理"}
              </button>
            ) : null}
            <button type="button" className="secondary-button" onClick={logout}>
              <LogOut size={16} />
              退出
            </button>
          </div>
        </div>
      </section>

      <section className="status-grid" aria-label="后端详情">
        <StatusTile
          icon={<HardDrive size={20} />}
          label="工作空间"
          value={health?.workspace_root ?? "等待后端响应"}
          revealFullValue={Boolean(health?.workspace_root)}
        />
        <StatusTile
          icon={<Database size={20} />}
          label="数据库"
          value={health?.database_path ?? "SQLite 将在启动时初始化"}
          revealFullValue={Boolean(health?.database_path)}
        />
        <StatusTile
          icon={<Activity size={20} />}
          label="计算设备"
          value={health ? `${health.devices.available.length} 个可用` : "检测中"}
        />
      </section>

      {healthError ? <div className="error-banner">{healthError}</div> : null}

      {showUserManagement && isAdmin ? <UserManagementView /> : null}

      {!showUserManagement ? (
      <WorkflowShell
        currentStep={currentStep}
        steps={workflowSteps}
        navigationNotice={navigationNotice}
        onNavigate={handleWorkflowNavigate}
      >
      {pendingWorkflowStep ? (
        <div role="alert" className="navigation-notice">
          <p>当前图像的标注尚未保存。离开后，这些修改将被放弃。</p>
          <div>
            <button type="button" className="secondary-button" onClick={handleStayOnAnnotation}>
              留在标注页
            </button>
            <button type="button" onClick={() => void handleDiscardAnnotationsAndLeave()}>
              放弃修改并离开
            </button>
          </div>
        </div>
      ) : null}
      {currentStep === "dataset" ? <div className="dataset-page">
        {isAdmin ? (
        <div className="dataset-primary-tabs" role="tablist" aria-label="项目与数据视图">
          <button
            type="button"
            role="tab"
            aria-selected={datasetPageTab === "import"}
            onClick={() => setDatasetPageTab("import")}
          >
            <Upload size={16} />
            导入数据
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={datasetPageTab === "management"}
            onClick={() => setDatasetPageTab("management")}
          >
            <Database size={16} />
            数据管理
          </button>
        </div>
        ) : null}

        {datasetPageTab === "import" || !isAdmin ? <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">{isAdmin ? "数据集导入" : "数据集加载"}</p>
            <h2>{isAdmin ? "扫描本地数据集" : "加载已保存数据集"}</h2>
          </div>
          <FolderSearch size={22} />
        </div>

        <form className="scan-form" onSubmit={handleScan}>
          {isAdmin ? (
          <>
          <label htmlFor="dataset-source">数据集来源</label>
          <div className="input-row dataset-source-row">
            <select
              id="dataset-source"
              aria-label="数据集来源"
              value={selectedDatasetSourceRef}
              onChange={(event) => {
                const option = datasetSourceOptions.find(
                  (item) => item.source_ref === event.target.value,
                );
                applyDatasetSourceOption(option);
                setScan(null);
              }}
            >
              {datasetSourceOptions.length === 0 ? (
                <option value="">暂无可扫描数据集，请先上传 ZIP</option>
              ) : null}
              {datasetSourceOptions.some((item) => item.kind === "server") ? (
                <optgroup label="服务器文件">
                  {datasetSourceOptions
                    .filter((item) => item.kind === "server")
                    .map((item) => (
                      <option key={item.source_ref} value={item.source_ref}>
                        {item.label}（{formatBytes(item.size_bytes)}）
                      </option>
                    ))}
                </optgroup>
              ) : null}
              {datasetSourceOptions.some((item) => item.kind === "upload") ? (
                <optgroup label="已上传压缩包">
                  {datasetSourceOptions
                    .filter((item) => item.kind === "upload")
                    .map((item) => (
                      <option key={item.source_ref} value={item.source_ref}>
                        {item.label}（{formatBytes(item.size_bytes)}）
                      </option>
                    ))}
                </optgroup>
              ) : null}
            </select>
            <label className="preview-upload-button">
              <Upload size={16} /> {isUploadingDatasetSource ? "上传中..." : "上传 ZIP"}
              <input
                aria-label="上传数据集 ZIP"
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => void handleUploadDatasetSource(event)}
                disabled={isUploadingDatasetSource}
              />
            </label>
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

          {datasetSourceOptions.some((item) => item.kind === "upload") ? (
            <div className="dataset-source-library" aria-label="已上传数据集">
              <table className="training-weight-table" aria-label="已上传数据集列表">
                <thead>
                  <tr>
                    <th scope="col">文件名</th>
                    <th scope="col">大小</th>
                    <th scope="col">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {datasetSourceOptions
                    .filter((item) => item.kind === "upload")
                    .map((item) => (
                      <tr
                        key={item.source_ref}
                        className={
                          selectedDatasetSourceRef === item.source_ref ? "is-selected" : undefined
                        }
                      >
                        <th scope="row">{item.original_filename}</th>
                        <td>{formatBytes(item.size_bytes)}</td>
                        <td>
                          <div className="training-weight-row-actions">
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() => {
                                applyDatasetSourceOption(item);
                                setScan(null);
                              }}
                            >
                              选用
                            </button>
                            <button
                              type="button"
                              className="secondary-button danger-button"
                              aria-label={`删除数据集 ${item.original_filename}`}
                              disabled={deletingDatasetSourceId === item.source_id}
                              onClick={() => void handleDeleteDatasetSource(item)}
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty-state dataset-source-hint">
              可上传本地 YOLO 数据集 ZIP 压缩包；上传后会出现在上方下拉菜单中。
            </p>
          )}

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
          </>
          ) : null}
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
                      {project.name} / {dataset.name}（{dataset.image_count} 张图像 · 已标注{" "}
                      {dataset.annotated_image_count ?? 0} 张 · {dataset.annotation_count ?? 0} 个边界框）
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
                    <p key={warning}>{formatScanWarning(warning)}</p>
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
        </section> : (
          <StorageManagementView
            loadedDatasetId={importedDataset?.dataset_id ?? null}
            openingDatasetId={openingStorageDatasetId}
            onOpenDataset={handleOpenDatasetFromStorage}
            onDatasetTrashed={() => clearLoadedDatasetWorkspace()}
            onStorageChanged={handleStorageChanged}
          />
        )}
      </div> : null}

      {currentStep === "quality" ? <section className="prep-grid" aria-label="训练准备情况">
        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">训练准备</p>
              <h2>质量审查</h2>
            </div>
            <div className="panel-heading-actions">
              {importedDataset && isAdmin ? (
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
                    <p key={issue}>{formatQualitySummaryIssue(issue)}</p>
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
                {isAdmin ? (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!canApplyQualityTags}
                  onClick={handleApplyQualityTags}
                >
                  <Tags size={16} />
                  {isApplyingQualityTags ? "正在应用标签" : "应用自动标签"}
                </button>
                ) : null}
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
                        <span>{formatQualityIssueMessage(issue)}</span>
                        <small>
                          {issue.image_path}
                          {issue.class_name ? ` | ${formatClassDisplayName(issue.class_name)}` : ""}
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
            {isAdmin ? (
            <>
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
                disabled={
                  !importedDataset
                  || versionDatasetIds.length === 0
                  || versionClassIds.length === 0
                  || isCreatingVersion
                  || (versionDatasetIds.length === 1 && !quality?.ready_for_training)
                }
                onClick={() => void handleCreateDatasetVersion()}
              >
                {isCreatingVersion
                  ? "导出中"
                  : versionDatasetIds.length > 1
                    ? "合并创建训练版本"
                    : "创建数据集版本"}
              </button>
            </div>
            <div className="subset-controls" aria-label="合并训练数据集">
              <span>参与训练的数据集</span>
              <p className="version-scope-hint">
                勾选同一项目下的多个数据集，可合并成一个训练版本，避免只训新数据导致旧数据遗忘。
                当前数据集默认选中且不可取消。
              </p>
              <div className="subset-grid version-dataset-grid">
                {projectDatasets.length === 0 ? (
                  <p className="empty-state">暂无同项目数据集可选。</p>
                ) : (
                  projectDatasets.map((dataset) => {
                    const isCurrent = importedDataset?.dataset_id === dataset.id;
                    const checked = versionDatasetIds.includes(dataset.id);
                    return (
                      <label key={dataset.id}>
                        <input
                          type="checkbox"
                          aria-label={`合并数据集 ${dataset.name}`}
                          checked={checked}
                          disabled={!importedDataset || isCurrent}
                          onChange={() => toggleVersionDataset(dataset.id)}
                        />
                        <span>
                          {dataset.name}
                          {isCurrent ? "（当前）" : ""}
                          {" · "}
                          {dataset.image_count} 张 / 已标注 {dataset.annotated_image_count ?? 0}
                        </span>
                      </label>
                    );
                  })
                )}
              </div>
              {versionDatasetIds.length > 1 ? (
                <p className="version-scope-hint">
                  已选 {versionDatasetIds.length} 个数据集，将合并导出后再训练。
                </p>
              ) : null}
            </div>
            <label htmlFor="version-image-scope">导出范围</label>
            <select
              id="version-image-scope"
              aria-label="版本导出范围"
              value={versionImageScope}
              disabled={!importedDataset}
              onChange={(event) => setVersionImageScope(event.target.value as "annotated" | "all")}
            >
              <option value="annotated">
                仅已标注图像（当前集 {quality?.annotated_image_count ?? 0} 张）
              </option>
              <option value="all">
                全部图像（当前集 {quality?.image_count ?? 0} 张）
              </option>
            </select>
            <p className="version-scope-hint">
              导出范围作用于所有勾选数据集。当前范围参考当前集约 {versionImageCount} 张；
              合并后实际数量以后端汇总为准。
              {versionImageScope === "annotated"
                ? "未标注图像不会复制到训练版本。"
                : "全量范围会包含尚未标注的图像，空标签将作为负样本参与训练。"}
            </p>
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
            </>
            ) : (
              <p className="muted">标注员可查看已有版本；创建版本由管理员操作。</p>
            )}
          </div>

          {versionError ? <div className="error-banner">{versionError}</div> : null}

          <div className="version-list" aria-label="数据集版本">
            {versions.length === 0 ? (
              <p className="empty-state">导出的 YOLO 数据集版本将显示在这里。</p>
            ) : (
              versions.map((version) => (
                <div className="version-row" key={version.id}>
                  <div>
                    <strong>
                      {version.name}
                      {version.merged ? " · 合并版" : ""}
                    </strong>
                    <span>{version.artifact_path}</span>
                    {version.source_dataset_ids && version.source_dataset_ids.length > 1 ? (
                      <span>
                        来源数据集：
                        {version.source_dataset_ids
                          .map((id) => {
                            const match = projectDatasets.find((item) => item.id === id);
                            return match ? match.name : `#${id}`;
                          })
                          .join(" + ")}
                      </span>
                    ) : null}
                  </div>
                  <span>
                    {formatVersionImageScope(version.image_scope)} |{" "}
                    {version.split_counts.train}/{version.split_counts.val}/
                    {version.split_counts.test}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>
      </section> : null}

      {currentStep === "training" ? <section className="training-grid" aria-label="训练设置与训练记录">
        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">模型训练</p>
              <h2>训练设置</h2>
            </div>
            <Play size={20} />
          </div>

          <div className="training-form">
            <label htmlFor="training-version">训练数据版本</label>
            <select
              id="training-version"
              aria-label="训练数据版本"
              value={selectedTrainingVersionId ?? ""}
              onChange={(event) =>
                setSelectedTrainingVersionId(
                  event.target.value ? Number(event.target.value) : null,
                )
              }
              disabled={versions.length === 0}
            >
              {versions.length === 0 ? (
                <option value="">请先在「质量与版本」创建标注版本</option>
              ) : null}
              {versions.map((version) => (
                <option key={version.id} value={version.id}>
                  {version.name}
                  {version.merged ? " · 合并版" : ""}
                  {" · "}
                  训练{version.split_counts.train}/验证{version.split_counts.val}/测试
                  {version.split_counts.test}
                </option>
              ))}
            </select>
            {selectedTrainingVersionId ? (
              <p className="version-scope-hint">
                {(() => {
                  const selected =
                    versions.find((item) => item.id === selectedTrainingVersionId) ?? null;
                  if (!selected) {
                    return "请选择要用于训练的标注版本。";
                  }
                  if (selected.merged && selected.source_dataset_ids?.length) {
                    const names = selected.source_dataset_ids
                      .map((id) => {
                        const match = projectDatasets.find((item) => item.id === id);
                        return match ? match.name : `#${id}`;
                      })
                      .join(" + ");
                    return `当前为合并版，来源：${names}`;
                  }
                  return `将使用版本「${selected.name}」进行训练。`;
                })()}
              </p>
            ) : null}

            <label htmlFor="training-model">起始模型</label>
            <select
              id="training-model"
              value={trainingModel}
              onChange={(event) => setTrainingModel(event.target.value)}
              disabled={versions.length === 0}
            >
              {!trainingModelOptions.some((item) => item.model_ref === trainingModel) &&
              trainingModel ? (
                <option value={trainingModel}>{trainingModel}（历史配置）</option>
              ) : null}
              <optgroup label="内置模型">
                {trainingModelOptions
                  .filter((item) => item.kind === "base")
                  .map((item) => (
                    <option key={item.model_ref} value={item.model_ref}>
                      {item.label}
                    </option>
                  ))}
              </optgroup>
              {trainingModelOptions.some((item) => item.kind === "trained") ? (
                <optgroup label="本项目训练结果">
                  {trainingModelOptions
                    .filter((item) => item.kind === "trained")
                    .map((item) => (
                      <option key={item.model_ref} value={item.model_ref}>
                        {item.label}
                      </option>
                    ))}
                </optgroup>
              ) : null}
              {trainingModelOptions.some((item) => item.kind === "upload") ? (
                <optgroup label="已上传权重">
                  {trainingModelOptions
                    .filter((item) => item.kind === "upload")
                    .map((item) => (
                      <option key={item.model_ref} value={item.model_ref}>
                        {item.label}
                      </option>
                    ))}
                </optgroup>
              ) : null}
            </select>

            <div className="training-weight-library" aria-label="权重库">
              <div className="training-weight-library-heading">
                <div>
                  <strong>权重库</strong>
                  <span>上传 .pt 后可在上方下拉中选择，用于继续训练</span>
                </div>
                <label className="preview-upload-button">
                  <Upload size={16} /> {isUploadingWeight ? "正在上传..." : "上传权重"}
                  <input
                    aria-label="上传训练权重"
                    type="file"
                    accept=".pt,application/octet-stream"
                    onChange={(event) => void handleUploadModelWeight(event)}
                    disabled={isUploadingWeight || !importedDataset || versions.length === 0}
                  />
                </label>
              </div>
              {modelWeights.length === 0 ? (
                <p className="empty-state">暂无已上传权重。</p>
              ) : (
                <table className="training-weight-table" aria-label="权重库列表">
                  <thead>
                    <tr>
                      <th scope="col">文件名</th>
                      <th scope="col">大小</th>
                      <th scope="col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelWeights.map((weight) => (
                      <tr
                        key={weight.id}
                        className={
                          trainingModel === `weight:${weight.id}` ? "is-selected" : undefined
                        }
                      >
                        <th scope="row">{weight.original_filename}</th>
                        <td>{formatBytes(weight.size_bytes)}</td>
                        <td>
                          <div className="training-weight-row-actions">
                            <button
                              type="button"
                              className="secondary-button"
                              onClick={() => setTrainingModel(`weight:${weight.id}`)}
                              disabled={versions.length === 0}
                            >
                              选用
                            </button>
                            <button
                              type="button"
                              className="secondary-button danger-button"
                              aria-label={`删除权重 ${weight.original_filename}`}
                              onClick={() => void handleDeleteModelWeight(weight)}
                              disabled={deletingWeightId === weight.id}
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

            <div className="training-number-grid">
              <label htmlFor="training-epochs">
                训练轮数
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
                图像尺寸
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
                批大小
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
                计算设备
                <input
                  id="training-device"
                  value={trainingDevice}
                  onChange={(event) => setTrainingDevice(event.target.value)}
                  placeholder={health?.devices.selected ?? "cpu"}
                  disabled={versions.length === 0}
                />
              </label>
            </div>

            <div className="augmentation-panel" aria-label="数据增强策略">
              <label htmlFor="augmentation-preset">
                策略名称
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
                  label="随机擦除"
                  value={augmentation.erasing}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("erasing", value)}
                />
                <AugmentationNumber
                  id="aug-scale"
                  label="缩放"
                  max={2}
                  value={augmentation.scale}
                  disabled={versions.length === 0}
                  onChange={(value) => setAugmentationValue("scale", value)}
                />
                <AugmentationNumber
                  id="aug-fliplr"
                  label="水平翻转"
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
                自动阈值扫描
              </label>
            </div>

            <button
              type="button"
              disabled={versions.length === 0 || !selectedTrainingVersionId || isStartingRun}
              onClick={() => void handleStartTrainingRun()}
            >
              <Play size={16} />
              {isStartingRun ? "正在启动" : "开始训练"}
            </button>
          </div>

          {trainingError ? <div className="error-banner">{trainingError}</div> : null}
          {trainingNotice ? <div className="success-banner">{trainingNotice}</div> : null}
        </section>

        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">实验</p>
              <h2>训练记录</h2>
              <span className="monitor-state">
                {hasActiveRun ? "自动刷新中" : "空闲"}
              </span>
            </div>
            <Activity size={20} />
          </div>

          <div className="run-list" aria-label="训练任务">
            {runs.length === 0 ? (
              <p className="empty-state">请先创建数据集版本，再开始训练。</p>
            ) : (
              runs.map((run) => (
                <div className="run-row" key={run.id}>
                  <div className="run-row-heading">
                    <strong>训练任务 #{run.id}</strong>
                    <span className={`run-status ${run.status}`}>{formatRunStatus(run.status)}</span>
                  </div>
                  <span>{run.artifact_path}</span>
                  <span>
                    {String(run.config.model ?? "未指定模型")} | {String(run.config.epochs ?? "?")} 轮 |{" "}
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
                    加载日志
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setLiveMonitorRunId(run.id)}
                  >
                    实时监控
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => applyRunConfigToForm(run)}
                  >
                    加载配置
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={versions.length === 0 || hasActiveRun || isStartingRun}
                    onClick={() => handleRerunTrainingRun(run)}
                  >
                    重新训练
                  </button>
                  {isActiveRun(run.status) ? (
                    <button
                      type="button"
                      className="secondary-button danger-button"
                      disabled={cancellingRunId === run.id}
                      onClick={() => handleCancelTrainingRun(run.id)}
                    >
                      {cancellingRunId === run.id ? "正在取消" : "取消训练"}
                    </button>
                  ) : null}
                  {runLogs[run.id] ? <pre className="log-preview">{runLogs[run.id]}</pre> : null}
                </div>
              ))
            )}
          </div>

        </section>
      </section> : null}

      {currentStep === "preview" ? (
        <PreviewView
          projectId={importedDataset?.project_id ?? null}
          datasets={previewDatasets}
          runs={runs}
          classes={classes}
        />
      ) : null}

      {currentStep === "evaluation" ? <>
      <section className="panel prediction-panel" aria-label="预测分析">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">模型评估</p>
            <h2>预测分析</h2>
            <span className="monitor-state">
              {hasActivePredictionJob ? "自动刷新中" : "空闲"}
            </span>
          </div>
          <Radar size={20} />
        </div>

        <div className="prediction-controls">
          <label htmlFor="eval-run">
            评估权重
            <select
              id="eval-run"
              value={selectedEvalRunId ?? ""}
              onChange={(event) => handleSelectEvalRun(Number(event.target.value))}
              disabled={completedEvalRuns.length === 0}
            >
              {completedEvalRuns.length === 0 ? (
                <option value="">暂无已完成训练任务</option>
              ) : (
                completedEvalRuns.map((run) => {
                  const version = projectVersions.find((item) => item.id === run.version_id);
                  const versionLabel = version
                    ? `${version.name}${version.merged ? " · 合并" : ""}`
                    : `版本 #${run.version_id}`;
                  return (
                    <option key={run.id} value={run.id}>
                      训练任务 #{run.id} · {versionLabel} · best.pt
                    </option>
                  );
                })
              )}
            </select>
          </label>
          <label htmlFor="eval-version">
            评估数据版本
            <select
              id="eval-version"
              value={selectedEvalVersionId ?? ""}
              onChange={(event) => setSelectedEvalVersionId(Number(event.target.value))}
              disabled={projectVersions.length === 0}
            >
              {projectVersions.length === 0 ? (
                <option value="">暂无数据版本</option>
              ) : (
                projectVersions.map((version) => (
                  <option key={version.id} value={version.id}>
                    #{version.id} · {version.name}
                    {version.merged ? " · 合并" : ""}
                    {" · "}
                    训练{version.split_counts.train}/验证{version.split_counts.val}/测试
                    {version.split_counts.test}
                  </option>
                ))
              )}
            </select>
          </label>
          <label htmlFor="prediction-scope">
            图像范围
            <select
              id="prediction-scope"
              value={predictionScope}
              onChange={(event) => setPredictionScope(event.target.value)}
              disabled={!selectedEvalRun}
            >
              <option value="all">全部</option>
              <option value="train">训练集</option>
              <option value="val">验证集</option>
              <option value="test">测试集</option>
            </select>
          </label>
          <label htmlFor="prediction-confidence">
            置信度阈值
            <input
              id="prediction-confidence"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={predictionConfidence}
              onChange={(event) => setPredictionConfidence(Number(event.target.value))}
              disabled={!selectedEvalRun}
            />
          </label>
          <button
            type="button"
            disabled={
              !selectedEvalRun ||
              selectedEvalRun.status !== "completed" ||
              !selectedEvalVersionId ||
              isCreatingPrediction
            }
            onClick={() => void handleCreatePredictionJob()}
          >
            <Radar size={16} />
            {isCreatingPrediction ? "正在分析" : "开始预测分析"}
          </button>
        </div>

        <div className="prediction-filter-panel" aria-label="预测样本筛选">
          <label htmlFor="prediction-filter-failure">
            结果类型
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
            预测类别
            <select
              id="prediction-filter-class"
              value={predictionFilters.class_id}
              onChange={(event) =>
                setPredictionFilters((current) => ({ ...current, class_id: event.target.value }))
              }
              disabled={predictionJobs.length === 0}
            >
              <option value="">全部类别</option>
              {classes.map((classItem) => (
                <option key={classItem.id} value={classItem.id}>
                  {classItem.name}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor="prediction-filter-conf-min">
            最低置信度
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
            最高置信度
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
          <div className="prediction-filter-actions">
            <button
              type="button"
              disabled={predictionJobs.length === 0}
              onClick={handleApplyPredictionFilters}
            >
              应用样本筛选
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={predictionJobs.length === 0}
              onClick={handleResetPredictionFilters}
            >
              重置
            </button>
          </div>
        </div>

        {predictionError ? <div className="error-banner">{predictionError}</div> : null}

        {predictionJobs[0] ? (
          <div className="prediction-summary">
            <Metric label="图像" value={predictionJobs[0].image_count.toLocaleString()} />
            <Metric label="匹配正确" value={predictionJobs[0].matched_count.toLocaleString()} />
            <Metric label="误报" value={predictionJobs[0].false_positive_count.toLocaleString()} />
            <Metric label="漏报" value={predictionJobs[0].false_negative_count.toLocaleString()} />
            <Metric label="类别混淆" value={predictionJobs[0].class_confusion_count.toLocaleString()} />
          </div>
        ) : (
          <p className="empty-state">选择已完成的训练权重与数据版本后，开始预测分析以审查模型输出。</p>
        )}

        <div className="prediction-layout">
          <div className="prediction-list" aria-label="预测样本">
            {predictions.length === 0 ? (
              <p className="empty-state">预测结果与问题样本将显示在这里。</p>
            ) : (
              predictions.map((prediction) => (
                <button
                  type="button"
                  className={
                    prediction.id === evalReviewFocusId
                      ? "prediction-row selected"
                      : "prediction-row"
                  }
                  key={prediction.id}
                  aria-pressed={prediction.id === evalReviewFocusId}
                  onClick={() => void openPredictionImage(prediction)}
                >
                  <div>
                    <strong>{formatFailureType(prediction.failure_type)}</strong>
                    <span>
                      图像 #{prediction.image_id} |{" "}
                      {classById.get(prediction.class_id)?.name ?? `类别 #${prediction.class_id}`}
                    </span>
                  </div>
                  <span>{prediction.confidence.toFixed(2)}</span>
                </button>
              ))
            )}
          </div>

          <div className="eval-review-panel" aria-label="问题样本查看器">
            {evalReview ? (
              <>
                <div className="eval-review-toolbar">
                  <div>
                    <strong>筛选样本查看</strong>
                    <span>
                      {evalReviewIndex >= 0
                        ? `${evalReviewIndex + 1} / ${predictions.length}`
                        : `— / ${predictions.length}`}
                      {" · "}
                      {evalReview.image.relative_path}
                    </span>
                  </div>
                  <div className="eval-review-nav">
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={evalReviewIndex <= 0}
                      onClick={() => void navigateEvalReview(-1)}
                    >
                      上一张
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={
                        evalReviewIndex < 0 || evalReviewIndex >= predictions.length - 1
                      }
                      onClick={() => void navigateEvalReview(1)}
                    >
                      下一张
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={openEvalReviewInAnnotation}
                    >
                      去标注页修正
                    </button>
                  </div>
                </div>

                {evalFocusedPrediction ? (
                  <div className="eval-review-focus">
                    <strong>{formatFailureType(evalFocusedPrediction.failure_type)}</strong>
                    <span>
                      {classById.get(evalFocusedPrediction.class_id)?.name ??
                        `类别 #${evalFocusedPrediction.class_id}`}{" "}
                      · 置信度 {evalFocusedPrediction.confidence.toFixed(2)}
                    </span>
                  </div>
                ) : null}

                <div className="eval-review-layers" aria-label="评估审查图层">
                  <label>
                    <input
                      type="checkbox"
                      checked={showGroundTruthLayer}
                      onChange={(event) => setShowGroundTruthLayer(event.target.checked)}
                    />
                    真值标注
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={showPredictionLayer}
                      onChange={(event) => setShowPredictionLayer(event.target.checked)}
                    />
                    预测框
                  </label>
                  <div className="review-legend" aria-label="预测结果图例">
                    <span className="legend matched">匹配正确</span>
                    <span className="legend false-positive">误报</span>
                    <span className="legend false-negative">漏报</span>
                  </div>
                </div>

                <div className="eval-review-counts">
                  匹配正确 {evalReview.counts.matched ?? 0} | 误报{" "}
                  {evalReview.counts.false_positive ?? 0} | 漏报{" "}
                  {evalReview.counts.false_negative ?? 0} | 类别混淆{" "}
                  {evalReview.counts.class_confusion ?? 0}
                </div>

                <div className="eval-review-canvas" aria-label="评估样本画布">
                  <img
                    src={authedMediaUrl(evalReview.image.image_url)}
                    alt={evalReview.image.relative_path}
                    draggable={false}
                  />
                  <svg viewBox="0 0 1 1" preserveAspectRatio="none">
                    {showGroundTruthLayer
                      ? evalReview.annotations.map((annotation) => (
                          <ReviewAnnotationRect
                            key={annotation.id}
                            annotation={annotation}
                            highlighted={
                              evalFocusedPrediction?.matched_annotation_id === annotation.id
                            }
                          />
                        ))
                      : null}
                    {showPredictionLayer
                      ? evalReview.predictions.map((prediction) => (
                          <PredictionRect
                            key={prediction.id}
                            prediction={prediction}
                            className={classById.get(prediction.class_id)?.name}
                            highlighted={prediction.id === evalReviewFocusId}
                          />
                        ))
                      : null}
                  </svg>
                </div>
              </>
            ) : (
              <p className="empty-state">
                点击左侧筛选结果即可在此查看图像。可用上一张 / 下一张在当前筛选列表中浏览。
              </p>
            )}
          </div>
        </div>

        <div className="prediction-jobs" aria-label="预测任务">
          {predictionJobs.length === 0 ? null : (
            predictionJobs.map((job) => (
              <div className="run-row" key={job.id}>
                <div className="run-row-heading">
                  <strong>预测任务 #{job.id}</strong>
                  <span className={`run-status ${job.status}`}>{formatRunStatus(job.status)}</span>
                </div>
                <span>
                  数据版本 #{job.version_id ?? "—"} · 范围 {job.image_scope} · 置信度{" "}
                  {job.confidence_threshold}
                </span>
                {job.error_message ? <p className="run-error">{job.error_message}</p> : null}
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => loadPredictionLogs(job.id)}
                >
                  加载预测日志
                </button>
                {predictionLogs[job.id] ? (
                  <pre className="log-preview">{predictionLogs[job.id]}</pre>
                ) : null}
              </div>
            ))
          )}
        </div>
      </section>

      {summaryError ? <div className="error-banner">{summaryError}</div> : null}
      <section className="panel eval-advanced-panel" aria-label="高级实验看板">
        <button
          type="button"
          className="secondary-button eval-advanced-toggle"
          onClick={() => setShowEvalAdvanced((current) => !current)}
        >
          {showEvalAdvanced ? "收起高级：实验看板" : "展开高级：实验看板"}
        </button>
        {showEvalAdvanced ? (
          <ExperimentDashboard
            summary={runSummary}
            projectSummary={projectExperimentSummary}
          />
        ) : null}
      </section>

      <section className="panel export-panel" aria-label="模型导出">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">部署产物</p>
            <h2>模型导出</h2>
          </div>
          <Share2 size={20} />
        </div>

        {exportError ? <div className="error-banner">{exportError}</div> : null}

        <div className="export-grid">
          <ExportOption
            title=".pt 权重"
            format="pt"
            enabled={Boolean(
              selectedEvalRun &&
                selectedEvalRun.status === "completed" &&
                exportCapabilities?.pt_available,
            )}
            reason={formatExportReason(exportCapabilities?.reasons.pt)}
            isCreating={isCreatingExport === "pt"}
            onCreate={handleCreateExport}
          />
          {selectedEvalRunId && exportCapabilities?.pt_available ? (
            <a
              className="secondary-button export-download-link"
              href={authedMediaUrl(getRunBestWeightsUrl(selectedEvalRunId))}
              download={`run-${selectedEvalRunId}-best.pt`}
            >
              <Download size={16} />
              下载训练权重 best.pt
            </a>
          ) : null}
        </div>

        {exportCapabilities?.weights_path ? (
          <p className="export-source">
            源权重（训练任务 #{selectedEvalRunId}）：{exportCapabilities.weights_path}
          </p>
        ) : (
          <p className="empty-state">
            请选择已完成并生成 `ultralytics/weights/best.pt` 的训练任务，以启用模型导出。
          </p>
        )}

        <div className="export-list" aria-label="导出产物">
          {exports.length === 0 ? (
            <p className="empty-state">导出的模型产物将显示在这里。</p>
          ) : (
            exports.map((artifact) => {
              const downloadUrl =
                artifact.download_url ||
                (artifact.status === "completed"
                  ? getExportDownloadUrl(artifact.run_id, artifact.id)
                  : null);
              return (
                <div className="export-row" key={artifact.id}>
                  <div>
                    <strong>
                      {artifact.format.toUpperCase()} 导出任务 #{artifact.id}
                    </strong>
                    <span>{artifact.artifact_path || "暂无产物路径"}</span>
                  </div>
                  <div className="export-row-actions">
                    <span className={`run-status ${artifact.status}`}>
                      {formatRunStatus(artifact.status)}
                    </span>
                    {downloadUrl ? (
                      <a
                        className="secondary-button"
                        href={authedMediaUrl(downloadUrl)}
                        download={`run-${artifact.run_id}.${artifact.format}`}
                      >
                        <Download size={16} />
                        下载
                      </a>
                    ) : null}
                  </div>
                  {artifact.error_message ? (
                    <p className="run-error">{artifact.error_message}</p>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </section>
      </> : null}

      {currentStep === "classes" ? <section className="panel class-management-panel" aria-label="类别管理">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">项目标签</p>
              <h2>类别库</h2>
            </div>
            <Library size={20} />
          </div>

          <div className="fixed-class-management">
            {isAdmin ? (
            <>
            <div className="fixed-class-heading">
              <div>
                <h3>从数据集文件名识别到的类别</h3>
                <p>
                  已根据 ZIP 中重复的文件名前缀自动扫描类别。默认全部勾选，可取消不需要的项后确认启用。
                </p>
              </div>
              <span className="fixed-class-count">
                已选 {selectedDetectedClassNames.length} / {detectedClasses.length}
              </span>
            </div>
            {isLoadingDetectedClasses ? (
              <p className="empty-state">正在扫描文件名类别...</p>
            ) : detectedClasses.length === 0 ? (
              <p className="empty-state">
                未能从文件名识别出重复前缀类别。请确认图像命名类似
                <code>fire_truck_h20_a045.jpg</code>，或稍后手动添加类别。
              </p>
            ) : (
              <fieldset
                className="fixed-class-picker"
                disabled={!importedDataset || isEnablingDetectedClasses}
              >
                <legend className="sr-only">扫描到的目标类别</legend>
                {detectedClasses.map((preset) => {
                  const isSelected = selectedDetectedClassNames.includes(preset.name);
                  const isExisting = classes.some((classItem) => classItem.name === preset.name);
                  return (
                    <label
                      className={isSelected ? "fixed-class-option selected" : "fixed-class-option"}
                      key={preset.name}
                    >
                      <input
                        type="checkbox"
                        aria-label={preset.name}
                        checked={isSelected}
                        onChange={() => {
                          setSelectedDetectedClassNames((current) =>
                            current.includes(preset.name)
                              ? current.filter((name) => name !== preset.name)
                              : [...current, preset.name],
                          );
                        }}
                      />
                      <span
                        className="fixed-class-option-color"
                        style={{ background: preset.color }}
                        aria-hidden="true"
                      />
                      <span className="fixed-class-option-name">{preset.name}</span>
                      <span className="fixed-class-option-meta">{preset.image_count} 张</span>
                      <span className="fixed-class-option-status">
                        {isExisting ? "已在类别库" : isSelected ? "待启用" : "未选择"}
                      </span>
                    </label>
                  );
                })}
              </fieldset>
            )}
            <div className="fixed-class-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={() =>
                  setSelectedDetectedClassNames(detectedClasses.map((item) => item.name))
                }
                disabled={
                  !importedDataset
                  || isEnablingDetectedClasses
                  || detectedClasses.length === 0
                  || isLoadingDetectedClasses
                }
              >
                全选
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setSelectedDetectedClassNames([])}
                disabled={
                  !importedDataset
                  || isEnablingDetectedClasses
                  || selectedDetectedClassNames.length === 0
                }
              >
                全不选
              </button>
              <button
                type="button"
                onClick={() => void handleEnableDetectedClasses()}
                disabled={
                  !importedDataset
                  || isEnablingDetectedClasses
                  || selectedDetectedClassNames.length === 0
                  || isLoadingDetectedClasses
                }
              >
                <CheckCircle2 size={16} />
                {isEnablingDetectedClasses ? "正在启用..." : "确认启用所选类别"}
              </button>
            </div>
            </>
            ) : (
              <p className="muted">标注员可查看类别列表，类别启用与编辑由管理员负责。</p>
            )}
          </div>

          {classError ? <div className="error-banner">{classError}</div> : null}

          <div className="class-list" aria-label="可用类别">
            {classes.length === 0 ? (
              <p className="empty-state">请先启用至少一个类别以绘制边界框。</p>
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
                    {isAdmin ? (
                    <>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`编辑 ${classItem.name}`}
                      title={`编辑 ${classItem.name}`}
                      onClick={() => beginEditClass(classItem)}
                    >
                      <Edit3 size={16} />
                    </button>
                    <button
                      type="button"
                      className="icon-button class-delete-button"
                      aria-label={`删除 ${classItem.name}`}
                      title={
                        (classItem.annotation_count ?? 0) > 0 || (classItem.version_count ?? 0) > 0
                          ? `已被 ${classItem.annotation_count ?? 0} 条标注或 ${classItem.version_count ?? 0} 个数据集版本使用，不能删除`
                          : `删除 ${classItem.name}`
                      }
                      disabled={(classItem.annotation_count ?? 0) > 0 || (classItem.version_count ?? 0) > 0}
                      onClick={() => setClassDeleteTarget(classItem)}
                    >
                      <Trash2 size={16} />
                    </button>
                    </>
                    ) : null}
                  </div>
                ),
              )
            )}
          </div>
      </section> : null}

      {classDeleteTarget ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-dialog" role="dialog" aria-modal="true" aria-label="确认删除类别">
            <header>
              <h2>删除类别</h2>
            </header>
            <p>
              确定删除类别“{classDeleteTarget.name}”吗？此操作不可恢复，但不会删除图像或其他类别。
            </p>
            <footer>
              <button
                type="button"
                className="secondary-button"
                disabled={isDeletingClass}
                onClick={() => setClassDeleteTarget(null)}
              >
                取消
              </button>
              <button
                type="button"
                className="danger-button"
                disabled={isDeletingClass}
                onClick={() => void handleDeleteClass()}
              >
                {isDeletingClass ? "正在删除..." : "确认删除"}
              </button>
            </footer>
          </div>
        </div>
      ) : null}

      {currentStep === "annotation" ? (
      <section className="annotation-workspace" aria-label="标注工作台">
        <AnnotationToolbar
          classes={classes}
          selectedClassId={selectedClassId}
          selectedImageIndex={images.findIndex((image) => image.id === selectedImageId)}
          imageCount={images.length}
          annotationsDirty={annotationsDirty}
          isSaving={isSavingAnnotations}
          error={annotationError}
          canGoPrevious={images.findIndex((image) => image.id === selectedImageId) > 0}
          canGoNext={
            images.findIndex((image) => image.id === selectedImageId) >= 0 &&
            images.findIndex((image) => image.id === selectedImageId) < images.length - 1
          }
          showReviewLayers={Boolean(activeReview && activeReview.image.id === selectedImageId)}
          showGroundTruth={showGroundTruthLayer}
          showPrediction={showPredictionLayer}
          onClassChange={setSelectedClassId}
          onPrevious={() => {
            const index = images.findIndex((image) => image.id === selectedImageId);
            const previous = images[index - 1];
            if (previous) {
              requestAnnotationImage(previous.id);
            }
          }}
          onNext={() => {
            const index = images.findIndex((image) => image.id === selectedImageId);
            const next = images[index + 1];
            if (next) {
              requestAnnotationImage(next.id);
            }
          }}
          onSave={() => { void handleSaveAnnotations(); }}
          onSaveAndNext={() => { void handleSaveAndNextImage(); }}
          canCopyPrevious={images.findIndex((image) => image.id === selectedImageId) > 0}
          canCopyNext={
            images.findIndex((image) => image.id === selectedImageId) >= 0 &&
            images.findIndex((image) => image.id === selectedImageId) < images.length - 1
          }
          onCopyPrevious={() => { void copyAdjacentAnnotations("previous"); }}
          onCopyNext={() => { void copyAdjacentAnnotations("next"); }}
          onGroundTruthChange={setShowGroundTruthLayer}
          onPredictionChange={setShowPredictionLayer}
          annotationStatus={annotationStatus}
          filenameSuggestion={filenameSuggestion}
          onApplySuggestion={() => {
            if (filenameSuggestion.selected) {
              setSelectedClassId(filenameSuggestion.selected.id);
            }
          }}
          onConfirmNegative={() => { void handleConfirmNegative(); }}
          canConfirmNegative={annotations.length === 0}
        />

      {pendingAnnotationImageId !== null ? (
        <div role="alert" className="navigation-notice annotation-switch-notice">
          <p>当前图像的标注尚未保存。切换图像后，这些修改将被放弃。</p>
          <div>
            <button type="button" className="secondary-button" onClick={handleStayOnCurrentImage}>
              留在当前图像
            </button>
            <button type="button" onClick={handleDiscardAnnotationsAndSwitch}>
              放弃修改并切换
            </button>
          </div>
        </div>
      ) : null}

      {annotationNotice ? <div className="annotation-notice" role="status">{annotationNotice}</div> : null}

      <div className="annotation-mobile-tabs" role="tablist" aria-label="移动端标注视图">
        {([
          ["canvas", "画布"],
          ["images", "图像"],
          ["properties", "属性"],
        ] as const).map(([pane, label]) => (
          <button
            type="button"
            role="tab"
            key={pane}
            aria-selected={mobileAnnotationPane === pane}
            className={mobileAnnotationPane === pane ? "active" : ""}
            onClick={() => setMobileAnnotationPane(pane)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="annotation-workspace-grid">
        <aside className={`annotation-browser-pane ${mobileAnnotationPane === "images" ? "active" : ""}`} aria-label="图像浏览器">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">数据集图像</p>
              <h2>图像浏览器</h2>
            </div>
            <ImageIcon size={20} />
          </div>

          <div className="image-filter-panel" aria-label="图像筛选器">
            <label htmlFor="filter-filename">文件名搜索</label>
            <input
              id="filter-filename"
              type="search"
              value={imageFilenameSearch}
              disabled={!importedDataset}
              onChange={(event) => setImageFilenameSearch(event.target.value)}
              placeholder="输入文件名"
            />
            <label htmlFor="filter-label-status">标注状态</label>
            <select
              id="filter-label-status"
              value={imageFilters.label_status}
              disabled={!importedDataset}
              onChange={(event) => handleImageLabelStatusChange(event.target.value as ImageFiltersState["label_status"])}
            >
              <option value="all">全部</option>
              <option value="annotated">已标注</option>
              <option value="unannotated">未标注</option>
            </select>
            <button
              type="button"
              className="secondary-button"
              disabled={!importedDataset}
              onClick={() => setIsImageFilterDrawerOpen(true)}
            >
              高级筛选{advancedFilterCount > 0 ? ` (${advancedFilterCount})` : ""}
            </button>
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
            {visibleImages.length === 0 ? (
              <p className="empty-state">导入的图像将显示在这里。</p>
            ) : (
              visibleImages.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  className={image.id === selectedImageId ? "image-row selected" : "image-row"}
                  onClick={() => requestAnnotationImage(image.id)}
                >
                  <strong>{image.relative_path}</strong>
                  <span>
                    {image.platform ?? "未知平台"} | {formatImageAltitude(image.altitude)} |{" "}
                    {image.annotation_count} 个边界框
                  </span>
                  <span className="image-row-status">
                    <b className={`annotation-status-chip ${image.annotation_status ?? (image.annotation_count ? "annotated" : "unreviewed")}`}>
                      {formatAnnotationStatus(image.annotation_status ?? (image.annotation_count ? "annotated" : "unreviewed"))}
                    </b>
                    {suggestClassFromFilename(image.relative_path, classes).selected
                      ? `建议：${suggestClassFromFilename(image.relative_path, classes).selected?.name}`
                      : ""}
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className={`annotation-canvas-pane ${mobileAnnotationPane === "canvas" ? "active" : ""}`} aria-label="标注画布区域">
          <header className="annotation-pane-heading">
            <div className="annotation-pane-heading-main">
              <div>
                <p className="eyebrow">标注与审查</p>
                <h2>标注</h2>
              </div>
              <AnnotationCanvasToolbar
                zoom={canvasViewport.zoom}
                canZoomOut={canvasViewport.zoom > manualMinZoom + 0.001}
                canZoomIn={canvasViewport.zoom < manualMaxZoom - 0.001}
                disabled={!selectedImage || annotationImageSize.width <= 0 || annotationImageSize.height <= 0}
                onZoomOut={() => handleCanvasZoom("out")}
                onZoomIn={() => handleCanvasZoom("in")}
                onActualSize={() => setCanvasZoom(1)}
                onFit={handleCanvasFit}
              />
            </div>
          </header>
          {selectedImage ? (
            <div
              ref={annotationViewportRef}
              aria-label="标注画布视口"
              className={`annotation-canvas-viewport${isSpacePressed ? " is-space-ready" : ""}${canvasPanState ? " is-panning" : ""}`}
              onPointerDown={handleViewportPointerDown}
              onPointerMove={handleViewportPointerMove}
              onPointerUp={handleViewportPointerUp}
              onPointerCancel={handleViewportPointerCancel}
            >
              <div
                className="annotation-canvas-scroll-content"
                data-testid="annotation-canvas-scroll-content"
                style={{
                  width: Math.max(
                    annotationImageSize.width * canvasViewport.zoom,
                    annotationViewportSize.width,
                  ),
                  height: Math.max(
                    annotationImageSize.height * canvasViewport.zoom,
                    annotationViewportSize.height,
                  ),
                }}
              >
                <div
                  className="annotation-transform-layer"
                  data-testid="annotation-transform-layer"
                  style={{
                    width: annotationImageSize.width,
                    height: annotationImageSize.height,
                    "--canvas-pan-x": `${canvasViewport.panX}px`,
                    "--canvas-pan-y": `${canvasViewport.panY}px`,
                    "--canvas-mobile-pan-x": `${Math.max(
                      0,
                      (annotationViewportSize.width - annotationImageSize.width * canvasViewport.zoom) / 2,
                    )}px`,
                    "--canvas-mobile-pan-y": `${Math.max(
                      0,
                      (annotationViewportSize.height - annotationImageSize.height * canvasViewport.zoom) / 2,
                    )}px`,
                    "--canvas-zoom": canvasViewport.zoom,
                  } as CSSProperties}
                >
                  <img
                    src={authedMediaUrl(selectedImage.image_url)}
                    alt={selectedImage.relative_path}
                    width={annotationImageSize.width || undefined}
                    height={annotationImageSize.height || undefined}
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
                            onPointerDown={beginMoveAnnotation}
                            onResizePointerDown={beginResizeAnnotation}
                            zoom={canvasViewport.zoom}
                            imageWidth={annotationImageSize.width}
                            imageHeight={annotationImageSize.height}
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
            </div>
          ) : (
            <p className="empty-state annotation-canvas-empty">
              {annotationGuidance(importedDataset, classes.length, selectedImage, selectedClass)}
            </p>
          )}
        </section>

        <aside className={`annotation-inspector-pane ${mobileAnnotationPane === "properties" ? "active" : ""}`} aria-label="边界框检查器">
          <div className="annotation-pane-heading inspector-heading">
            <div>
              <p className="eyebrow">当前图像</p>
              <h2>边界框</h2>
            </div>
            <span>{annotations.length}</span>
          </div>

          {annotationError ? <div className="error-banner">{annotationError}</div> : null}

          <div className="compact-box-list" aria-label="边界框列表">
            {annotations.length === 0 ? (
              <p className="empty-state">
                {annotationReady
                  ? "在图像上拖动以添加边界框。"
                  : annotationGuidance(importedDataset, classes.length, selectedImage, selectedClass)}
              </p>
            ) : (
              annotations.map((annotation, index) => {
                const className =
                  classById.get(annotation.class_id)?.name ??
                  annotation.class_name ??
                  `类别 ${annotation.class_id}`;
                return (
                  <button
                    type="button"
                    key={annotation.local_id}
                    className={
                      annotation.local_id === selectedAnnotationId
                        ? "compact-box-row selected"
                        : "compact-box-row"
                    }
                    aria-label={`选择边界框 ${index + 1} ${className}`}
                    aria-pressed={annotation.local_id === selectedAnnotationId}
                    onClick={() => setSelectedAnnotationId(annotation.local_id)}
                  >
                    <span style={{ background: resolveClassColor(annotation, classById) }} />
                    <strong>{className}</strong>
                    <small>#{index + 1}</small>
                  </button>
                );
              })
            )}
          </div>

          {selectedAnnotation ? (
            <div className="box-editor selected inspector-editor">
              <div className="box-editor-title">
                <span style={{ background: resolveClassColor(selectedAnnotation, classById) }} />
                <strong>
                  {classById.get(selectedAnnotation.class_id)?.name ??
                    selectedAnnotation.class_name ??
                    `类别 ${selectedAnnotation.class_id}`}
                </strong>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`删除边界框 ${selectedAnnotationIndex + 1}`}
                  onClick={() => deleteAnnotation(selectedAnnotation.local_id)}
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <label className="box-class-control" htmlFor={`box-class-${selectedAnnotation.local_id}`}>
                类别
                <select
                  id={`box-class-${selectedAnnotation.local_id}`}
                  aria-label={`边界框 ${selectedAnnotationIndex + 1} 类别`}
                  value={selectedAnnotation.class_id}
                  onChange={(event) =>
                    updateAnnotationClass(selectedAnnotation.local_id, Number(event.target.value))
                  }
                  disabled={classes.length === 0}
                >
                  {classes.map((classItem) => (
                    <option key={classItem.id} value={classItem.id}>{classItem.name}</option>
                  ))}
                </select>
              </label>

              <div className="nudge-controls" aria-label={`移动边界框 ${selectedAnnotationIndex + 1}`}>
                <button type="button" className="icon-button" aria-label={`向左移动边界框 ${selectedAnnotationIndex + 1}`} title="向左移动" onClick={() => nudgeAnnotation(selectedAnnotation.local_id, -0.01, 0)}><ArrowLeft size={15} /></button>
                <button type="button" className="icon-button" aria-label={`向上移动边界框 ${selectedAnnotationIndex + 1}`} title="向上移动" onClick={() => nudgeAnnotation(selectedAnnotation.local_id, 0, -0.01)}><ArrowUp size={15} /></button>
                <button type="button" className="icon-button" aria-label={`向下移动边界框 ${selectedAnnotationIndex + 1}`} title="向下移动" onClick={() => nudgeAnnotation(selectedAnnotation.local_id, 0, 0.01)}><ArrowDown size={15} /></button>
                <button type="button" className="icon-button" aria-label={`向右移动边界框 ${selectedAnnotationIndex + 1}`} title="向右移动" onClick={() => nudgeAnnotation(selectedAnnotation.local_id, 0.01, 0)}><ArrowRight size={15} /></button>
              </div>

              <button
                type="button"
                className="inspector-disclosure"
                aria-expanded={showAnnotationGeometry}
                onClick={() => setShowAnnotationGeometry((visible) => !visible)}
              >
                {showAnnotationGeometry ? "收起坐标参数" : "展开坐标参数"}
              </button>
              {showAnnotationGeometry ? (
                <div className="geometry-grid" aria-label="归一化坐标">
                  {(["x_center", "y_center", "width", "height"] as const).map((field) => (
                    <label key={field} htmlFor={`box-${field}-${selectedAnnotation.local_id}`}>
                      {field === "x_center" ? "X" : field === "y_center" ? "Y" : field === "width" ? "W" : "H"}
                      <input
                        id={`box-${field}-${selectedAnnotation.local_id}`}
                        type="number"
                        min={field === "width" || field === "height" ? 0.001 : 0}
                        max={1}
                        step={0.001}
                        value={formatGeometryValue(selectedAnnotation[field])}
                        onChange={(event) =>
                          updateAnnotationGeometry(
                            selectedAnnotation.local_id,
                            field,
                            Number(event.target.value),
                          )
                        }
                      />
                    </label>
                  ))}
                </div>
              ) : null}

              <label htmlFor={`track-${selectedAnnotation.local_id}`}>目标轨迹 ID</label>
              <input
                id={`track-${selectedAnnotation.local_id}`}
                value={selectedAnnotation.track_id ?? ""}
                onChange={(event) =>
                  updateAnnotation(selectedAnnotation.local_id, { track_id: event.target.value })
                }
              />

              <label htmlFor={`tags-${selectedAnnotation.local_id}`}>边缘案例标签</label>
              <input
                id={`tags-${selectedAnnotation.local_id}`}
                value={(selectedAnnotation.edge_tags ?? []).join(", ")}
                onChange={(event) =>
                  updateAnnotation(selectedAnnotation.local_id, { edge_tags: parseTags(event.target.value) })
                }
                placeholder="occluded, small"
              />
              <div className="edge-tag-presets" aria-label={`边界框 ${selectedAnnotationIndex + 1} 边缘案例标签预设`}>
                {edgeTagPresets.map((tag) => {
                  const isSelected = (selectedAnnotation.edge_tags ?? []).includes(tag);
                  return (
                    <button
                      type="button"
                      key={tag}
                      className={isSelected ? "tag-chip selected" : "tag-chip"}
                      aria-pressed={isSelected}
                      onClick={() =>
                        updateAnnotation(selectedAnnotation.local_id, {
                          edge_tags: toggleTag(selectedAnnotation.edge_tags, tag),
                        })
                      }
                    >
                      {formatEdgeTag(tag)}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : annotations.length > 0 ? (
            <p className="empty-state inspector-selection-hint">选择一个边界框以编辑属性。</p>
          ) : null}

          {activeReview && selectedImage && activeReview.image.id === selectedImage.id ? (
            <div className="review-banner inspector-review">
              <div className="review-banner-heading">
                <div>
                  <strong>预测结果叠加</strong>
                  <span>
                    匹配正确 {activeReview.counts.matched ?? 0} | 误报 {activeReview.counts.false_positive ?? 0} | 漏报 {activeReview.counts.false_negative ?? 0} | 类别混淆 {activeReview.counts.class_confusion ?? 0}
                  </span>
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
                        <span>类别 #{prediction.class_id} | 置信度 {prediction.confidence.toFixed(2)}</span>
                      </div>
                      {prediction.failure_type === "false_positive" ? (
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => addPredictionAsAnnotation(prediction)}
                          disabled={annotations.some(
                            (annotation) => annotation.local_id === predictionDraftId(prediction.id),
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
        </aside>
      </div>

      {isImageFilterDrawerOpen ? (
        <AnnotationFilterDrawer
          filters={advancedImageFilters}
          classes={classes}
          failureOptions={predictionFailureOptions.map((option) => ({
            value: option.value,
            label: option.value === "all" ? "全部识别结果" : formatFailureType(option.value),
          }))}
          onChange={handleAdvancedImageFilterChange}
          onReset={handleResetAdvancedImageFilters}
          onApply={handleApplyImageFilters}
          onClose={() => setIsImageFilterDrawerOpen(false)}
        />
      ) : null}
      </section>
      ) : null}
      </WorkflowShell>
      ) : null}
      {liveMonitorRunId != null ? (
        <TrainingLiveMonitor
          runId={liveMonitorRunId}
          onClose={() => setLiveMonitorRunId(null)}
        />
      ) : null}
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
  const geometry = serializeClampedGeometry(annotation);
  return {
    class_id: annotation.class_id,
    x_center: geometry.x_center,
    y_center: geometry.y_center,
    width: geometry.width,
    height: geometry.height,
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

/** Round edges first, then rebuild center/size so edge boxes survive 6-digit rounding. */
function serializeClampedGeometry(annotation: Pick<DraftBox, "x_center" | "y_center" | "width" | "height">) {
  let left = roundGeometry(annotation.x_center - annotation.width / 2);
  let right = roundGeometry(annotation.x_center + annotation.width / 2);
  let top = roundGeometry(annotation.y_center - annotation.height / 2);
  let bottom = roundGeometry(annotation.y_center + annotation.height / 2);
  left = clamp(left);
  right = clamp(right);
  top = clamp(top);
  bottom = clamp(bottom);
  if (right <= left) {
    right = Math.min(1, left + 0.001);
  }
  if (bottom <= top) {
    bottom = Math.min(1, top + 0.001);
  }
  const width = roundGeometry(right - left);
  const height = roundGeometry(bottom - top);
  return {
    x_center: roundGeometry(left + width / 2),
    y_center: roundGeometry(top + height / 2),
    width: Math.max(width, 0.001),
    height: Math.max(height, 0.001),
  };
}

function setPointerCaptureSafe(element: Element | null, pointerId: number) {
  if (element && typeof element.setPointerCapture === "function") {
    element.setPointerCapture(pointerId);
  }
}

function releasePointerCaptureSafe(element: Element, pointerId: number) {
  if (
    typeof element.hasPointerCapture === "function" &&
    typeof element.releasePointerCapture === "function" &&
    element.hasPointerCapture(pointerId)
  ) {
    element.releasePointerCapture(pointerId);
  }
}

function isEditableEventTarget(target: EventTarget | null) {
  return target instanceof Element && Boolean(
    target.closest("input, textarea, select, button, [contenteditable]:not([contenteditable='false'])"),
  );
}

function screenPixelsToNormalized(pixels: number, imageDimension: number, zoom: number) {
  if (imageDimension <= 0 || zoom <= 0) {
    return 0.014;
  }
  return pixels / (imageDimension * zoom);
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
  zoom: number;
  imageWidth: number;
  imageHeight: number;
  onPointerDown: (event: PointerEvent<SVGElement>, annotation: DraftBox) => void;
  onResizePointerDown: (
    event: PointerEvent<SVGElement>,
    annotation: DraftBox,
    handle: BoxResizeHandle,
  ) => void;
}) {
  const {
    annotation,
    color,
    selected,
    zoom,
    imageWidth,
    imageHeight,
    onPointerDown,
    onResizePointerDown,
  } = props;
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
  const handleWidth = screenPixelsToNormalized(10, imageWidth, zoom);
  const handleHeight = screenPixelsToNormalized(10, imageHeight, zoom);

  return (
    <g
      className={selected ? "annotation-box selected" : "annotation-box"}
      onPointerDown={(event) => onPointerDown(event, annotation)}
      role="button"
      aria-label={`画布边界框 ${annotation.class_name ?? annotation.class_id}`}
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
        data-testid={`annotation-frame-${annotation.local_id}`}
        x={left}
        y={top}
        width={annotation.width}
        height={annotation.height}
        fill={selected ? "rgba(31, 111, 120, 0.08)" : "transparent"}
        stroke={color}
        strokeWidth={selected ? 3 : 2}
        vectorEffect="non-scaling-stroke"
        onPointerDown={(event) => onPointerDown(event, annotation)}
      />
      {selected
        ? handles.map((handle) => (
            <rect
              key={handle.handle}
              className={`resize-handle ${handle.handle}`}
              x={handle.x - handleWidth / 2}
              y={handle.y - handleHeight / 2}
              width={handleWidth}
              height={handleHeight}
              rx={Math.min(handleWidth, handleHeight) * 0.15}
              fill={color}
              stroke="#ffffff"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
              role="button"
              aria-label={`调整边界框 ${annotation.class_name ?? annotation.class_id} ${formatResizeHandle(
                handle.handle,
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

function PredictionRect(props: {
  prediction: Prediction;
  className?: string;
  highlighted?: boolean;
}) {
  const { prediction, className, highlighted = false } = props;
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
    <g opacity={highlighted ? 1 : 0.85}>
      <title>{label}</title>
      <rect
        x={left}
        y={top}
        width={prediction.width}
        height={prediction.height}
        fill={highlighted ? `${color}22` : "transparent"}
        stroke={color}
        strokeDasharray={prediction.failure_type === "matched" ? "0" : "0.02 0.012"}
        strokeWidth={highlighted ? 0.01 : 0.006}
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

function ReviewAnnotationRect(props: {
  annotation: PredictionReviewAnnotation;
  highlighted?: boolean;
}) {
  const { annotation, highlighted = false } = props;
  const left = annotation.x_center - annotation.width / 2;
  const top = annotation.y_center - annotation.height / 2;
  const color = annotation.class_color || "#38bdf8";
  const label = annotation.class_name || `类别 ${annotation.class_id}`;

  return (
    <g opacity={highlighted ? 1 : 0.9}>
      <title>{label}</title>
      <rect
        x={left}
        y={top}
        width={annotation.width}
        height={annotation.height}
        fill="transparent"
        stroke={color}
        strokeWidth={highlighted ? 0.01 : 0.005}
        vectorEffect="non-scaling-stroke"
      />
      <text
        x={clamp(left)}
        y={clamp(top - 0.018)}
        fill={color}
        fontSize={0.02}
        fontWeight={800}
        paintOrder="stroke"
        stroke="#111820"
        strokeWidth={0.005}
      >
        GT | {label}
      </text>
    </g>
  );
}

function StatusTile(props: {
  icon: ReactNode;
  label: string;
  value: string;
  revealFullValue?: boolean;
}) {
  return (
    <div className="status-tile">
      {props.icon}
      <div className="status-tile-content">
        <span>{props.label}</span>
        <strong title={props.revealFullValue ? props.value : undefined}>{props.value}</strong>
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
        <span>{enabled ? "可用" : reason ?? "等待已完成的训练任务"}</span>
      </div>
      <button
        type="button"
        className="secondary-button"
        disabled={!enabled || isCreating}
        onClick={() => onCreate(format)}
      >
        {isCreating
          ? "正在导出"
          : `导出 ${format === "tensorrt" ? "TensorRT" : format.toUpperCase()}`}
      </button>
    </div>
  );
}

function RunArtifactList(props: { summary?: TrainingRunArtifactSummary }) {
  const { summary } = props;
  const visibleItems = summary?.items.slice(0, 8) ?? [];

  return (
    <div className="run-artifacts" aria-label="训练产物">
      <div className="run-artifacts-heading">
        <strong>训练产物</strong>
        <span>{summary ? `${summary.total_count} 个文件` : "加载中"}</span>
      </div>
      {!summary ? (
        <p className="empty-state">训练产物尚未生成。</p>
      ) : visibleItems.length === 0 ? (
        <p className="empty-state">尚未写入任何文件。</p>
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
            <span className="artifact-more">另有 {summary.total_count - visibleItems.length} 个</span>
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
    <div className="experiment-dashboard" aria-label="实验看板">
      <div className="dashboard-heading">
        <div>
          <strong>实验看板</strong>
          <span>
            {summary?.latest_prediction_job_id
              ? `最新预测任务 #${summary.latest_prediction_job_id}`
              : "等待训练任务数据"}
          </span>
        </div>
      </div>

      {!hasData ? (
        <p className="empty-state">训练指标、类别检测结果和阈值扫描将显示在这里。</p>
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
      <strong>任务对比</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">已完成的训练任务将显示在这里以供对比。</p>
      ) : (
        <div className="compact-table-wrap">
          <table>
            <thead>
              <tr>
                <th>任务</th>
                <th>状态</th>
                <th>模型</th>
                <th>训练轮数</th>
                <th>mAP50</th>
                <th>边界框损失</th>
                <th>匹配正确</th>
                <th>误报</th>
                <th>漏报</th>
                <th>最佳 F1</th>
                <th>最佳置信度</th>
                <th>产物</th>
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row) => (
                <tr key={row.run_id}>
                  <td>#{row.run_id}</td>
                  <td>{formatRunStatus(row.status)}</td>
                  <td title={row.model}>{row.model}</td>
                  <td>{row.epochs ?? "暂无"}</td>
                  <td>{row.map50 === null ? "暂无" : row.map50.toFixed(3)}</td>
                  <td>{row.box_loss === null ? "暂无" : row.box_loss.toFixed(3)}</td>
                  <td>{row.matched}</td>
                  <td>{row.false_positive}</td>
                  <td>{row.false_negative}</td>
                  <td>{row.best_f1 === null ? "暂无" : formatPercent(row.best_f1)}</td>
                  <td>{row.best_threshold === null ? "暂无" : row.best_threshold.toFixed(2)}</td>
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
        <span>{series.latest === null ? "暂无" : series.latest.toFixed(3)}</span>
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
      <strong>类别检测结果</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">请执行预测分析以生成类别检测结果。</p>
      ) : (
        <div className="compact-table-wrap">
          <table>
            <thead>
              <tr>
                <th>类别</th>
                <th>匹配正确</th>
                <th>误报</th>
                <th>漏报</th>
                <th>类别混淆</th>
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
      <strong>混淆矩阵</strong>
      {props.cells.length === 0 ? (
        <p className="empty-state">预测分析结果将用于生成混淆矩阵。</p>
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
      <strong>阈值扫描</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">请使用不同置信度阈值执行预测任务。</p>
      ) : (
        <>
          {props.recommendation ? (
            <div className="threshold-recommendation">
              <span>最佳阈值</span>
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
                  <th>置信度</th>
                  <th>精确率</th>
                  <th>召回率</th>
                  <th>F1</th>
                  <th>匹配正确</th>
                  <th>误报</th>
                  <th>漏报</th>
                  <th>类别混淆</th>
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

function formatResizeHandle(handle: BoxResizeHandle) {
  const labels: Record<BoxResizeHandle, string> = {
    "top-left": "左上角",
    "top-right": "右上角",
    "bottom-left": "左下角",
    "bottom-right": "右下角",
  };
  return labels[handle];
}

function formatScanWarning(message: string) {
  if (message === "No YOLO label .txt files were found") {
    return "未找到 YOLO 标签 .txt 文件";
  }
  if (message === "No data.yaml or dataset.yaml file was found") {
    return "未找到 data.yaml 或 dataset.yaml 文件";
  }
  const parseError = message.match(/^(.+) contains (\d+) unparsable rows$/);
  return parseError
    ? `${parseError[1]} 包含 ${parseError[2]} 行无法解析的数据`
    : message;
}

function formatQualitySummaryIssue(message: string) {
  const exactLabels: Record<string, string> = {
    "Dataset has no images.": "数据集中没有图像。",
    "Project has no active classes.": "项目中没有启用的类别。",
    "Dataset has no saved annotations.": "数据集中没有已保存的标注。",
  };
  if (exactLabels[message]) {
    return exactLabels[message];
  }

  const patterns: Array<[RegExp, (count: string) => string]> = [
    [/^(\d+) images? (?:has|have) no annotations\.$/, (count) => `${count} 张图像没有标注。`],
    [/^(\d+) boxes? (?:has|have) invalid geometry\.$/, (count) => `${count} 个边界框的几何参数无效。`],
    [/^(\d+) boxes? (?:is|are) smaller than 10x10 pixels\.$/, (count) => `${count} 个边界框小于 10x10 像素。`],
    [/^(\d+) boxes? duplicates? another box on the same image and class\.$/, (count) => `${count} 个边界框与同一图像、同一类别中的其他边界框重复。`],
    [/^(\d+) images? (?:is|are) missing platform, altitude, timestamp, or source metadata\.$/, (count) => `${count} 张图像缺少平台、高度、时间戳或来源元数据。`],
    [/^(\d+) images? (?:has|have) unreadable image dimensions\.$/, (count) => `${count} 张图像的尺寸无法读取。`],
    [/^(\d+) label references? unknown class indexes\.$/, (count) => `${count} 行标签引用了未知类别索引。`],
  ];
  for (const [pattern, formatter] of patterns) {
    const match = message.match(pattern);
    if (match) {
      return formatter(match[1]);
    }
  }
  return message;
}

function formatQualityIssueMessage(issue: DatasetQualityIssue) {
  if (issue.issue_type === "unannotated_image") {
    return "图像没有已保存的标注。";
  }
  if (issue.issue_type === "invalid_box") {
    return "边界框超出归一化图像坐标范围。";
  }
  if (issue.issue_type === "tiny_box") {
    return "边界框小于 10x10 像素。";
  }
  if (issue.issue_type === "duplicate_box") {
    const annotationId = issue.message.match(/annotation (\d+)/)?.[1];
    return annotationId
      ? `边界框与同一图像、同一类别中的标注 #${annotationId} 重复。`
      : "边界框与同一图像、同一类别中的其他标注重复。";
  }
  if (issue.issue_type === "missing_metadata") {
    const fields = issue.message.match(/^Image is missing (.+)\.$/)?.[1];
    return fields
      ? `图像缺少${formatMetadataFields(fields)}。`
      : "图像缺少必要元数据。";
  }
  if (issue.issue_type === "missing_image_dimensions") {
    return "无法读取图像宽度或高度。";
  }
  if (issue.issue_type === "unknown_class_reference") {
    const match = issue.message.match(/^(\d+) label rows? references? unknown class index (.+)\.$/);
    return match
      ? `${match[1]} 行标签引用了未知类别索引 ${match[2]}。`
      : "标签引用了未知类别索引。";
  }
  return issue.message;
}

function formatMetadataFields(fields: string) {
  const labels: Record<string, string> = {
    platform: "平台",
    altitude: "高度",
    timestamp: "时间戳",
    "source metadata row": "来源元数据行",
  };
  return fields
    .split(/, | or /)
    .map((field) => labels[field] ?? field)
    .join("、");
}

function formatClassDisplayName(className: string) {
  return className.replace(/^YOLO class (.+)$/, "YOLO 类别 $1");
}

function formatExportReason(reason?: string) {
  if (!reason) {
    return undefined;
  }
  const missingWeights = reason.match(/^Model weights were not found: (.+)$/);
  if (missingWeights) {
    return `未找到模型权重：${missingWeights[1]}`;
  }
  const labels: Record<string, string> = {
    "Ultralytics is not installed": "未安装 Ultralytics",
    "TensorRT Python package is not installed": "未安装 TensorRT Python 包",
  };
  return labels[reason] ?? reason;
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

function formatAnnotationStatus(status: DatasetImage["annotation_status"]) {
  return {
    unreviewed: "待标注",
    annotated: "已标注",
    negative: "已确认无目标",
  }[status];
}

function formatVersionImageScope(scope: DatasetVersion["image_scope"] = "annotated") {
  return scope === "all" ? "全量图像" : "仅已标注图像";
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
