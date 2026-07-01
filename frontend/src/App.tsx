import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Database,
  FolderSearch,
  HardDrive,
  Image as ImageIcon,
  Library,
  PackageCheck,
  Play,
  Radar,
  Save,
  Share2,
  Trash2,
  Upload,
} from "lucide-react";
import type { FormEvent, PointerEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type {
  Annotation,
  DatasetImage,
  DatasetImportResponse,
  DatasetQualitySummary,
  DatasetScanSummary,
  DatasetVersion,
  ExportArtifact,
  ExportCapabilities,
  HealthResponse,
  ProjectClass,
  AnnotationWrite,
  ClassOutcome,
  ConfusionCell,
  MetricSeries,
  Prediction,
  PredictionFilters,
  PredictionImageReview,
  PredictionJob,
  RunExperimentSummary,
  TrainingAugmentationConfig,
  TrainingRun,
} from "./api";
import {
  cancelTrainingRun,
  createClass,
  createDatasetVersion,
  createPredictionJob,
  createPredictionThresholdScan,
  createRunExport,
  createTrainingRun,
  getAnnotations,
  getExportCapabilities,
  getHealth,
  getPredictionJobLogs,
  getPredictionImageReview,
  getQuality,
  getTrainingRunLogs,
  getTrainingRunSummary,
  importDataset,
  listClasses,
  listDatasetVersions,
  listImages,
  listPredictionJobs,
  listPredictions,
  listRunExports,
  listTrainingRuns,
  replaceAnnotations,
  scanDataset,
} from "./api";

const defaultDatasetPath = "/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip";
const defaultClassColor = "#ef4444";
const monitorRefreshMs = 2500;
const activeRunStatuses = new Set(["queued", "preparing", "running"]);
const activePredictionStatuses = new Set(["queued", "running"]);
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
    failure_type: "all" as "all" | "matched" | "false_positive" | "false_negative",
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

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
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
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [imageFilters, setImageFilters] = useState({
    platform: "",
    label_status: "all" as "all" | "annotated" | "unannotated",
    class_id: "",
    edge_tag: "",
    altitude_min: "",
    altitude_max: "",
  });
  const [imageFilterError, setImageFilterError] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<DraftBox[]>([]);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [quality, setQuality] = useState<DatasetQualitySummary | null>(null);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [isLoadingQuality, setIsLoadingQuality] = useState(false);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [versionName, setVersionName] = useState("");
  const [versionError, setVersionError] = useState<string | null>(null);
  const [isCreatingVersion, setIsCreatingVersion] = useState(false);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [runLogs, setRunLogs] = useState<Record<number, string>>({});
  const [runSummary, setRunSummary] = useState<RunExperimentSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
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

  useEffect(() => {
    if (!selectedImageId) {
      setAnnotations([]);
      return;
    }
    if (activeReview?.image.id === selectedImageId) {
      return;
    }

    setAnnotationError(null);
    setActiveReview(null);
    getAnnotations(selectedImageId)
      .then((response) => {
        setAnnotations(response.items.map(toDraftBox));
      })
      .catch((error: Error) => setAnnotationError(error.message));
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
      setScanError(error instanceof Error ? error.message : "Dataset scan failed");
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

    try {
      const imported = await importDataset(datasetPath);
      setImportedDataset(imported);

      const [classResponse, imageResponse, qualityResponse, versionResponse] = await Promise.all([
        listClasses(imported.project_id),
        listImages(imported.dataset_id),
        getQuality(imported.dataset_id),
        listDatasetVersions(imported.dataset_id),
      ]);
      const runResponse = await listTrainingRuns(imported.project_id);

      setClasses(classResponse.items);
      setSelectedClassId(classResponse.items[0]?.id ?? null);
      setVersionClassIds(classResponse.items.map((classItem) => classItem.id));
      setImages(imageResponse.items);
      setSelectedImageId(imageResponse.items[0]?.id ?? null);
      setQuality(qualityResponse);
      setVersions(versionResponse.items);
      setRuns(runResponse.items);
      if (runResponse.items[0]) {
        await refreshRunSummary(runResponse.items[0].id);
        await refreshPredictionJobs(runResponse.items[0].id);
      }
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Dataset import failed");
    } finally {
      setIsImporting(false);
    }
  }

  async function refreshTrainingPrep(datasetId = importedDataset?.dataset_id) {
    if (!datasetId) {
      return;
    }

    setIsLoadingQuality(true);
    setQualityError(null);

    try {
      const [qualityResponse, versionResponse] = await Promise.all([
        getQuality(datasetId),
        listDatasetVersions(datasetId),
      ]);
      setQuality(qualityResponse);
      setVersions(versionResponse.items);
      if (importedDataset) {
        const runResponse = await listTrainingRuns(importedDataset.project_id);
        setRuns(runResponse.items);
      }
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "Quality refresh failed");
    } finally {
      setIsLoadingQuality(false);
    }
  }

  async function refreshImages(datasetId = importedDataset?.dataset_id) {
    if (!datasetId) {
      return;
    }

    setImageFilterError(null);
    try {
      const response = await listImages(datasetId, toImageFilterRequest(imageFilters));
      setImages(response.items);
      setSelectedImageId((current) => {
        if (response.items.some((image) => image.id === current)) {
          return current;
        }
        return response.items[0]?.id ?? null;
      });
    } catch (error) {
      setImageFilterError(error instanceof Error ? error.message : "Image filters failed");
    }
  }

  function handleApplyImageFilters() {
    void refreshImages();
  }

  function handleResetImageFilters() {
    const nextFilters = {
      platform: "",
      label_status: "all" as const,
      class_id: "",
      edge_tag: "",
      altitude_min: "",
      altitude_max: "",
    };
    setImageFilters(nextFilters);
    if (importedDataset) {
      setImageFilterError(null);
      listImages(importedDataset.dataset_id)
        .then((response) => {
          setImages(response.items);
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

  async function refreshTrainingRuns(projectId = importedDataset?.project_id) {
    if (!projectId) {
      return;
    }

    try {
      const runResponse = await listTrainingRuns(projectId);
      setRuns(runResponse.items);
      if (runResponse.items[0]) {
        await refreshRunSummary(runResponse.items[0].id);
        await refreshPredictionJobs(runResponse.items[0].id);
      }
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run refresh failed");
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
      setClassError(error instanceof Error ? error.message : "Class creation failed");
    } finally {
      setIsCreatingClass(false);
    }
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
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!dragState) {
      return;
    }

    const point = getRelativePoint(event);
    setDragState((current) =>
      current ? { ...current, currentX: point.x, currentY: point.y } : current,
    );
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
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

    setAnnotations((current) => [
      ...current,
      {
        ...box,
        class_id: selectedClass.id,
        class_name: selectedClass.name,
        class_color: selectedClass.color,
        local_id: `draft-${Date.now()}-${current.length}`,
        track_id: "",
        edge_tags: [],
      },
    ]);
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

  function deleteAnnotation(localId: string) {
    setAnnotations((current) => current.filter((annotation) => annotation.local_id !== localId));
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
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "Copy annotations failed");
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

    try {
      const response = await replaceAnnotations(selectedImageId, annotations.map(toAnnotationWrite));
      setAnnotations(response.items.map(toDraftBox));
      setImages((current) =>
        current.map((image) =>
          image.id === selectedImageId
            ? { ...image, annotation_count: response.items.length }
            : image,
        ),
      );
      void refreshTrainingPrep();
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "Saving annotations failed");
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
      setVersionError(error instanceof Error ? error.message : "Version export failed");
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

    try {
      const run = await createTrainingRun({
        version_id: version.id,
        model: trainingModel.trim() || "yolov8n.pt",
        epochs: trainingEpochs,
        image_size: trainingImageSize,
        batch_size: trainingBatchSize,
        device: trainingDevice.trim() || undefined,
        augmentation_preset: augmentationPreset,
        augmentation,
        tta: trainingTta,
        threshold_scan: thresholdScan,
      });
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

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Application status">
        <div>
          <p className="eyebrow">Local Detection Workbench</p>
          <h1>YOLO Trainer</h1>
        </div>
        <div className="device-pill">
          <Activity size={16} />
          <span>{health?.devices.selected ?? "connecting"}</span>
        </div>
      </section>

      <section className="status-grid" aria-label="Backend details">
        <StatusTile
          icon={<HardDrive size={20} />}
          label="Workspace"
          value={health?.workspace_root ?? "Waiting for backend"}
        />
        <StatusTile
          icon={<Database size={20} />}
          label="Database"
          value={health?.database_path ?? "SQLite will initialize on startup"}
        />
        <StatusTile
          icon={<Activity size={20} />}
          label="Devices"
          value={health ? `${health.devices.available.length} available` : "Detecting"}
        />
      </section>

      {healthError ? <div className="error-banner">{healthError}</div> : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Dataset Intake</p>
            <h2>Scan Local Zip</h2>
          </div>
          <FolderSearch size={22} />
        </div>

        <form className="scan-form" onSubmit={handleScan}>
          <label htmlFor="dataset-path">Dataset zip path</label>
          <div className="input-row">
            <input
              id="dataset-path"
              value={datasetPath}
              onChange={(event) => setDatasetPath(event.target.value)}
            />
            <button type="submit" disabled={isScanning || datasetPath.trim().length === 0}>
              {isScanning ? "Scanning" : "Scan Dataset"}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={isImporting || datasetPath.trim().length === 0}
              onClick={handleImportDataset}
            >
              <Upload size={16} />
              {isImporting ? "Importing" : "Import Dataset"}
            </button>
          </div>
        </form>

        {scanError ? <div className="error-banner">{scanError}</div> : null}
        {importError ? <div className="error-banner">{importError}</div> : null}

        {scan ? (
          <div className="scan-results">
            <div className="metrics-row">
              <Metric label="Images" value={scan.total_images.toLocaleString()} />
              <Metric label="Metadata rows" value={scan.total_metadata_rows.toLocaleString()} />
              <Metric label="YOLO labels" value={scan.total_yolo_labels.toLocaleString()} />
              <Metric label="Class config" value={scan.has_data_yaml ? "Found" : "Missing"} />
            </div>

            <div className="group-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Images</th>
                    <th>Metadata</th>
                    <th>Altitude</th>
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
              Grouped images: {totalGroupImages.toLocaleString()} from {scan.archive_name}
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
            Imported {importedDataset.image_count.toLocaleString()} images into{" "}
            {importedDataset.project_name} / {importedDataset.dataset_name}
          </div>
        ) : null}
      </section>

      <section className="prep-grid" aria-label="Training readiness">
        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Training Prep</p>
              <h2>Quality Review</h2>
            </div>
            {quality?.ready_for_training ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
          </div>

          {quality ? (
            <>
              <div className="readiness-row">
                <strong>{quality.ready_for_training ? "Ready to export" : "Needs attention"}</strong>
                <span>{isLoadingQuality ? "Refreshing" : `${quality.annotation_count} boxes`}</span>
              </div>

              <div className="metrics-row quality-metrics">
                <Metric label="Images" value={quality.image_count.toLocaleString()} />
                <Metric
                  label="Annotated"
                  value={quality.annotated_image_count.toLocaleString()}
                />
                <Metric label="Classes" value={quality.class_count.toLocaleString()} />
                <Metric label="Tiny boxes" value={quality.tiny_box_count.toLocaleString()} />
              </div>

              {quality.issues.length > 0 ? (
                <div className="issue-list">
                  {quality.issues.map((issue) => (
                    <p key={issue}>{issue}</p>
                  ))}
                </div>
              ) : (
                <p className="empty-state">No blocking quality issues detected.</p>
              )}
            </>
          ) : (
            <p className="empty-state">Import a dataset to compute label quality and export readiness.</p>
          )}

          {qualityError ? <div className="error-banner">{qualityError}</div> : null}
        </section>

        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Frozen Dataset</p>
              <h2>Version Export</h2>
            </div>
            <PackageCheck size={20} />
          </div>

          <div className="version-controls">
            <label htmlFor="version-name">Version name</label>
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
                {isCreatingVersion ? "Exporting" : "Create Dataset Version"}
              </button>
            </div>
            <div className="subset-controls" aria-label="Version class subset">
              <span>Class subset</span>
              <div className="subset-grid">
                {classes.length === 0 ? (
                  <p className="empty-state">Create project classes before freezing a subset.</p>
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

          <div className="version-list" aria-label="Dataset versions">
            {versions.length === 0 ? (
              <p className="empty-state">Exported YOLO versions will appear here.</p>
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
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => loadRunLogs(run.id)}
                  >
                    Load Logs
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
          <ExperimentDashboard summary={runSummary} />
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
              <option value="all">All</option>
              <option value="matched">Matched</option>
              <option value="false_positive">False positive</option>
              <option value="false_negative">False negative</option>
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

      <section className="workbench-grid" aria-label="Annotation workbench">
        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Project Labels</p>
              <h2>Class Library</h2>
            </div>
            <Library size={20} />
          </div>

          <form className="class-form" onSubmit={handleCreateClass}>
            <label htmlFor="class-name">Class name</label>
            <input
              id="class-name"
              value={className}
              disabled={!importedDataset}
              onChange={(event) => setClassName(event.target.value)}
              placeholder="target"
            />
            <label htmlFor="class-color">Class color</label>
            <div className="color-row">
              <input
                id="class-color"
                type="color"
                value={classColor}
                disabled={!importedDataset}
                onChange={(event) => setClassColor(event.target.value)}
                aria-label="Class color"
              />
              <button
                type="submit"
                disabled={!importedDataset || isCreatingClass || className.trim().length === 0}
              >
                Create Class
              </button>
            </div>
          </form>

          {classError ? <div className="error-banner">{classError}</div> : null}

          <div className="class-list" aria-label="Available classes">
            {classes.length === 0 ? (
              <p className="empty-state">Import a dataset, then create a class to draw boxes.</p>
            ) : (
              classes.map((classItem) => (
                <button
                  key={classItem.id}
                  type="button"
                  className={
                    classItem.id === selectedClassId ? "class-chip selected" : "class-chip"
                  }
                  onClick={() => setSelectedClassId(classItem.id)}
                >
                  <span style={{ background: classItem.color }} />
                  {classItem.name}
                </button>
              ))
            )}
          </div>
        </aside>

        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Dataset Frames</p>
              <h2>Image Browser</h2>
            </div>
            <ImageIcon size={20} />
          </div>

          <div className="image-filter-panel" aria-label="Image filters">
            <label htmlFor="filter-platform">Platform</label>
            <input
              id="filter-platform"
              value={imageFilters.platform}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({ ...current, platform: event.target.value }))
              }
              placeholder="iris"
            />
            <label htmlFor="filter-label-status">Label status</label>
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
              <option value="all">All</option>
              <option value="annotated">Annotated</option>
              <option value="unannotated">Unannotated</option>
            </select>
            <label htmlFor="filter-class">Class</label>
            <select
              id="filter-class"
              value={imageFilters.class_id}
              disabled={!importedDataset}
              onChange={(event) =>
                setImageFilters((current) => ({ ...current, class_id: event.target.value }))
              }
            >
              <option value="">All classes</option>
              {classes.map((classItem) => (
                <option key={classItem.id} value={classItem.id}>
                  {classItem.name}
                </option>
              ))}
            </select>
            <label htmlFor="filter-edge-tag">Edge tag</label>
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
                Min altitude
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
                Max altitude
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
                Apply Filters
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={!importedDataset}
                onClick={handleResetImageFilters}
              >
                Reset
              </button>
            </div>
          </div>

          {imageFilterError ? <div className="error-banner">{imageFilterError}</div> : null}

          <div className="image-list" aria-label="Imported images">
            {images.length === 0 ? (
              <p className="empty-state">Imported images will appear here.</p>
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
                    {image.platform ?? "unknown"} | {formatImageAltitude(image.altitude)} |{" "}
                    {image.annotation_count} boxes
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="panel annotation-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Draw And Review</p>
              <h2>Annotation</h2>
            </div>
            <Box size={20} />
          </div>

          {selectedImage ? (
            <div className="annotation-layout">
              <div className="viewer-wrap">
                <div className="image-stage">
                  <img src={selectedImage.image_url} alt={selectedImage.relative_path} />
                  <svg
                    aria-label="Annotation canvas"
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
                  <strong>Boxes</strong>
                  <div className="box-list-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => copyAdjacentAnnotations("previous")}
                      disabled={images.findIndex((image) => image.id === selectedImage.id) <= 0}
                    >
                      Copy Previous
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
                      Copy Next
                    </button>
                    <button
                      type="button"
                      onClick={handleSaveAnnotations}
                      disabled={isSavingAnnotations || !selectedImage}
                    >
                      <Save size={16} />
                      {isSavingAnnotations ? "Saving" : "Save Annotations"}
                    </button>
                  </div>
                </div>

                {annotationError ? <div className="error-banner">{annotationError}</div> : null}

                {activeReview && activeReview.image.id === selectedImage.id ? (
                  <div className="review-banner">
                    <div className="review-banner-heading">
                      <div>
                        <strong>Prediction overlay</strong>
                        <span>
                          matched {activeReview.counts.matched ?? 0} | false+{" "}
                          {activeReview.counts.false_positive ?? 0} | false-{" "}
                          {activeReview.counts.false_negative ?? 0}
                        </span>
                      </div>
                      <div className="layer-toggles" aria-label="Annotation review layers">
                        <label>
                          <input
                            type="checkbox"
                            checked={showGroundTruthLayer}
                            onChange={(event) => setShowGroundTruthLayer(event.target.checked)}
                          />
                          GT
                        </label>
                        <label>
                          <input
                            type="checkbox"
                            checked={showPredictionLayer}
                            onChange={(event) => setShowPredictionLayer(event.target.checked)}
                          />
                          Pred
                        </label>
                      </div>
                    </div>
                    <div className="review-legend" aria-label="Prediction legend">
                      <span className="legend matched">Matched</span>
                      <span className="legend false-positive">False +</span>
                      <span className="legend false-negative">False -</span>
                    </div>
                    <div className="review-actions" aria-label="Prediction correction actions">
                      {activeReview.predictions
                        .filter((prediction) => prediction.failure_type !== "matched")
                        .map((prediction) => (
                          <div className="review-action-row" key={prediction.id}>
                            <div>
                              <strong>{formatFailureType(prediction.failure_type)}</strong>
                              <span>
                                Class #{prediction.class_id} | {prediction.confidence.toFixed(2)}
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
                                Add as annotation
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => markFalseNegativeReviewed(prediction)}
                                disabled={!prediction.matched_annotation_id}
                              >
                                Mark reviewed
                              </button>
                            )}
                          </div>
                        ))}
                    </div>
                  </div>
                ) : null}

                {annotations.length === 0 ? (
                  <p className="empty-state">
                    Select a class, then drag over the image to add a bounding box.
                  </p>
                ) : (
                  annotations.map((annotation, index) => (
                    <div className="box-editor" key={annotation.local_id}>
                      <div className="box-editor-title">
                        <span
                          style={{ background: resolveClassColor(annotation, classById) }}
                        />
                        <strong>
                          {annotation.class_name ??
                            classById.get(annotation.class_id)?.name ??
                            `Class ${annotation.class_id}`}
                        </strong>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`Delete box ${index + 1}`}
                          onClick={() => deleteAnnotation(annotation.local_id)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>

                      <div className="geometry-grid" aria-label={`Box ${index + 1} geometry`}>
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

                      <label htmlFor={`track-${annotation.local_id}`}>Track ID</label>
                      <input
                        id={`track-${annotation.local_id}`}
                        value={annotation.track_id ?? ""}
                        onChange={(event) =>
                          updateAnnotation(annotation.local_id, { track_id: event.target.value })
                        }
                      />

                      <label htmlFor={`tags-${annotation.local_id}`}>Edge tags</label>
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
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <p className="empty-state">Import a dataset and select an image to begin annotation.</p>
          )}
        </section>
      </section>
    </main>
  );
}

function formatAltitude(minimum: number | null, maximum: number | null) {
  if (minimum === null || maximum === null) {
    return "Not available";
  }

  return `${minimum.toFixed(1)}-${maximum.toFixed(1)}m`;
}

function formatImageAltitude(altitude: number | null) {
  return altitude === null ? "altitude n/a" : `${altitude.toFixed(1)}m`;
}

function formatGeometryValue(value: number) {
  return Number(value.toFixed(4));
}

function toImageFilterRequest(filters: {
  platform: string;
  label_status: "all" | "annotated" | "unannotated";
  class_id: string;
  edge_tag: string;
  altitude_min: string;
  altitude_max: string;
}) {
  return {
    platform: filters.platform.trim() || undefined,
    label_status: filters.label_status,
    class_id: filters.class_id ? Number(filters.class_id) : undefined,
    edge_tag: filters.edge_tag.trim() || undefined,
    altitude_min: filters.altitude_min === "" ? undefined : Number(filters.altitude_min),
    altitude_max: filters.altitude_max === "" ? undefined : Number(filters.altitude_max),
  };
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
    x_center: annotation.x_center,
    y_center: annotation.y_center,
    width: annotation.width,
    height: annotation.height,
    track_id: annotation.track_id?.trim() ? annotation.track_id.trim() : null,
    edge_tags: annotation.edge_tags ?? [],
  };
}

function getRelativePoint(event: PointerEvent<SVGSVGElement>) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
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

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

function clampDimension(value: number) {
  return Math.min(1, Math.max(0.001, value));
}

function mergeTags(existing: string[] | undefined, tags: string[]) {
  return Array.from(new Set([...(existing ?? []), ...tags]));
}

function formatFailureType(value: string) {
  return value.replace(/_/g, " ");
}

function isActiveRun(status: string) {
  return activeRunStatuses.has(status);
}

function isActivePredictionJob(status: string) {
  return activePredictionStatuses.has(status);
}

function resolveClassColor(annotation: DraftBox, classById: Map<number, ProjectClass>) {
  return annotation.class_color ?? classById.get(annotation.class_id)?.color ?? defaultClassColor;
}

function BoxRect(props: { annotation: DraftBox; color: string }) {
  const { annotation, color } = props;
  return (
    <rect
      x={annotation.x_center - annotation.width / 2}
      y={annotation.y_center - annotation.height / 2}
      width={annotation.width}
      height={annotation.height}
      fill="transparent"
      stroke={color}
      strokeWidth={0.004}
      vectorEffect="non-scaling-stroke"
    />
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
        : "#dc2626";
  const left = prediction.x_center - prediction.width / 2;
  const top = prediction.y_center - prediction.height / 2;
  const label = `${formatFailureType(prediction.failure_type)} | ${
    className ?? `Class ${prediction.class_id}`
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

function ExperimentDashboard(props: { summary: RunExperimentSummary | null }) {
  const { summary } = props;
  const hasData =
    summary &&
    (summary.metric_series.length > 0 ||
      summary.class_outcomes.length > 0 ||
      summary.confusion_matrix.length > 0 ||
      summary.threshold_scan.length > 0);

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
          <div className="curve-grid">
            {summary.metric_series.slice(0, 4).map((series) => (
              <MetricCurve key={series.name} series={series} />
            ))}
          </div>

          <div className="analysis-grid">
            <ClassOutcomeTable rows={summary.class_outcomes} />
            <ConfusionMatrix cells={summary.confusion_matrix} />
          </div>

          <ThresholdScanTable rows={summary.threshold_scan} />
        </>
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
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row) => (
                <tr key={row.class_id}>
                  <td>{row.class_name}</td>
                  <td>{row.matched}</td>
                  <td>{row.false_positive}</td>
                  <td>{row.false_negative}</td>
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

function ThresholdScanTable(props: { rows: RunExperimentSummary["threshold_scan"] }) {
  return (
    <div className="analysis-panel threshold-panel">
      <strong>Threshold Scan</strong>
      {props.rows.length === 0 ? (
        <p className="empty-state">Run prediction jobs at different confidence thresholds.</p>
      ) : (
        <div className="compact-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Conf</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Matched</th>
                <th>False +</th>
                <th>False -</th>
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row) => (
                <tr key={row.job_id}>
                  <td>{row.confidence_threshold.toFixed(2)}</td>
                  <td>{formatPercent(row.precision)}</td>
                  <td>{formatPercent(row.recall)}</td>
                  <td>{row.matched}</td>
                  <td>{row.false_positive}</td>
                  <td>{row.false_negative}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}
