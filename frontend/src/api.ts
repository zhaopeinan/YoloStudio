export type DeviceInfo = {
  selected: "cuda" | "mps" | "cpu";
  available: string[];
  details: Record<string, string>;
};

export type HealthResponse = {
  status: string;
  app: string;
  workspace_root: string;
  database_path: string;
  devices: DeviceInfo;
};

export type DatasetGroupSummary = {
  name: string;
  image_count: number;
  metadata_rows: number;
  altitude_min: number | null;
  altitude_max: number | null;
  first_timestamp: number | null;
  last_timestamp: number | null;
};

export type DatasetScanSummary = {
  source_path: string;
  archive_name: string;
  total_files: number;
  total_images: number;
  total_yolo_labels: number;
  total_yaml_files: number;
  total_json_files: number;
  total_metadata_rows: number;
  has_yolo_labels: boolean;
  has_data_yaml: boolean;
  groups: DatasetGroupSummary[];
  warnings: string[];
};

export type DatasetImportGroup = {
  name: string;
  image_count: number;
  metadata_rows: number;
};

export type DatasetImportResponse = {
  project_id: number;
  dataset_id: number;
  project_name: string;
  dataset_name: string;
  image_count: number;
  groups: DatasetImportGroup[];
};

export type ProjectDatasetSummary = {
  id: number;
  project_id: number;
  name: string;
  source_type: string;
  import_status: string;
  image_count: number;
};

export type ProjectSummary = {
  id: number;
  name: string;
  datasets: ProjectDatasetSummary[];
};

export type ProjectListResponse = {
  items: ProjectSummary[];
};

export type DatasetImage = {
  id: number;
  relative_path: string;
  width: number | null;
  height: number | null;
  platform: string | null;
  altitude: number | null;
  timestamp: number | null;
  annotation_count: number;
  image_url: string;
};

export type DatasetImageListResponse = {
  items: DatasetImage[];
  limit: number;
  offset: number;
  total: number;
};

export type DatasetDimensionRefreshSummary = {
  dataset_id: number;
  scanned_count: number;
  updated_count: number;
  missing_count: number;
};

export type CoverageBucket = {
  label: string;
  image_count: number;
  annotated_image_count: number;
  annotation_count: number;
};

export type ClassCoverageBucket = {
  class_id: number;
  class_name: string;
  class_color: string;
  image_count: number;
  annotation_count: number;
};

export type EdgeTagCoverageBucket = {
  tag: string;
  image_count: number;
  annotation_count: number;
};

export type DatasetCoverageSummary = {
  dataset_id: number;
  image_count: number;
  annotated_image_count: number;
  annotation_count: number;
  platforms: CoverageBucket[];
  altitude_bands: CoverageBucket[];
  classes: ClassCoverageBucket[];
  edge_tags: EdgeTagCoverageBucket[];
};

export type PredictionFailureType =
  | "all"
  | "matched"
  | "false_positive"
  | "false_negative"
  | "class_confusion";

export type DatasetImageFilters = {
  platform?: string;
  label_status?: "all" | "annotated" | "unannotated";
  class_id?: number;
  edge_tag?: string;
  failure_type?: PredictionFailureType;
  altitude_min?: number;
  altitude_max?: number;
};

export type DatasetImageListOptions = {
  limit?: number;
  offset?: number;
};

export type ProjectClass = {
  id: number;
  project_id: number;
  name: string;
  color: string;
  description: string | null;
  active: boolean;
  annotation_count: number;
  version_count: number;
};

export type ClassListResponse = {
  items: ProjectClass[];
};

export type Annotation = {
  id?: number;
  image_id?: number;
  class_id: number;
  class_name?: string;
  class_color?: string;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  track_id?: string | null;
  edge_tags?: string[];
};

export type AnnotationWrite = {
  class_id: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  track_id?: string | null;
  edge_tags?: string[];
};

export type AnnotationListResponse = {
  items: Annotation[];
};

export type DatasetQualitySummary = {
  dataset_id: number;
  image_count: number;
  annotated_image_count: number;
  unannotated_image_count: number;
  annotation_count: number;
  class_count: number;
  tiny_box_count: number;
  invalid_box_count: number;
  duplicate_box_count: number;
  missing_metadata_count: number;
  missing_image_dimensions_count: number;
  unknown_class_reference_count: number;
  ready_for_training: boolean;
  issues: string[];
};

export type DatasetQualityIssue = {
  issue_type: Exclude<DatasetQualityIssueType, "all">;
  severity: "warning" | "error";
  message: string;
  image_id: number;
  image_path: string;
  image_url: string;
  annotation_id: number | null;
  class_id: number | null;
  class_name: string | null;
  x_center: number | null;
  y_center: number | null;
  width: number | null;
  height: number | null;
};

export type DatasetQualityIssueType =
  | "all"
  | "unannotated_image"
  | "tiny_box"
  | "invalid_box"
  | "duplicate_box"
  | "missing_metadata"
  | "missing_image_dimensions"
  | "unknown_class_reference";

export type DatasetQualityIssueListResponse = {
  dataset_id: number;
  limit: number;
  offset: number;
  total: number;
  items: DatasetQualityIssue[];
};

export type QualityTagApplySummary = {
  dataset_id: number;
  issue_type: DatasetQualityIssueType;
  scanned_issue_count: number;
  updated_annotation_count: number;
  applied_tag_count: number;
};

export type DatasetVersion = {
  id: number;
  project_id: number;
  dataset_id: number;
  name: string;
  class_mapping: Record<string, number>;
  split_counts: {
    train: number;
    val: number;
    test: number;
  };
  artifact_path: string;
  frozen: boolean;
  created_at: string;
};

export type DatasetVersionListResponse = {
  items: DatasetVersion[];
};

export type TrainingRun = {
  id: number;
  project_id: number;
  version_id: number;
  status: string;
  device: string;
  config: Record<string, unknown>;
  artifact_path: string;
  log_path: string;
  error_message: string | null;
  latest_metrics: Record<string, number>;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
};

export type TrainingRunListResponse = {
  items: TrainingRun[];
};

export type TrainingRunLogsResponse = {
  run_id: number;
  text: string;
};

export type TrainingRunArtifact = {
  relative_path: string;
  category: string;
  size_bytes: number;
};

export type TrainingRunArtifactSummary = {
  run_id: number;
  artifact_root: string;
  total_count: number;
  items: TrainingRunArtifact[];
};

export type MetricPoint = {
  epoch: number | null;
  step: number | null;
  value: number;
};

export type MetricSeries = {
  name: string;
  points: MetricPoint[];
  latest: number | null;
};

export type ClassOutcome = {
  class_id: number;
  class_name: string;
  matched: number;
  false_positive: number;
  false_negative: number;
  class_confusion: number;
};

export type ConfusionCell = {
  actual_class_id: number;
  actual_class_name: string;
  predicted_class_id: number;
  predicted_class_name: string;
  count: number;
};

export type ThresholdPoint = {
  job_id: number;
  confidence_threshold: number;
  matched: number;
  false_positive: number;
  false_negative: number;
  class_confusion: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ThresholdRecommendation = {
  job_id: number;
  confidence_threshold: number;
  precision: number;
  recall: number;
  f1: number;
};

export type RunExperimentSummary = {
  run_id: number;
  metric_series: MetricSeries[];
  class_outcomes: ClassOutcome[];
  confusion_matrix: ConfusionCell[];
  threshold_scan: ThresholdPoint[];
  threshold_recommendation: ThresholdRecommendation | null;
  latest_prediction_job_id: number | null;
};

export type RunComparisonRow = {
  run_id: number;
  status: string;
  model: string;
  epochs: number | null;
  device: string;
  artifact_path: string;
  map50: number | null;
  box_loss: number | null;
  latest_prediction_job_id: number | null;
  matched: number;
  false_positive: number;
  false_negative: number;
  class_confusion: number;
  best_threshold: number | null;
  best_f1: number | null;
};

export type ProjectExperimentSummary = {
  project_id: number;
  runs: RunComparisonRow[];
};

export type TrainingAugmentationConfig = {
  mosaic: number;
  mixup: number;
  copy_paste: number;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  translate: number;
  scale: number;
  fliplr: number;
  erasing: number;
  gridmask: boolean;
};

export type TrainingRunCreate = {
  version_id: number;
  model: string;
  epochs: number;
  image_size: number;
  batch_size: number;
  device?: string;
  augmentation_preset: string;
  augmentation: TrainingAugmentationConfig;
  tta: boolean;
  threshold_scan: boolean;
};

export type PredictionJob = {
  id: number;
  run_id: number;
  project_id: number;
  status: string;
  image_scope: string;
  confidence_threshold: number;
  image_filters: DatasetImageFilters | null;
  artifact_path: string;
  log_path: string;
  image_count: number;
  prediction_count: number;
  matched_count: number;
  false_positive_count: number;
  false_negative_count: number;
  class_confusion_count: number;
  error_message: string | null;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
};

export type PredictionJobListResponse = {
  items: PredictionJob[];
};

export type PredictionThresholdScanResponse = {
  items: PredictionJob[];
};

export type PredictionFilters = {
  failure_type?: PredictionFailureType;
  class_id?: number;
  confidence_min?: number;
  confidence_max?: number;
  platform?: string;
  altitude_min?: number;
  altitude_max?: number;
  timestamp_min?: number;
  timestamp_max?: number;
};

export type Prediction = {
  id: number;
  run_id: number;
  job_id: number;
  image_id: number;
  class_id: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  confidence: number;
  matched_annotation_id: number | null;
  failure_type: string;
};

export type PredictionListResponse = {
  items: Prediction[];
};

export type PredictionJobLogsResponse = {
  job_id: number;
  text: string;
};

export type PredictionReviewImage = {
  id: number;
  relative_path: string;
  image_url: string;
  platform: string | null;
  altitude: number | null;
  timestamp: number | null;
};

export type PredictionReviewAnnotation = Annotation & {
  id: number;
  image_id: number;
  class_name: string;
  class_color: string;
};

export type PredictionImageReview = {
  image: PredictionReviewImage;
  annotations: PredictionReviewAnnotation[];
  predictions: Prediction[];
  counts: Record<string, number>;
};

export type ExportArtifact = {
  id: number;
  run_id: number;
  project_id: number;
  format: string;
  status: string;
  artifact_path: string;
  error_message: string | null;
  metadata: Record<string, unknown>;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
};

export type ExportArtifactListResponse = {
  items: ExportArtifact[];
};

export type ExportCapabilities = {
  pt_available: boolean;
  onnx_available: boolean;
  tensorrt_available: boolean;
  weights_path: string | null;
  reasons: Record<string, string>;
};

export type StorageEntityType = "dataset" | "dataset_version" | "training_run";

export type StorageBlockerEntityType =
  | StorageEntityType
  | "prediction_job"
  | "export_artifact";

export type StorageBlocker = {
  entity_type: StorageBlockerEntityType;
  entity_id: number;
  display_name: string;
  status: string | null;
};

export type StorageSplitCounts = {
  train: number;
  val: number;
  test: number;
};

export type StorageItem = {
  entity_type: "dataset" | "dataset_version";
  entity_id: number;
  display_name: string;
  project_id: number;
  project_name: string;
  dataset_id: number | null;
  version_id: number | null;
  artifact_path: string;
  size_bytes: number;
  image_count: number | null;
  annotation_count: number | null;
  split_counts: StorageSplitCounts | null;
  created_at: string;
  protected: boolean;
  blockers: StorageBlocker[];
};

export type StorageItemListResponse = {
  items: StorageItem[];
  total_size_bytes: number;
};

export type StorageRelatedRun = {
  id: number;
  status: string;
  model: string;
  size_bytes: number;
  prediction_job_count: number;
  export_count: number;
  created_at: string;
};

export type StorageItemDetail = StorageItem & {
  class_names: string[];
  related_runs: StorageRelatedRun[];
};

export type TrashItemSummary = {
  name?: string;
  image_count?: number;
  annotation_count?: number;
  split_counts?: StorageSplitCounts;
  class_names?: string[];
  status?: string;
  model?: string;
  prediction_job_count?: number;
  export_count?: number;
};

export type TrashItem = {
  id: number;
  entity_type: StorageEntityType;
  entity_id: number;
  display_name: string;
  project_id: number;
  dataset_id: number | null;
  version_id: number | null;
  original_path: string;
  trash_path: string;
  size_bytes: number;
  summary: TrashItemSummary;
  status: string;
  error_message: string | null;
  deleted_at: string;
  purge_after: string;
};

export type TrashItemListResponse = {
  items: TrashItem[];
  total_size_bytes: number;
};

export type StorageMutationResponse = {
  trash_id: number;
  entity_type: StorageEntityType;
  entity_id: number;
  status: string;
  message: string;
};

export type StoragePurgeSummary = {
  purged_count: number;
  failed_count: number;
};

function apiErrorMessage(text: string, status: number): string {
  if (text) {
    try {
      const payload = JSON.parse(text) as {
        detail?: string | { message?: string };
        message?: string;
      };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        return payload.detail;
      }
      if (
        payload.detail &&
        typeof payload.detail === "object" &&
        typeof payload.detail.message === "string" &&
        payload.detail.message.trim()
      ) {
        return payload.detail.message;
      }
      if (typeof payload.message === "string" && payload.message.trim()) {
        return payload.message;
      }
    } catch {
      return text;
    }
    return text;
  }
  return `请求失败（${status}）`;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(apiErrorMessage(text, response.status));
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function listProjects(): Promise<ProjectListResponse> {
  return requestJson<ProjectListResponse>("/api/projects");
}

export function scanDataset(sourcePath: string): Promise<DatasetScanSummary> {
  return requestJson<DatasetScanSummary>("/api/datasets/scan", {
    method: "POST",
    body: JSON.stringify({ source_path: sourcePath }),
  });
}

export function importDataset(
  sourcePath: string,
  projectName = "YOLO Trainer Project",
  datasetName = "image_dataset",
): Promise<DatasetImportResponse> {
  return requestJson<DatasetImportResponse>("/api/datasets/import", {
    method: "POST",
    body: JSON.stringify({
      source_path: sourcePath,
      project_name: projectName,
      dataset_name: datasetName,
    }),
  });
}

export function listImages(
  datasetId: number,
  filters: DatasetImageFilters = {},
  options: DatasetImageListOptions = {},
): Promise<DatasetImageListResponse> {
  const params = new URLSearchParams({
    limit: String(options.limit ?? 50),
    offset: String(options.offset ?? 0),
  });
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== "" && value !== "all") {
      params.set(key, String(value));
    }
  });
  return requestJson<DatasetImageListResponse>(`/api/datasets/${datasetId}/images?${params}`);
}

export function refreshImageDimensions(
  datasetId: number,
): Promise<DatasetDimensionRefreshSummary> {
  return requestJson<DatasetDimensionRefreshSummary>(
    `/api/datasets/${datasetId}/refresh-image-dimensions`,
    { method: "POST" },
  );
}

export function getDatasetCoverage(datasetId: number): Promise<DatasetCoverageSummary> {
  return requestJson<DatasetCoverageSummary>(`/api/datasets/${datasetId}/coverage`);
}

export function listClasses(projectId: number): Promise<ClassListResponse> {
  return requestJson<ClassListResponse>(`/api/projects/${projectId}/classes`);
}

export function createClass(
  projectId: number,
  body: { name: string; color?: string; description?: string },
): Promise<ProjectClass> {
  return requestJson<ProjectClass>(`/api/projects/${projectId}/classes`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function updateClass(
  projectId: number,
  classId: number,
  body: { name?: string; color?: string; description?: string | null },
): Promise<ProjectClass> {
  return requestJson<ProjectClass>(`/api/projects/${projectId}/classes/${classId}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

export function deleteClass(projectId: number, classId: number): Promise<ProjectClass> {
  return requestJson<ProjectClass>(`/api/projects/${projectId}/classes/${classId}`, {
    method: "DELETE",
  });
}

export function getAnnotations(imageId: number): Promise<AnnotationListResponse> {
  return requestJson<AnnotationListResponse>(`/api/images/${imageId}/annotations`);
}

export function replaceAnnotations(
  imageId: number,
  annotations: AnnotationWrite[],
): Promise<AnnotationListResponse> {
  return requestJson<AnnotationListResponse>(`/api/images/${imageId}/annotations`, {
    method: "PUT",
    body: JSON.stringify({ annotations }),
  });
}

export function getQuality(datasetId: number): Promise<DatasetQualitySummary> {
  return requestJson<DatasetQualitySummary>(`/api/datasets/${datasetId}/quality`);
}

export function listQualityIssues(
  datasetId: number,
  issueType: DatasetQualityIssueType = "all",
): Promise<DatasetQualityIssueListResponse> {
  const params = new URLSearchParams({ issue_type: issueType });
  return requestJson<DatasetQualityIssueListResponse>(
    `/api/datasets/${datasetId}/quality/issues?${params}`,
  );
}

export function applyQualityTags(
  datasetId: number,
  issueType: DatasetQualityIssueType = "all",
): Promise<QualityTagApplySummary> {
  return requestJson<QualityTagApplySummary>(`/api/datasets/${datasetId}/quality/apply-tags`, {
    method: "POST",
    body: JSON.stringify({ issue_type: issueType }),
  });
}

export function createDatasetVersion(
  datasetId: number,
  name?: string,
  classIds?: number[],
): Promise<DatasetVersion> {
  return requestJson<DatasetVersion>(`/api/datasets/${datasetId}/versions`, {
    method: "POST",
    body: JSON.stringify({ name, class_ids: classIds && classIds.length > 0 ? classIds : undefined }),
  });
}

export function listDatasetVersions(datasetId: number): Promise<DatasetVersionListResponse> {
  return requestJson<DatasetVersionListResponse>(`/api/datasets/${datasetId}/versions`);
}

export function listTrainingRuns(projectId: number): Promise<TrainingRunListResponse> {
  return requestJson<TrainingRunListResponse>(`/api/projects/${projectId}/training/runs`);
}

export function createTrainingRun(body: TrainingRunCreate): Promise<TrainingRun> {
  return requestJson<TrainingRun>("/api/training/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function cancelTrainingRun(runId: number): Promise<TrainingRun> {
  return requestJson<TrainingRun>(`/api/training/runs/${runId}/cancel`, {
    method: "POST",
  });
}

export function getTrainingRunLogs(runId: number): Promise<TrainingRunLogsResponse> {
  return requestJson<TrainingRunLogsResponse>(`/api/training/runs/${runId}/logs`);
}

export function getTrainingRunArtifacts(runId: number): Promise<TrainingRunArtifactSummary> {
  return requestJson<TrainingRunArtifactSummary>(`/api/training/runs/${runId}/artifacts`);
}

export function getTrainingRunSummary(runId: number): Promise<RunExperimentSummary> {
  return requestJson<RunExperimentSummary>(`/api/training/runs/${runId}/summary`);
}

export function getProjectTrainingSummary(projectId: number): Promise<ProjectExperimentSummary> {
  return requestJson<ProjectExperimentSummary>(`/api/projects/${projectId}/training/summary`);
}

export function createPredictionJob(
  runId: number,
  body: { image_scope: string; confidence_threshold: number; image_filters?: DatasetImageFilters },
): Promise<PredictionJob> {
  return requestJson<PredictionJob>(`/api/training/runs/${runId}/prediction-jobs`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function createPredictionThresholdScan(
  runId: number,
  body: { image_scope: string; thresholds: number[]; image_filters?: DatasetImageFilters },
): Promise<PredictionThresholdScanResponse> {
  return requestJson<PredictionThresholdScanResponse>(
    `/api/training/runs/${runId}/prediction-threshold-scan`,
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

export function listPredictionJobs(runId: number): Promise<PredictionJobListResponse> {
  return requestJson<PredictionJobListResponse>(`/api/training/runs/${runId}/prediction-jobs`);
}

export function listPredictions(
  jobId: number,
  filters: PredictionFilters = {},
): Promise<PredictionListResponse> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== "" && value !== "all") {
      params.set(key, String(value));
    }
  });
  const query = params.toString();
  return requestJson<PredictionListResponse>(
    `/api/prediction-jobs/${jobId}/predictions${query ? `?${query}` : ""}`,
  );
}

export function getPredictionJobLogs(jobId: number): Promise<PredictionJobLogsResponse> {
  return requestJson<PredictionJobLogsResponse>(`/api/prediction-jobs/${jobId}/logs`);
}

export function getPredictionImageReview(
  jobId: number,
  imageId: number,
): Promise<PredictionImageReview> {
  return requestJson<PredictionImageReview>(
    `/api/prediction-jobs/${jobId}/images/${imageId}/review`,
  );
}

export function getExportCapabilities(runId: number): Promise<ExportCapabilities> {
  return requestJson<ExportCapabilities>(`/api/training/runs/${runId}/exports/capabilities`);
}

export function listRunExports(runId: number): Promise<ExportArtifactListResponse> {
  return requestJson<ExportArtifactListResponse>(`/api/training/runs/${runId}/exports`);
}

export function createRunExport(runId: number, format: string): Promise<ExportArtifact> {
  return requestJson<ExportArtifact>(`/api/training/runs/${runId}/exports`, {
    method: "POST",
    body: JSON.stringify({ format }),
  });
}

export function listStorageItems(): Promise<StorageItemListResponse> {
  return requestJson<StorageItemListResponse>("/api/storage/items");
}

export function getStorageItem(
  entityType: StorageItem["entity_type"],
  entityId: number,
): Promise<StorageItemDetail> {
  return requestJson<StorageItemDetail>(`/api/storage/items/${entityType}/${entityId}`);
}

export function trashStorageItem(
  entityType: StorageEntityType,
  entityId: number,
): Promise<TrashItem> {
  return requestJson<TrashItem>(`/api/storage/items/${entityType}/${entityId}/trash`, {
    method: "POST",
  });
}

export function listTrashItems(): Promise<TrashItemListResponse> {
  return requestJson<TrashItemListResponse>("/api/storage/trash");
}

export function restoreTrashItem(trashId: number): Promise<StorageMutationResponse> {
  return requestJson<StorageMutationResponse>(`/api/storage/trash/${trashId}/restore`, {
    method: "POST",
  });
}

export function purgeTrashItem(
  trashId: number,
  confirmName: string,
): Promise<StorageMutationResponse> {
  return requestJson<StorageMutationResponse>(`/api/storage/trash/${trashId}`, {
    method: "DELETE",
    body: JSON.stringify({ confirm_name: confirmName }),
  });
}

export function purgeExpiredTrash(): Promise<StoragePurgeSummary> {
  return requestJson<StoragePurgeSummary>("/api/storage/trash/purge-expired", {
    method: "POST",
  });
}
