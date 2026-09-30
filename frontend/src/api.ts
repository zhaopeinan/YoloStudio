export type DeviceInfo = {
  selected: "cuda" | "mps" | "cpu";
  available: string[];
  details: Record<string, string>;
};

export type UserRole = "admin" | "annotator";

export type AuthUser = {
  id: number;
  username: string;
  role: UserRole;
  is_active: boolean;
  created_at?: string | null;
  updated_at?: string | null;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};

export type UserListResponse = {
  items: AuthUser[];
};

let authToken: string | null = null;
let onUnauthorized: (() => void) | null = null;

export function setAuthToken(token: string | null): void {
  authToken = token;
}

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

/** Attach JWT for <img>/<video> tags that cannot send Authorization headers. */
export function authedMediaUrl(url: string): string {
  if (!authToken || !url.startsWith("/api/")) {
    return url;
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}access_token=${encodeURIComponent(authToken)}`;
}

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

export type DatasetSourceOption = {
  source_ref: string;
  label: string;
  kind: "server" | "upload";
  source_path: string;
  original_filename: string;
  size_bytes: number;
  source_id: number | null;
};

export type DatasetSource = {
  id: number;
  original_filename: string;
  size_bytes: number;
  source_path: string;
  created_at: string;
  updated_at: string;
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
  annotated_image_count?: number;
  annotation_count?: number;
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
  annotation_status: "unreviewed" | "annotated" | "negative";
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
  annotation_status: "unreviewed" | "annotated" | "negative";
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
  image_scope?: "annotated" | "all";
  split_counts: {
    train: number;
    val: number;
    test: number;
  };
  source_dataset_ids?: number[];
  merged?: boolean;
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

export type GpuProcessInfo = {
  pid: number | null;
  name: string;
  memory_mb: number | null;
};

export type GpuDeviceInfo = {
  index: number;
  name: string;
  utilization_gpu: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
  processes: GpuProcessInfo[];
};

export type GpuStatus = {
  available: boolean;
  gpus: GpuDeviceInfo[];
  error: string | null;
  queried_at: string;
};

export type TrainingLiveProgress = {
  epoch: number | null;
  total_epochs: number | null;
  percent: number | null;
  elapsed_sec: number | null;
  eta_sec: number | null;
  phase: string;
};

export type TrainingLiveSnapshot = {
  run_id: number;
  project_id: number;
  status: string;
  status_label: string;
  device: string;
  config: Record<string, unknown>;
  progress: TrainingLiveProgress;
  latest: Record<string, number>;
  series: Record<string, Array<number | null>>;
  log_tail: string[];
  error_message: string | null;
  started_at: string | null;
  ended_at: string | null;
  updated_at: string;
  gpu: GpuStatus | null;
};

export type TrainingModelOption = {
  model_ref: string;
  label: string;
  kind: "base" | "trained" | "upload";
  run_id: number | null;
  weight_id: number | null;
  status: string;
};

export type ModelWeight = {
  id: number;
  project_id: number;
  original_filename: string;
  size_bytes: number;
  created_at: string;
  updated_at: string;
};

export type ModelWeightListResponse = {
  items: ModelWeight[];
};

export type PreviewModelOption = {
  model_ref: string;
  label: string;
  kind: "base" | "trained";
  run_id: number | null;
  status: string | null;
};

export type PreviewBox = {
  class_id: number;
  class_name: string;
  color: string;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  confidence: number | null;
};

export type ImagePreviewResponse = {
  image_id: number;
  filename: string;
  image_url: string;
  width: number;
  height: number;
  model_ref: string;
  confidence_threshold: number;
  annotations: PreviewBox[];
  predictions: PreviewBox[];
};

export type PreviewJob = {
  id: number;
  project_id: number;
  dataset_id: number | null;
  run_id: number | null;
  video_id: number | null;
  model_ref: string;
  kind: string;
  source_filename: string;
  status: string;
  confidence_threshold: number;
  frame_step: number;
  fps: number | null;
  total_frames: number;
  processed_frames: number;
  error_message: string | null;
  result_url: string | null;
  stream_url: string | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
};

export type PreviewJobListResponse = {
  items: PreviewJob[];
};

export type PreviewVideo = {
  id: number;
  project_id: number;
  original_filename: string;
  size_bytes: number;
  created_at: string;
};

export type PreviewVideoListResponse = {
  items: PreviewVideo[];
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
  version_id: number | null;
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
  download_url: string | null;
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
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }
  const response = await fetch(url, {
    ...init,
    headers,
  });

  if (response.status === 401 && onUnauthorized) {
    onUnauthorized();
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(apiErrorMessage(text, response.status));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function login(username: string, password: string): Promise<LoginResponse> {
  return requestJson<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function getMe(): Promise<AuthUser> {
  return requestJson<AuthUser>("/api/auth/me");
}

export function listUsers(): Promise<UserListResponse> {
  return requestJson<UserListResponse>("/api/users");
}

export function createUser(payload: {
  username: string;
  password: string;
  role: UserRole;
  is_active?: boolean;
}): Promise<AuthUser> {
  return requestJson<AuthUser>("/api/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateUser(
  userId: number,
  payload: { password?: string; role?: UserRole; is_active?: boolean },
): Promise<AuthUser> {
  return requestJson<AuthUser>(`/api/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteUser(userId: number): Promise<void> {
  return requestJson<void>(`/api/users/${userId}`, { method: "DELETE" });
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function listProjects(): Promise<ProjectListResponse> {
  return requestJson<ProjectListResponse>("/api/projects");
}

export type DetectedClassSuggestion = {
  name: string;
  image_count: number;
  color: string;
  sample_filenames: string[];
};

export type DetectedClassListResponse = {
  dataset_id: number;
  total_images: number;
  method: string;
  items: DetectedClassSuggestion[];
};

export function listDetectedClasses(datasetId: number): Promise<DetectedClassListResponse> {
  return requestJson<DetectedClassListResponse>(`/api/datasets/${datasetId}/detected-classes`);
}

export function scanDataset(sourcePath: string): Promise<DatasetScanSummary> {
  return requestJson<DatasetScanSummary>("/api/datasets/scan", {
    method: "POST",
    body: JSON.stringify({ source_path: sourcePath }),
  });
}

export function listDatasetSources(): Promise<{ items: DatasetSourceOption[] }> {
  return requestJson<{ items: DatasetSourceOption[] }>("/api/datasets/sources");
}

export function uploadDatasetSource(body: FormData): Promise<DatasetSource> {
  return requestJson<DatasetSource>("/api/datasets/sources", {
    method: "POST",
    body,
  });
}

export function deleteDatasetSource(sourceId: number): Promise<void> {
  return requestJson<void>(`/api/datasets/sources/${sourceId}`, {
    method: "DELETE",
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

/** Fetch every image in a dataset by paging through the list API. */
export async function listAllImages(
  datasetId: number,
  filters: DatasetImageFilters = {},
  pageSize = 500,
): Promise<DatasetImage[]> {
  const first = await listImages(datasetId, filters, { limit: pageSize, offset: 0 });
  const items = [...first.items];
  while (items.length < first.total) {
    const page = await listImages(datasetId, filters, {
      limit: pageSize,
      offset: items.length,
    });
    if (page.items.length === 0) {
      break;
    }
    items.push(...page.items);
  }
  return items;
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
  annotationStatus?: "unreviewed" | "annotated" | "negative",
): Promise<AnnotationListResponse> {
  return requestJson<AnnotationListResponse>(`/api/images/${imageId}/annotations`, {
    method: "PUT",
    body: JSON.stringify({ annotations, annotation_status: annotationStatus }),
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
  imageScope: "annotated" | "all" = "annotated",
  datasetIds?: number[],
): Promise<DatasetVersion> {
  return requestJson<DatasetVersion>(`/api/datasets/${datasetId}/versions`, {
    method: "POST",
    body: JSON.stringify({
      name,
      class_ids: classIds && classIds.length > 0 ? classIds : undefined,
      image_scope: imageScope,
      dataset_ids: datasetIds && datasetIds.length > 0 ? datasetIds : undefined,
    }),
  });
}

export function listDatasetVersions(datasetId: number): Promise<DatasetVersionListResponse> {
  return requestJson<DatasetVersionListResponse>(`/api/datasets/${datasetId}/versions`);
}

export function listProjectVersions(projectId: number): Promise<DatasetVersionListResponse> {
  return requestJson<DatasetVersionListResponse>(`/api/projects/${projectId}/versions`);
}

export function listTrainingRuns(projectId: number): Promise<TrainingRunListResponse> {
  return requestJson<TrainingRunListResponse>(`/api/projects/${projectId}/training/runs`);
}

export function listTrainingModels(projectId: number): Promise<{ items: TrainingModelOption[] }> {
  return requestJson<{ items: TrainingModelOption[] }>(
    `/api/training/models?project_id=${projectId}`,
  );
}

export function listModelWeights(projectId: number): Promise<ModelWeightListResponse> {
  return requestJson<ModelWeightListResponse>(`/api/training/weights?project_id=${projectId}`);
}

export function uploadModelWeight(body: FormData): Promise<ModelWeight> {
  return requestJson<ModelWeight>("/api/training/weights", {
    method: "POST",
    body,
  });
}

export function deleteModelWeight(weightId: number): Promise<void> {
  return requestJson<void>(`/api/training/weights/${weightId}`, {
    method: "DELETE",
  });
}

export function listPreviewModels(projectId?: number): Promise<{ items: PreviewModelOption[] }> {
  const query = projectId ? `?project_id=${projectId}` : "";
  return requestJson<{ items: PreviewModelOption[] }>(`/api/preview/models${query}`);
}

export function previewImage(
  imageId: number,
  body: { model_ref: string; confidence_threshold: number },
): Promise<ImagePreviewResponse> {
  return requestJson<ImagePreviewResponse>(`/api/preview/images/${imageId}`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function createVideoPreview(body: FormData): Promise<PreviewJob> {
  return requestJson<PreviewJob>("/api/preview/video-jobs", {
    method: "POST",
    body,
  });
}

export function listPreviewVideos(projectId: number): Promise<PreviewVideoListResponse> {
  return requestJson<PreviewVideoListResponse>(`/api/preview/videos?project_id=${projectId}`);
}

export function uploadPreviewVideo(body: FormData): Promise<PreviewVideo> {
  return requestJson<PreviewVideo>("/api/preview/videos", {
    method: "POST",
    body,
  });
}

export function deletePreviewVideo(videoId: number): Promise<void> {
  return requestJson<void>(`/api/preview/videos/${videoId}`, {
    method: "DELETE",
  });
}

export function listVideoPreviewJobs(projectId: number): Promise<PreviewJobListResponse> {
  return requestJson<PreviewJobListResponse>(`/api/preview/video-jobs?project_id=${projectId}`);
}

export function getVideoPreviewStatus(previewId: number): Promise<PreviewJob> {
  return requestJson<PreviewJob>(`/api/preview/video-jobs/${previewId}`);
}

export function getVideoPreviewStreamUrl(previewId: number): string {
  return `/api/preview/video-jobs/${previewId}/stream`;
}

export function getVideoPreviewResultUrl(previewId: number): string {
  return `/api/preview/video-jobs/${previewId}/result`;
}

export function deleteVideoPreview(previewId: number): Promise<void> {
  return requestJson<void>(`/api/preview/video-jobs/${previewId}`, {
    method: "DELETE",
  });
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

export function getTrainingRunLive(
  runId: number,
  includeGpu = true,
): Promise<TrainingLiveSnapshot> {
  const query = includeGpu ? "" : "?include_gpu=false";
  return requestJson<TrainingLiveSnapshot>(`/api/training/runs/${runId}/live${query}`);
}

export function getSystemGpu(): Promise<GpuStatus> {
  return requestJson<GpuStatus>("/api/system/gpu");
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
  body: {
    image_scope: string;
    confidence_threshold: number;
    version_id?: number;
    image_filters?: DatasetImageFilters;
  },
): Promise<PredictionJob> {
  return requestJson<PredictionJob>(`/api/training/runs/${runId}/prediction-jobs`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function createPredictionThresholdScan(
  runId: number,
  body: {
    image_scope: string;
    thresholds: number[];
    version_id?: number;
    image_filters?: DatasetImageFilters;
  },
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

export function getRunBestWeightsUrl(runId: number): string {
  return `/api/training/runs/${runId}/weights/best`;
}

export function getExportDownloadUrl(runId: number, exportId: number): string {
  return `/api/training/runs/${runId}/exports/${exportId}/file`;
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

export function cascadePurgeTrashItem(
  trashId: number,
  confirmName: string,
): Promise<StorageMutationResponse> {
  return requestJson<StorageMutationResponse>(
    `/api/storage/trash/${trashId}/cascade-purge`,
    {
      method: "POST",
      body: JSON.stringify({ confirm_name: confirmName }),
    },
  );
}

export function purgeExpiredTrash(): Promise<StoragePurgeSummary> {
  return requestJson<StoragePurgeSummary>("/api/storage/trash/purge-expired", {
    method: "POST",
  });
}
