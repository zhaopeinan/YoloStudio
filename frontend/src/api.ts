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

export type DatasetImage = {
  id: number;
  relative_path: string;
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

export type ProjectClass = {
  id: number;
  project_id: number;
  name: string;
  color: string;
  description: string | null;
  active: boolean;
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
  ready_for_training: boolean;
  issues: string[];
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

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
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

export function listImages(datasetId: number): Promise<DatasetImageListResponse> {
  return requestJson<DatasetImageListResponse>(
    `/api/datasets/${datasetId}/images?limit=50&offset=0`,
  );
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

export function createDatasetVersion(
  datasetId: number,
  name?: string,
): Promise<DatasetVersion> {
  return requestJson<DatasetVersion>(`/api/datasets/${datasetId}/versions`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function listDatasetVersions(datasetId: number): Promise<DatasetVersionListResponse> {
  return requestJson<DatasetVersionListResponse>(`/api/datasets/${datasetId}/versions`);
}
