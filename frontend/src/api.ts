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
