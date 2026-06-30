import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  getHealth: async () => ({
    status: "ok",
    app: "YOLO Trainer",
    workspace_root: "/tmp/workspace",
    database_path: "/tmp/workspace/app.db",
    devices: { selected: "cpu", available: ["cpu"], details: {} },
  }),
  scanDataset: async () => ({
    source_path: "/tmp/image_dataset.zip",
    archive_name: "image_dataset.zip",
    total_files: 1102,
    total_images: 1099,
    total_yolo_labels: 0,
    total_yaml_files: 0,
    total_json_files: 2,
    total_metadata_rows: 1099,
    has_yolo_labels: false,
    has_data_yaml: false,
    groups: [
      {
        name: "iris",
        image_count: 433,
        metadata_rows: 433,
        altitude_min: 12,
        altitude_max: 13,
        first_timestamp: 1,
        last_timestamp: 2,
      },
      {
        name: "vtol",
        image_count: 666,
        metadata_rows: 666,
        altitude_min: 40,
        altitude_max: 42,
        first_timestamp: 3,
        last_timestamp: 4,
      },
    ],
    warnings: ["No YOLO label .txt files were found"],
  }),
  importDataset: async () => ({
    project_id: 1,
    dataset_id: 1,
    project_name: "YOLO Trainer Project",
    dataset_name: "image_dataset",
    image_count: 1,
    groups: [{ name: "iris", image_count: 1, metadata_rows: 1 }],
  }),
  listImages: async () => ({
    items: [
      {
        id: 10,
        relative_path: "iris/frame001.jpg",
        platform: "iris",
        altitude: 12,
        timestamp: 1,
        annotation_count: 0,
        image_url: "/api/images/10/file",
      },
    ],
    limit: 50,
    offset: 0,
    total: 1,
  }),
  listClasses: async () => ({
    items: [
      {
        id: 1,
        project_id: 1,
        name: "target",
        color: "#ef4444",
        description: null,
        active: true,
      },
    ],
  }),
  createClass: async () => ({
    id: 2,
    project_id: 1,
    name: "vehicle",
    color: "#22c55e",
    description: null,
    active: true,
  }),
  getAnnotations: async () => ({ items: [] }),
  replaceAnnotations: async () => ({ items: [] }),
  getQuality: async () => ({
    dataset_id: 1,
    image_count: 1,
    annotated_image_count: 1,
    unannotated_image_count: 0,
    annotation_count: 1,
    class_count: 1,
    tiny_box_count: 0,
    invalid_box_count: 0,
    ready_for_training: true,
    issues: [],
  }),
  listDatasetVersions: async () => ({
    items: [
      {
        id: 1,
        project_id: 1,
        dataset_id: 1,
        name: "smoke-export",
        class_mapping: { "1": 0 },
        split_counts: { train: 1, val: 0, test: 0 },
        artifact_path: "/tmp/workspace/projects/1/versions/1",
        frozen: true,
        created_at: "2026-06-30T00:00:00",
      },
    ],
  }),
  createDatasetVersion: async () => ({
    id: 2,
    project_id: 1,
    dataset_id: 1,
    name: "mvp-quality-pass",
    class_mapping: { "1": 0 },
    split_counts: { train: 1, val: 0, test: 0 },
    artifact_path: "/tmp/workspace/projects/1/versions/2",
    frozen: true,
    created_at: "2026-06-30T00:01:00",
  }),
  listTrainingRuns: async () => ({
    items: [
      {
        id: 1,
        project_id: 1,
        version_id: 1,
        status: "completed",
        device: "cpu",
        config: { model: "yolov8n.pt", epochs: 2 },
        artifact_path: "/tmp/workspace/projects/1/runs/1",
        log_path: "/tmp/workspace/projects/1/runs/1/logs.txt",
        error_message: null,
        latest_metrics: { "metrics/mAP50(B)": 0.42 },
        started_at: "2026-06-30T00:00:00",
        ended_at: "2026-06-30T00:01:00",
        created_at: "2026-06-30T00:00:00",
        updated_at: "2026-06-30T00:01:00",
      },
    ],
  }),
  createTrainingRun: async () => ({
    id: 2,
    project_id: 1,
    version_id: 1,
    status: "queued",
    device: "cpu",
    config: { model: "yolov8n.pt", epochs: 50 },
    artifact_path: "/tmp/workspace/projects/1/runs/2",
    log_path: "/tmp/workspace/projects/1/runs/2/logs.txt",
    error_message: null,
    latest_metrics: {},
    started_at: null,
    ended_at: null,
    created_at: "2026-06-30T00:02:00",
    updated_at: "2026-06-30T00:02:00",
  }),
  getTrainingRunLogs: async () => ({
    run_id: 1,
    text: "training queued\ntraining completed\n",
  }),
  listPredictionJobs: async () => ({
    items: [
      {
        id: 1,
        run_id: 1,
        project_id: 1,
        status: "completed",
        image_scope: "all",
        confidence_threshold: 0.25,
        artifact_path: "/tmp/workspace/projects/1/runs/1/predictions/1",
        log_path: "/tmp/workspace/projects/1/runs/1/predictions/1/logs.txt",
        image_count: 2,
        prediction_count: 3,
        matched_count: 1,
        false_positive_count: 1,
        false_negative_count: 1,
        error_message: null,
        started_at: "2026-06-30T00:03:00",
        ended_at: "2026-06-30T00:04:00",
        created_at: "2026-06-30T00:03:00",
        updated_at: "2026-06-30T00:04:00",
      },
    ],
  }),
  createPredictionJob: async () => ({
    id: 2,
    run_id: 1,
    project_id: 1,
    status: "failed",
    image_scope: "all",
    confidence_threshold: 0.25,
    artifact_path: "/tmp/workspace/projects/1/runs/1/predictions/2",
    log_path: "/tmp/workspace/projects/1/runs/1/predictions/2/logs.txt",
    image_count: 0,
    prediction_count: 0,
    matched_count: 0,
    false_positive_count: 0,
    false_negative_count: 0,
    error_message: "Model weights were not found",
    started_at: "2026-06-30T00:05:00",
    ended_at: "2026-06-30T00:05:01",
    created_at: "2026-06-30T00:05:00",
    updated_at: "2026-06-30T00:05:01",
  }),
  listPredictions: async () => ({
    items: [
      {
        id: 1,
        run_id: 1,
        job_id: 1,
        image_id: 10,
        class_id: 1,
        x_center: 0.5,
        y_center: 0.5,
        width: 0.4,
        height: 0.4,
        confidence: 0.91,
        matched_annotation_id: 1,
        failure_type: "matched",
      },
      {
        id: 2,
        run_id: 1,
        job_id: 1,
        image_id: 10,
        class_id: 1,
        x_center: 0.1,
        y_center: 0.1,
        width: 0.1,
        height: 0.1,
        confidence: 0.77,
        matched_annotation_id: null,
        failure_type: "false_positive",
      },
    ],
  }),
  getPredictionJobLogs: async () => ({
    job_id: 1,
    text: "prediction queued\nprediction completed\n",
  }),
}));

describe("App", () => {
  it("renders local app status and dataset scan controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
    expect(await screen.findByText("cpu")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset zip path")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Scan Dataset" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Import Dataset" })).toBeInTheDocument();
    expect(screen.getByText("Class Library")).toBeInTheDocument();
    expect(screen.getByText("Image Browser")).toBeInTheDocument();
    expect(screen.getByText("Annotation")).toBeInTheDocument();
    expect(screen.getByText("Quality Review")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Dataset Version" })).toBeInTheDocument();
    expect(screen.getByText("Training Setup")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start Training Run" })).toBeInTheDocument();
    expect(screen.getByText("Run History")).toBeInTheDocument();
    expect(screen.getByText("Prediction Analysis")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Import Dataset" }));

    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Save Annotations" })).toBeInTheDocument();
    expect(await screen.findByText("Ready to export")).toBeInTheDocument();
    expect(await screen.findByText("smoke-export")).toBeInTheDocument();
    expect(await screen.findByText("Run #1")).toBeInTheDocument();
    expect(await screen.findByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    expect(await screen.findByText("false_positive")).toBeInTheDocument();
    expect(await screen.findByText("Matched")).toBeInTheDocument();
  });
});
