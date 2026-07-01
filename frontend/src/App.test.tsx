import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const apiMock = vi.hoisted(() => {
  const completedRun = {
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
  };
  const runningRun = {
    ...completedRun,
    id: 2,
    status: "running",
    latest_metrics: {},
    started_at: "2026-06-30T00:02:00",
    ended_at: null,
    created_at: "2026-06-30T00:02:00",
    updated_at: "2026-06-30T00:02:30",
  };
  const completedPredictionJob = {
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
  };
  const runningPredictionJob = {
    ...completedPredictionJob,
    id: 2,
    run_id: 2,
    status: "running",
    image_count: 1,
    prediction_count: 0,
    matched_count: 0,
    false_positive_count: 0,
    false_negative_count: 0,
    ended_at: null,
    created_at: "2026-06-30T00:05:00",
    updated_at: "2026-06-30T00:05:30",
  };
  const trainingRunsResponseQueue: Array<{ items: Array<typeof completedRun | typeof runningRun> }> =
    [];
  const predictionJobsResponseQueue: Array<{
    items: Array<typeof completedPredictionJob | typeof runningPredictionJob>;
  }> = [];
  const listTrainingRuns = vi.fn(async () => {
    return trainingRunsResponseQueue.shift() ?? { items: [completedRun] };
  });
  const listPredictionJobs = vi.fn(async () => {
    return predictionJobsResponseQueue.shift() ?? { items: [completedPredictionJob] };
  });
  const listImages = vi.fn(async (_datasetId: number, filters = {}) => ({
    items: [
      {
        id: 10,
        relative_path: "iris/frame001.jpg",
        platform: "iris",
        altitude: 12,
        timestamp: 1,
        annotation_count: filters.label_status === "unannotated" ? 0 : 1,
        image_url: "/api/images/10/file",
      },
      {
        id: 11,
        relative_path: "iris/frame002.jpg",
        platform: "iris",
        altitude: 13,
        timestamp: 2,
        annotation_count: 1,
        image_url: "/api/images/11/file",
      },
    ],
    limit: 50,
    offset: 0,
    total: 2,
  }));
  const createDatasetVersion = vi.fn(async (_datasetId: number, name?: string, classIds?: number[]) => ({
    id: 2,
    project_id: 1,
    dataset_id: 1,
    name: name || "mvp-quality-pass",
    class_mapping: Object.fromEntries((classIds ?? [1]).map((classId, index) => [String(classId), index])),
    split_counts: { train: 1, val: 0, test: 0 },
    artifact_path: "/tmp/workspace/projects/1/versions/2",
    frozen: true,
    created_at: "2026-06-30T00:01:00",
  }));
  const getAnnotations = vi.fn(async (imageId: number) => ({
    items:
      imageId === 11
        ? [
            {
              id: 101,
              image_id: 11,
              class_id: 1,
              class_name: "target",
              class_color: "#ef4444",
              x_center: 0.6,
              y_center: 0.55,
              width: 0.25,
              height: 0.2,
              track_id: "copy-source",
              edge_tags: ["occluded"],
            },
          ]
        : [],
  }));
  const replaceAnnotations = vi.fn(async (
    _imageId: number,
    annotations: Array<{
      class_id: number;
      x_center: number;
      y_center: number;
      width: number;
      height: number;
      track_id?: string | null;
      edge_tags?: string[];
    }>,
  ) => ({
    items: annotations.map((annotation, index: number) => ({
      id: index + 1,
      image_id: _imageId,
      class_name: "target",
      class_color: "#ef4444",
      ...annotation,
    })),
  }));
  const getTrainingRunLogs = vi.fn(async (runId: number) => ({
    run_id: runId,
    text: `training log for run ${runId}\n`,
  }));
  const getPredictionJobLogs = vi.fn(async (jobId: number) => ({
    job_id: jobId,
    text: `prediction log for job ${jobId}\n`,
  }));
  const getTrainingRunSummary = vi.fn(async (runId: number) => ({
    run_id: runId,
    metric_series: [
      {
        name: "metrics/mAP50(B)",
        latest: 0.42,
        points: [
          { epoch: 1, step: null, value: 0.21 },
          { epoch: 2, step: null, value: 0.42 },
        ],
      },
      {
        name: "train/box_loss",
        latest: 1.12,
        points: [
          { epoch: 1, step: null, value: 1.68 },
          { epoch: 2, step: null, value: 1.12 },
        ],
      },
    ],
    class_outcomes: [
      {
        class_id: 1,
        class_name: "target",
        matched: 1,
        false_positive: 1,
        false_negative: 1,
      },
    ],
    confusion_matrix: [
      {
        actual_class_id: 1,
        actual_class_name: "target",
        predicted_class_id: 1,
        predicted_class_name: "target",
        count: 1,
      },
    ],
    threshold_scan: [
      {
        job_id: 1,
        confidence_threshold: 0.25,
        matched: 2,
        false_positive: 1,
        false_negative: 1,
        precision: 2 / 3,
        recall: 2 / 3,
      },
    ],
    latest_prediction_job_id: 1,
  }));
  const getExportCapabilities = vi.fn(async () => ({
    pt_available: true,
    onnx_available: false,
    tensorrt_available: false,
    weights_path: "/tmp/workspace/projects/1/runs/1/ultralytics/weights/best.pt",
    reasons: {
      onnx: "Ultralytics is not installed",
      tensorrt: "TensorRT Python package is not installed",
    },
  }));
  const exportArtifacts: Array<{
    id: number;
    run_id: number;
    project_id: number;
    format: string;
    status: string;
    artifact_path: string;
    error_message: string | null;
    metadata: Record<string, string>;
    started_at: string;
    ended_at: string;
    created_at: string;
    updated_at: string;
  }> = [];
  const listRunExports = vi.fn(async () => ({
    items: exportArtifacts,
  }));
  const createRunExport = vi.fn(async (runId: number, format: string) => {
    const artifact = {
      id: exportArtifacts.length + 1,
      run_id: runId,
      project_id: 1,
      format,
      status: "completed",
      artifact_path: `/tmp/workspace/projects/1/runs/${runId}/exports/run-${runId}.${format}`,
      error_message: null,
      metadata: { format },
      started_at: "2026-06-30T00:06:00",
      ended_at: "2026-06-30T00:06:01",
      created_at: "2026-06-30T00:06:00",
      updated_at: "2026-06-30T00:06:01",
    };
    exportArtifacts.unshift(artifact);
    return artifact;
  });

  return {
    completedRun,
    runningRun,
    completedPredictionJob,
    runningPredictionJob,
    trainingRunsResponseQueue,
    predictionJobsResponseQueue,
    listTrainingRuns,
    listPredictionJobs,
    listImages,
    createDatasetVersion,
    getAnnotations,
    replaceAnnotations,
    getTrainingRunLogs,
    getPredictionJobLogs,
    getTrainingRunSummary,
    getExportCapabilities,
    listRunExports,
    createRunExport,
    exportArtifacts,
  };
});

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
  listImages: apiMock.listImages,
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
  getAnnotations: apiMock.getAnnotations,
  replaceAnnotations: apiMock.replaceAnnotations,
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
  createDatasetVersion: apiMock.createDatasetVersion,
  listTrainingRuns: apiMock.listTrainingRuns,
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
  getTrainingRunLogs: apiMock.getTrainingRunLogs,
  getTrainingRunSummary: apiMock.getTrainingRunSummary,
  listPredictionJobs: apiMock.listPredictionJobs,
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
  getPredictionJobLogs: apiMock.getPredictionJobLogs,
  getPredictionImageReview: async () => ({
    image: {
      id: 10,
      relative_path: "iris/frame001.jpg",
      image_url: "/api/images/10/file",
      platform: "iris",
      altitude: 12,
      timestamp: 1,
    },
    annotations: [
      {
        id: 1,
        image_id: 10,
        class_id: 1,
        class_name: "target",
        class_color: "#ef4444",
        x_center: 0.5,
        y_center: 0.5,
        width: 0.4,
        height: 0.4,
        track_id: null,
        edge_tags: ["occluded"],
      },
    ],
    predictions: [
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
      {
        id: 3,
        run_id: 1,
        job_id: 1,
        image_id: 10,
        class_id: 1,
        x_center: 0.7,
        y_center: 0.7,
        width: 0.12,
        height: 0.12,
        confidence: 0,
        matched_annotation_id: 1,
        failure_type: "false_negative",
      },
    ],
    counts: { matched: 1, false_positive: 1, false_negative: 1 },
  }),
  getExportCapabilities: apiMock.getExportCapabilities,
  listRunExports: apiMock.listRunExports,
  createRunExport: apiMock.createRunExport,
}));

describe("App", () => {
  beforeEach(() => {
    apiMock.trainingRunsResponseQueue.length = 0;
    apiMock.predictionJobsResponseQueue.length = 0;
    apiMock.listImages.mockClear();
    apiMock.createDatasetVersion.mockClear();
    apiMock.getAnnotations.mockClear();
    apiMock.replaceAnnotations.mockClear();
    apiMock.listTrainingRuns.mockClear();
    apiMock.listPredictionJobs.mockClear();
    apiMock.getTrainingRunLogs.mockClear();
    apiMock.getPredictionJobLogs.mockClear();
    apiMock.getTrainingRunSummary.mockClear();
    apiMock.getExportCapabilities.mockClear();
    apiMock.listRunExports.mockClear();
    apiMock.createRunExport.mockClear();
    apiMock.exportArtifacts.length = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

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
    expect(screen.getByLabelText("Image filters")).toBeInTheDocument();
    expect(screen.getByLabelText("Version class subset")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Label status"), "annotated");
    await user.type(screen.getByLabelText("Edge tag"), "occluded");
    await user.click(screen.getByRole("button", { name: "Apply Filters" }));
    expect(apiMock.listImages).toHaveBeenLastCalledWith(1, {
      platform: undefined,
      label_status: "annotated",
      class_id: undefined,
      edge_tag: "occluded",
      altitude_min: undefined,
      altitude_max: undefined,
    });
    expect(await screen.findByRole("button", { name: "Save Annotations" })).toBeInTheDocument();
    expect(await screen.findByText("Ready to export")).toBeInTheDocument();
    expect(await screen.findByText("smoke-export")).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("button", { name: "Create Dataset Version" }));
    expect(apiMock.createDatasetVersion).toHaveBeenCalledWith(1, undefined, [1]);
    expect(await screen.findByText("Run #1")).toBeInTheDocument();
    expect(await screen.findByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    expect(await screen.findByText("Experiment Dashboard")).toBeInTheDocument();
    expect(await screen.findByText("mAP50")).toBeInTheDocument();
    expect(await screen.findByText("box_loss")).toBeInTheDocument();
    expect(await screen.findByText("Class Outcomes")).toBeInTheDocument();
    expect(await screen.findByText("Threshold Scan")).toBeInTheDocument();
    expect(await screen.findAllByText("67%")).toHaveLength(2);
    expect(await screen.findByText("false_positive")).toBeInTheDocument();
    expect((await screen.findAllByText("Matched")).length).toBeGreaterThanOrEqual(2);
    expect(await screen.findByText("Model Export")).toBeInTheDocument();
    expect(await screen.findByText(".pt Weights")).toBeInTheDocument();
    expect(await screen.findByText("Ultralytics is not installed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Export PT" }));

    expect(apiMock.createRunExport).toHaveBeenCalledWith(1, "pt");
    expect(await screen.findByText("PT export #1")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Open Image" })[0]);

    expect(await screen.findByText("Prediction overlay")).toBeInTheDocument();
    expect(screen.getByLabelText("Annotation review layers")).toBeInTheDocument();
    expect(screen.getByLabelText("Prediction legend")).toBeInTheDocument();
    expect(screen.getByText("false positive")).toBeInTheDocument();
    expect(screen.getByText("false negative")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Add as annotation" }));

    expect(screen.getByDisplayValue("false_positive, reviewed_prediction")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Mark reviewed" }));

    expect(
      await screen.findByDisplayValue("occluded, false_negative, reviewed_prediction"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Copy Next" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
    const xInput = screen.getByLabelText("X");
    await user.clear(xInput);
    await user.type(xInput, "0.42");
    await user.click(screen.getByRole("button", { name: "Save Annotations" }));
    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      10,
      expect.arrayContaining([
        expect.objectContaining({
          x_center: 0.42,
          y_center: 0.55,
          width: 0.25,
          height: 0.2,
          track_id: "copy-source",
        }),
      ]),
    );
  });

  it("auto-refreshes active training runs and prediction jobs until idle", async () => {
    apiMock.trainingRunsResponseQueue.push(
      { items: [apiMock.runningRun] },
      { items: [apiMock.completedRun] },
    );
    apiMock.predictionJobsResponseQueue.push(
      { items: [apiMock.runningPredictionJob] },
      { items: [apiMock.completedPredictionJob] },
    );

    render(<App />);
    await flushPromises();
    vi.useFakeTimers();

    fireEvent.click(screen.getByRole("button", { name: "Import Dataset" }));
    await flushPromises();

    expect(screen.getByText("Run #2")).toBeInTheDocument();
    expect(screen.getAllByText("Auto refresh on").length).toBeGreaterThanOrEqual(2);

    await act(async () => {
      vi.advanceTimersByTime(2500);
      await Promise.resolve();
    });
    await flushPromises();

    expect(screen.getByText("Run #1")).toBeInTheDocument();
    expect(screen.getByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    expect(apiMock.getTrainingRunLogs).toHaveBeenCalledWith(2);
    expect(apiMock.getPredictionJobLogs).toHaveBeenCalledWith(2);
    expect(screen.getAllByText("Idle")).toHaveLength(2);
  });
});

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}
