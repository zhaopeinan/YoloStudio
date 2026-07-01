import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { DatasetQualityIssueListResponse, DatasetQualitySummary } from "./api";

function pointerEvent(type: string, clientX: number, clientY: number) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    clientX: { value: clientX },
    clientY: { value: clientY },
    pointerId: { value: 1 },
  });
  return event;
}

const apiMock = vi.hoisted(() => {
  const completedRun = {
    id: 1,
    project_id: 1,
    version_id: 1,
    status: "completed",
    device: "cpu",
    config: {
      model: "custom-drone.pt",
      epochs: 12,
      image_size: 512,
      batch_size: 4,
      device: "mps",
      augmentation_preset: "edge-case",
      augmentation: {
        mosaic: 0.8,
        mixup: 0.15,
        copy_paste: 0.25,
        hsv_h: 0.02,
        hsv_s: 0.6,
        hsv_v: 0.5,
        translate: 0.08,
        scale: 0.7,
        fliplr: 0.4,
        erasing: 0.2,
        gridmask: true,
      },
      tta: true,
      threshold_scan: true,
    },
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
  const cancelledRun = {
    ...runningRun,
    status: "cancelled",
    ended_at: "2026-06-30T00:03:00",
    updated_at: "2026-06-30T00:03:00",
  };
  const completedPredictionJob = {
    id: 1,
    run_id: 1,
    project_id: 1,
    status: "completed",
    image_scope: "all",
    confidence_threshold: 0.25,
    image_filters: null,
    artifact_path: "/tmp/workspace/projects/1/runs/1/predictions/1",
    log_path: "/tmp/workspace/projects/1/runs/1/predictions/1/logs.txt",
    image_count: 2,
    prediction_count: 3,
    matched_count: 1,
    false_positive_count: 1,
    false_negative_count: 1,
    class_confusion_count: 1,
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
    class_confusion_count: 0,
    ended_at: null,
    created_at: "2026-06-30T00:05:00",
    updated_at: "2026-06-30T00:05:30",
  };
  const trainingRunsResponseQueue: Array<{
    items: Array<typeof completedRun | typeof runningRun | typeof cancelledRun>;
  }> = [];
  const predictionJobsResponseQueue: Array<{
    items: Array<typeof completedPredictionJob | typeof runningPredictionJob>;
  }> = [];
  const listTrainingRuns = vi.fn(async () => {
    return trainingRunsResponseQueue.shift() ?? { items: [completedRun] };
  });
  const listPredictionJobs = vi.fn(async () => {
    return predictionJobsResponseQueue.shift() ?? { items: [completedPredictionJob] };
  });
  const importDataset = vi.fn(async () => ({
    project_id: 1,
    dataset_id: 1,
    project_name: "Drone QA Project",
    dataset_name: "camouflage-set",
    image_count: 1,
    groups: [{ name: "iris", image_count: 1, metadata_rows: 1 }],
  }));
  const listProjects = vi.fn(async () => ({
    items: [
      {
        id: 1,
        name: "Drone QA Project",
        datasets: [
          {
            id: 1,
            project_id: 1,
            name: "camouflage-set",
            source_type: "zip",
            import_status: "imported",
            image_count: 2,
          },
        ],
      },
    ],
  }));
  const datasetImages = [
    {
      id: 10,
      relative_path: "iris/frame001.jpg",
      width: 640,
      height: 480,
      platform: "iris",
      altitude: 12,
      timestamp: 1,
      annotation_count: 1,
      image_url: "/api/images/10/file",
    },
    {
      id: 11,
      relative_path: "iris/frame002.jpg",
      width: 640,
      height: 480,
      platform: "iris",
      altitude: 13,
      timestamp: 2,
      annotation_count: 1,
      image_url: "/api/images/11/file",
    },
    {
      id: 12,
      relative_path: "iris/frame051.jpg",
      width: 640,
      height: 480,
      platform: "iris",
      altitude: 14,
      timestamp: 3,
      annotation_count: 0,
      image_url: "/api/images/12/file",
    },
  ];
  const listImages = vi.fn(
    async (
      _datasetId: number,
      filters: {
        label_status?: "all" | "annotated" | "unannotated";
        failure_type?: "all" | "matched" | "false_positive" | "false_negative" | "class_confusion";
      } = {},
      options: { limit?: number; offset?: number } = {},
    ) => {
      const offset = options.offset ?? 0;
      const limit = options.limit ?? 50;
      const items =
        offset >= 50
          ? [datasetImages[2]]
          : datasetImages.slice(0, 2).map((image, index) => ({
              ...image,
              annotation_count: filters.label_status === "unannotated" && index === 0 ? 0 : 1,
            }));
      return {
        items,
        limit,
        offset,
        total: 51,
      };
    },
  );
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
  const createTrainingRun = vi.fn(async (body: { version_id: number }) => ({
    id: 2,
    project_id: 1,
    version_id: body.version_id,
    status: "queued",
    device: "cpu",
    config: body,
    artifact_path: "/tmp/workspace/projects/1/runs/2",
    log_path: "/tmp/workspace/projects/1/runs/2/logs.txt",
    error_message: null,
    latest_metrics: {},
    started_at: null,
    ended_at: null,
    created_at: "2026-06-30T00:02:00",
    updated_at: "2026-06-30T00:02:00",
  }));
  const cancelTrainingRun = vi.fn(async (runId: number) => ({
    ...cancelledRun,
    id: runId,
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
  const createPredictionThresholdScan = vi.fn(
    async (
      _runId: number,
      body: {
        image_scope: string;
        thresholds: number[];
        image_filters?: {
          platform?: string;
          label_status?: "all" | "annotated" | "unannotated";
          class_id?: number;
          edge_tag?: string;
          failure_type?: "all" | "matched" | "false_positive" | "false_negative" | "class_confusion";
          altitude_min?: number;
          altitude_max?: number;
        };
      },
    ) => ({
      items: body.thresholds.map((threshold, index) => ({
        ...completedPredictionJob,
        id: 10 + index,
        confidence_threshold: threshold,
        image_filters: body.image_filters ?? null,
        artifact_path: `/tmp/workspace/projects/1/runs/1/predictions/${10 + index}`,
        log_path: `/tmp/workspace/projects/1/runs/1/predictions/${10 + index}/logs.txt`,
        created_at: `2026-06-30T00:0${index}:00`,
        updated_at: `2026-06-30T00:0${index}:30`,
      })),
    }),
  );
  const createPredictionJob = vi.fn(
    async (
      _runId: number,
      body: {
        image_scope: string;
        confidence_threshold: number;
        image_filters?: {
          platform?: string;
          label_status?: "all" | "annotated" | "unannotated";
          class_id?: number;
          edge_tag?: string;
          failure_type?: "all" | "matched" | "false_positive" | "false_negative" | "class_confusion";
          altitude_min?: number;
          altitude_max?: number;
        };
      },
    ) => ({
      id: 2,
      run_id: 1,
      project_id: 1,
      status: "failed",
      image_scope: body.image_scope,
      confidence_threshold: body.confidence_threshold,
      image_filters: body.image_filters ?? null,
      artifact_path: "/tmp/workspace/projects/1/runs/1/predictions/2",
      log_path: "/tmp/workspace/projects/1/runs/1/predictions/2/logs.txt",
      image_count: 0,
      prediction_count: 0,
      matched_count: 0,
      false_positive_count: 0,
      false_negative_count: 0,
      class_confusion_count: 0,
      error_message: "Model weights were not found",
      started_at: "2026-06-30T00:05:00",
      ended_at: "2026-06-30T00:05:01",
      created_at: "2026-06-30T00:05:00",
      updated_at: "2026-06-30T00:05:01",
    }),
  );
  const listPredictions = vi.fn(async () => ({
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
      {
        id: 4,
        run_id: 1,
        job_id: 1,
        image_id: 11,
        class_id: 2,
        x_center: 0.58,
        y_center: 0.5,
        width: 0.3,
        height: 0.3,
        confidence: 0.68,
        matched_annotation_id: 2,
        failure_type: "class_confusion",
      },
    ],
  }));
  const defaultQuality = (): DatasetQualitySummary => ({
    dataset_id: 1,
    image_count: 1,
    annotated_image_count: 1,
    unannotated_image_count: 0,
    annotation_count: 1,
    class_count: 1,
    tiny_box_count: 0,
    invalid_box_count: 0,
    duplicate_box_count: 0,
    missing_metadata_count: 0,
    missing_image_dimensions_count: 0,
    unknown_class_reference_count: 0,
    ready_for_training: true,
    issues: [],
  });
  const defaultQualityIssues = (): DatasetQualityIssueListResponse => ({
    dataset_id: 1,
    limit: 50,
    offset: 0,
    total: 1,
    items: [
      {
        issue_type: "tiny_box",
        severity: "warning",
        message: "Box is smaller than 10x10 pixels.",
        image_id: 11,
        image_path: "iris/frame002.jpg",
        image_url: "/api/images/11/file",
        annotation_id: 101,
        class_id: 1,
        class_name: "target",
        x_center: 0.6,
        y_center: 0.55,
        width: 0.25,
        height: 0.2,
      },
    ],
  });
  const defaultClasses = () => [
    {
      id: 1,
      project_id: 1,
      name: "target",
      color: "#ef4444",
      description: null,
      active: true,
    },
  ];
  const listClasses = vi.fn(async () => ({
    items: defaultClasses(),
  }));
  const createClass = vi.fn(async () => ({
    id: 2,
    project_id: 1,
    name: "vehicle",
    color: "#22c55e",
    description: null,
    active: true,
  }));
  const updateClass = vi.fn(
    async (
      _projectId: number,
      classId: number,
      body: { name?: string; color?: string; description?: string | null },
    ) => ({
      id: classId,
      project_id: 1,
      name: body.name ?? "target",
      color: body.color ?? "#ef4444",
      description: body.description ?? null,
      active: true,
    }),
  );
  const getQuality = vi.fn(async () => defaultQuality());
  const listQualityIssues = vi.fn(async () => defaultQualityIssues());
  const applyQualityTags = vi.fn(async () => ({
    dataset_id: 1,
    issue_type: "duplicate_box",
    scanned_issue_count: 1,
    updated_annotation_count: 1,
    applied_tag_count: 1,
  }));
  const refreshImageDimensions = vi.fn(async () => ({
    dataset_id: 1,
    scanned_count: 2,
    updated_count: 1,
    missing_count: 0,
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
        class_confusion: 1,
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
        class_confusion: 1,
        precision: 2 / 3,
        recall: 2 / 3,
        f1: 2 / 3,
      },
    ],
    threshold_recommendation: {
      job_id: 1,
      confidence_threshold: 0.25,
      precision: 2 / 3,
      recall: 2 / 3,
      f1: 2 / 3,
    },
    latest_prediction_job_id: 1,
  }));
  const getProjectTrainingSummary = vi.fn(async () => ({
    project_id: 1,
    runs: [
      {
        run_id: 2,
        status: "queued",
        model: "custom-drone.pt",
        epochs: 12,
        device: "cpu",
        artifact_path: "/tmp/workspace/projects/1/runs/2",
        map50: null,
        box_loss: null,
        latest_prediction_job_id: null,
        matched: 0,
        false_positive: 0,
        false_negative: 0,
        class_confusion: 0,
        best_threshold: null,
        best_f1: null,
      },
      {
        run_id: 1,
        status: "completed",
        model: "custom-drone.pt",
        epochs: 12,
        device: "mps",
        artifact_path: "/tmp/workspace/projects/1/runs/1",
        map50: 0.42,
        box_loss: 1.12,
        latest_prediction_job_id: 1,
        matched: 2,
        false_positive: 1,
        false_negative: 1,
        class_confusion: 1,
        best_threshold: 0.25,
        best_f1: 2 / 3,
      },
    ],
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
    cancelledRun,
    completedPredictionJob,
    runningPredictionJob,
    trainingRunsResponseQueue,
    predictionJobsResponseQueue,
    listTrainingRuns,
    listPredictionJobs,
    importDataset,
    listProjects,
    listImages,
    createDatasetVersion,
    createTrainingRun,
    cancelTrainingRun,
    getAnnotations,
    replaceAnnotations,
    getTrainingRunLogs,
    getPredictionJobLogs,
    createPredictionJob,
    createPredictionThresholdScan,
    listPredictions,
    getQuality,
    listQualityIssues,
    applyQualityTags,
    refreshImageDimensions,
    defaultQuality,
    defaultQualityIssues,
    defaultClasses,
    listClasses,
    createClass,
    updateClass,
    getTrainingRunSummary,
    getProjectTrainingSummary,
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
  listProjects: apiMock.listProjects,
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
  importDataset: apiMock.importDataset,
  listImages: apiMock.listImages,
  listClasses: apiMock.listClasses,
  createClass: apiMock.createClass,
  updateClass: apiMock.updateClass,
  getAnnotations: apiMock.getAnnotations,
  replaceAnnotations: apiMock.replaceAnnotations,
  getQuality: apiMock.getQuality,
  listQualityIssues: apiMock.listQualityIssues,
  applyQualityTags: apiMock.applyQualityTags,
  refreshImageDimensions: apiMock.refreshImageDimensions,
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
  createTrainingRun: apiMock.createTrainingRun,
  cancelTrainingRun: apiMock.cancelTrainingRun,
  getTrainingRunLogs: apiMock.getTrainingRunLogs,
  getTrainingRunSummary: apiMock.getTrainingRunSummary,
  getProjectTrainingSummary: apiMock.getProjectTrainingSummary,
  listPredictionJobs: apiMock.listPredictionJobs,
  createPredictionJob: apiMock.createPredictionJob,
  createPredictionThresholdScan: apiMock.createPredictionThresholdScan,
  listPredictions: apiMock.listPredictions,
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
      {
        id: 4,
        run_id: 1,
        job_id: 1,
        image_id: 10,
        class_id: 2,
        x_center: 0.58,
        y_center: 0.5,
        width: 0.3,
        height: 0.3,
        confidence: 0.68,
        matched_annotation_id: 1,
        failure_type: "class_confusion",
      },
    ],
    counts: { matched: 1, false_positive: 1, false_negative: 1, class_confusion: 1 },
  }),
  getExportCapabilities: apiMock.getExportCapabilities,
  listRunExports: apiMock.listRunExports,
  createRunExport: apiMock.createRunExport,
}));

describe("App", () => {
  beforeEach(() => {
    apiMock.trainingRunsResponseQueue.length = 0;
    apiMock.predictionJobsResponseQueue.length = 0;
    apiMock.importDataset.mockClear();
    apiMock.listProjects.mockClear();
    apiMock.listImages.mockClear();
    apiMock.listClasses.mockReset();
    apiMock.listClasses.mockImplementation(async () => ({
      items: apiMock.defaultClasses(),
    }));
    apiMock.createClass.mockClear();
    apiMock.updateClass.mockReset();
    apiMock.updateClass.mockImplementation(
      async (
        _projectId: number,
        classId: number,
        body: { name?: string; color?: string; description?: string | null },
      ) => ({
        id: classId,
        project_id: 1,
        name: body.name ?? "target",
        color: body.color ?? "#ef4444",
        description: body.description ?? null,
        active: true,
      }),
    );
    apiMock.createDatasetVersion.mockClear();
    apiMock.createTrainingRun.mockClear();
    apiMock.cancelTrainingRun.mockClear();
    apiMock.getAnnotations.mockClear();
    apiMock.replaceAnnotations.mockClear();
    apiMock.listTrainingRuns.mockClear();
    apiMock.listPredictionJobs.mockClear();
    apiMock.getTrainingRunLogs.mockClear();
    apiMock.getPredictionJobLogs.mockClear();
    apiMock.createPredictionJob.mockClear();
    apiMock.createPredictionThresholdScan.mockClear();
    apiMock.listPredictions.mockClear();
    apiMock.getQuality.mockReset();
    apiMock.getQuality.mockImplementation(async () => apiMock.defaultQuality());
    apiMock.listQualityIssues.mockReset();
    apiMock.listQualityIssues.mockImplementation(async () => apiMock.defaultQualityIssues());
    apiMock.applyQualityTags.mockReset();
    apiMock.applyQualityTags.mockImplementation(async () => ({
      dataset_id: 1,
      issue_type: "duplicate_box",
      scanned_issue_count: 1,
      updated_annotation_count: 1,
      applied_tag_count: 1,
    }));
    apiMock.refreshImageDimensions.mockReset();
    apiMock.refreshImageDimensions.mockImplementation(async () => ({
      dataset_id: 1,
      scanned_count: 2,
      updated_count: 1,
      missing_count: 0,
    }));
    apiMock.getTrainingRunSummary.mockClear();
    apiMock.getProjectTrainingSummary.mockClear();
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
    expect(screen.getByLabelText("Dataset path")).toBeInTheDocument();
    expect(screen.getByLabelText("Project name")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset name")).toBeInTheDocument();
    expect(await screen.findByLabelText("Saved dataset")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load Dataset" })).toBeInTheDocument();
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

    await user.clear(screen.getByLabelText("Project name"));
    await user.type(screen.getByLabelText("Project name"), "Drone QA Project");
    await user.clear(screen.getByLabelText("Dataset name"));
    await user.type(screen.getByLabelText("Dataset name"), "camouflage-set");
    await user.click(screen.getByRole("button", { name: "Import Dataset" }));

    expect(apiMock.importDataset).toHaveBeenCalledWith(
      "/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip",
      "Drone QA Project",
      "camouflage-set",
    );

    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
    expect(screen.getByLabelText("Image filters")).toBeInTheDocument();
    expect(screen.getByLabelText("Version class subset")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Label status"), "annotated");
    await user.selectOptions(screen.getByLabelText("Failure"), "false_negative");
    fireEvent.change(screen.getByLabelText("Edge tag"), { target: { value: "occluded" } });
    await user.click(screen.getByRole("button", { name: "Apply Filters" }));
    expect(apiMock.listImages).toHaveBeenLastCalledWith(
      1,
      {
        platform: undefined,
        label_status: "annotated",
        class_id: undefined,
        edge_tag: "occluded",
        failure_type: "false_negative",
        altitude_min: undefined,
        altitude_max: undefined,
      },
      {
        limit: 50,
        offset: 0,
      },
    );
    expect(screen.getByLabelText("Image pagination")).toHaveTextContent("1-2 of 51");
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(apiMock.listImages).toHaveBeenLastCalledWith(
      1,
      {
        platform: undefined,
        label_status: "annotated",
        class_id: undefined,
        edge_tag: "occluded",
        failure_type: "false_negative",
        altitude_min: undefined,
        altitude_max: undefined,
      },
      {
        limit: 50,
        offset: 50,
      },
    );
    expect(await screen.findByText("iris/frame051.jpg")).toBeInTheDocument();
    expect(screen.getByLabelText("Image pagination")).toHaveTextContent("51-51 of 51");
    await user.click(screen.getByRole("button", { name: "Previous" }));
    expect(apiMock.listImages).toHaveBeenLastCalledWith(
      1,
      {
        platform: undefined,
        label_status: "annotated",
        class_id: undefined,
        edge_tag: "occluded",
        failure_type: "false_negative",
        altitude_min: undefined,
        altitude_max: undefined,
      },
      {
        limit: 50,
        offset: 0,
      },
    );
    expect(await screen.findByRole("button", { name: "Save Annotations" })).toBeInTheDocument();
    expect(await screen.findByText("Ready to export")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Quality metrics")).getByText("Duplicate boxes"),
    ).toBeInTheDocument();
    expect(await screen.findByText("tiny box")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Open Issue" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
    expect(await screen.findByText("smoke-export")).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("button", { name: "Create Dataset Version" }));
    expect(apiMock.createDatasetVersion).toHaveBeenCalledWith(1, undefined, [1]);
    expect(await screen.findByText("Run #1")).toBeInTheDocument();
    expect(await screen.findByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("MixUp"), { target: { value: "0.2" } });
    fireEvent.change(screen.getByLabelText("Copy-Paste"), { target: { value: "0.35" } });
    await user.click(screen.getByRole("checkbox", { name: "GridMask" }));
    await user.click(screen.getByRole("checkbox", { name: "Auto threshold scan" }));
    await user.click(screen.getByRole("button", { name: "Start Training Run" }));
    expect(apiMock.createTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        augmentation_preset: "balanced",
        threshold_scan: true,
        augmentation: expect.objectContaining({
          mosaic: 1,
          mixup: 0.2,
          copy_paste: 0.35,
          gridmask: true,
        }),
      }),
    );
    expect(await screen.findByText("Experiment Dashboard")).toBeInTheDocument();
    expect(apiMock.getProjectTrainingSummary).toHaveBeenCalledWith(1);
    const comparisonPanel = (await screen.findByText("Run Comparison")).closest(
      ".run-comparison-panel",
    ) as HTMLElement;
    expect(within(comparisonPanel).getByText("#2")).toBeInTheDocument();
    expect(within(comparisonPanel).getByText("#1")).toBeInTheDocument();
    expect(within(comparisonPanel).getAllByText("custom-drone.pt")).toHaveLength(2);
    expect(within(comparisonPanel).getByText("0.420")).toBeInTheDocument();
    expect(within(comparisonPanel).getByText("1.120")).toBeInTheDocument();
    expect(within(comparisonPanel).getByText("67%")).toBeInTheDocument();
    expect(within(comparisonPanel).getByText("0.25")).toBeInTheDocument();
    expect(within(comparisonPanel).getByText("runs/1")).toBeInTheDocument();
    expect((await screen.findAllByText("mAP50")).length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("box_loss")).toBeInTheDocument();
    expect(await screen.findByText("Class Outcomes")).toBeInTheDocument();
    expect(await screen.findByText("Threshold Scan")).toBeInTheDocument();
    expect(await screen.findByText("Best threshold")).toBeInTheDocument();
    expect((await screen.findAllByText("0.25")).length).toBeGreaterThanOrEqual(2);
    expect(await screen.findByText("F1")).toBeInTheDocument();
    expect(await screen.findByText("F1 67% | P 67% | R 67%")).toBeInTheDocument();
    expect((await screen.findAllByText("67%")).length).toBeGreaterThanOrEqual(3);
    expect(await screen.findByText("false_positive")).toBeInTheDocument();
    expect(await screen.findByText("class_confusion")).toBeInTheDocument();
    expect((await screen.findAllByText("Confused")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("Matched")).length).toBeGreaterThanOrEqual(2);
    await user.selectOptions(screen.getByLabelText("Failure type"), "class_confusion");
    await user.selectOptions(screen.getByLabelText("Prediction class"), "1");
    fireEvent.change(screen.getByLabelText("Min conf"), { target: { value: "0.5" } });
    fireEvent.change(screen.getByLabelText("Prediction platform"), { target: { value: "iris" } });
    await user.click(screen.getByRole("button", { name: "Apply Sample Filters" }));
    expect(apiMock.listPredictions).toHaveBeenLastCalledWith(1, {
      failure_type: "class_confusion",
      class_id: 1,
      confidence_min: 0.5,
      confidence_max: undefined,
      platform: "iris",
      altitude_min: undefined,
      altitude_max: undefined,
      timestamp_min: undefined,
      timestamp_max: undefined,
    });
    await user.click(screen.getByRole("checkbox", { name: "Use image filters" }));
    expect(
      await screen.findByText(
        "Image filters: annotated | tag occluded | false negative",
      ),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Run Prediction Analysis" }));
    expect(apiMock.createPredictionJob).toHaveBeenCalledWith(1, {
      image_scope: "all",
      confidence_threshold: 0.25,
      image_filters: {
        platform: undefined,
        label_status: "annotated",
        class_id: undefined,
        edge_tag: "occluded",
        failure_type: "false_negative",
        altitude_min: undefined,
        altitude_max: undefined,
      },
    });
    expect(await screen.findByText("Prediction #2")).toBeInTheDocument();
    const thresholdInput = screen.getByLabelText("Scan thresholds");
    fireEvent.change(thresholdInput, { target: { value: "0.1, 0.25, 0.55" } });
    await user.click(screen.getByRole("button", { name: "Run Threshold Scan" }));
    expect(apiMock.createPredictionThresholdScan).toHaveBeenCalledWith(1, {
      image_scope: "all",
      thresholds: [0.1, 0.25, 0.55],
      image_filters: {
        platform: undefined,
        label_status: "annotated",
        class_id: undefined,
        edge_tag: "occluded",
        failure_type: "false_negative",
        altitude_min: undefined,
        altitude_max: undefined,
      },
    });
    expect(await screen.findByText("Prediction #12")).toBeInTheDocument();
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
    expect(screen.getByText("class confusion")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Add as annotation" }));

    expect(screen.getByDisplayValue("false_positive, reviewed_prediction")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Mark reviewed" })[0]);

    expect(
      await screen.findByDisplayValue("occluded, false_negative, reviewed_prediction"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Copy Next" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("X"), { target: { value: "0.42" } });
    await user.click(screen.getByRole("button", { name: "camouflaged" }));
    await user.click(screen.getByRole("button", { name: "hard negative" }));
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
          edge_tags: ["occluded", "camouflaged", "hard_negative"],
        }),
      ]),
    );
  }, 15000);

  it("shows annotation readiness before a dataset is loaded", async () => {
    render(<App />);

    expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
    const readiness = screen.getByLabelText("Annotation readiness");

    expect(within(readiness).getByText("Needed | Dataset loaded")).toBeInTheDocument();
    expect(within(readiness).getByText("Needed | Class library")).toBeInTheDocument();
    expect(within(readiness).getByText("Needed | Image selected")).toBeInTheDocument();
    expect(within(readiness).getByText("Needed | Class selected")).toBeInTheDocument();
    expect(screen.getByText("Load or import a dataset to begin annotation.")).toBeInTheDocument();
  });

  it("guides empty class libraries and selects a created class", async () => {
    const user = userEvent.setup();
    apiMock.listClasses.mockImplementation(async () => ({ items: [] }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));

    const readiness = screen.getByLabelText("Annotation readiness");
    expect(within(readiness).getByText("Ready | Dataset loaded")).toBeInTheDocument();
    expect(within(readiness).getByText("Needed | Class library")).toBeInTheDocument();
    expect(within(readiness).getByText("Ready | Image selected")).toBeInTheDocument();
    expect(within(readiness).getByText("Needed | Class selected")).toBeInTheDocument();
    expect(screen.getByText("Create a project class before drawing boxes.")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Class name"), "vehicle");
    await user.click(screen.getByRole("button", { name: "Create Class" }));

    expect(apiMock.createClass).toHaveBeenCalledWith(1, {
      name: "vehicle",
      color: "#ef4444",
    });
    expect(await screen.findByRole("button", { name: "vehicle" })).toHaveClass("selected");
    expect(within(readiness).getByText("Ready | Class library")).toBeInTheDocument();
    expect(within(readiness).getByText("Ready | Class selected")).toBeInTheDocument();
    expect(screen.getByText("Drag over the image to add a bounding box.")).toBeInTheDocument();
  });

  it("edits a project class and refreshes dependent annotation labels", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));
    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit target" }));
    await user.clear(screen.getByLabelText("Edit class name target"));
    await user.type(screen.getByLabelText("Edit class name target"), "vehicle");
    fireEvent.change(screen.getByLabelText("Edit class color target"), {
      target: { value: "#22c55e" },
    });
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(apiMock.updateClass).toHaveBeenCalledWith(1, 1, {
      name: "vehicle",
      color: "#22c55e",
    });

    const classLibrary = screen.getByLabelText("Available classes");
    expect(await within(classLibrary).findByRole("button", { name: "vehicle" })).toHaveClass(
      "selected",
    );
    expect(screen.queryByRole("button", { name: "target" })).not.toBeInTheDocument();

    const versionSubset = screen.getByLabelText("Version class subset");
    expect(within(versionSubset).getByRole("checkbox", { name: "vehicle" })).toBeChecked();

    expect(
      screen.getByText("vehicle", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();
  });

  it("changes an existing annotation box class before saving", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));
    await user.type(screen.getByLabelText("Class name"), "vehicle");
    await user.click(screen.getByRole("button", { name: "Create Class" }));
    expect(await screen.findByRole("button", { name: "vehicle" })).toHaveClass("selected");

    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Box 1 class"), "2");
    expect(
      screen.getByText("vehicle", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Save Annotations" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      11,
      expect.arrayContaining([
        expect.objectContaining({
          class_id: 2,
          x_center: 0.6,
          y_center: 0.55,
          width: 0.25,
          height: 0.2,
          track_id: "copy-source",
          edge_tags: ["occluded"],
        }),
      ]),
    );
  });

  it("moves an existing annotation box from the canvas and nudge controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    const canvas = screen.getByLabelText("Annotation canvas");
    Object.defineProperty(canvas, "getBoundingClientRect", {
      configurable: true,
      value: () => ({ left: 0, top: 0, width: 1000, height: 1000, right: 1000, bottom: 1000 }),
    });

    const canvasBox = screen.getByTestId("annotation-box-annotation-101");
    fireEvent(canvasBox, pointerEvent("pointerdown", 600, 550));
    expect(canvasBox).toHaveClass("selected");
    await act(async () => {
      fireEvent(window, pointerEvent("pointermove", 650, 520));
      fireEvent(window, pointerEvent("pointerup", 650, 520));
    });
    await user.click(screen.getByRole("button", { name: "Move box 1 left" }));

    await user.click(screen.getByRole("button", { name: "Save Annotations" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      11,
      expect.arrayContaining([
        expect.objectContaining({
          class_id: 1,
          x_center: 0.64,
          y_center: 0.52,
          width: 0.25,
          height: 0.2,
          track_id: "copy-source",
          edge_tags: ["occluded"],
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

  it("loads a historical run config and reruns it on the latest version", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));
    expect(await screen.findByText("Run #1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load Config" }));

    expect(screen.getByLabelText("Model preset or local weights")).toHaveValue("custom-drone.pt");
    expect(screen.getByLabelText("Epochs")).toHaveValue(12);
    expect(screen.getByLabelText("Image size")).toHaveValue(512);
    expect(screen.getByLabelText("Batch")).toHaveValue(4);
    expect(screen.getByLabelText("Device")).toHaveValue("mps");
    expect(screen.getByLabelText("Strategy name")).toHaveValue("edge-case");
    expect(screen.getByLabelText("Mosaic")).toHaveValue(0.8);
    expect(screen.getByLabelText("MixUp")).toHaveValue(0.15);
    expect(screen.getByRole("checkbox", { name: "GridMask" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "TTA" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Auto threshold scan" })).toBeChecked();
    expect(await screen.findByText("Loaded config from Run #1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Rerun" }));

    expect(apiMock.createTrainingRun).toHaveBeenLastCalledWith(
      expect.objectContaining({
        version_id: 1,
        model: "custom-drone.pt",
        epochs: 12,
        image_size: 512,
        batch_size: 4,
        device: "mps",
        augmentation_preset: "edge-case",
        tta: true,
        threshold_scan: true,
        augmentation: expect.objectContaining({
          mosaic: 0.8,
          mixup: 0.15,
          copy_paste: 0.25,
          gridmask: true,
        }),
      }),
    );
    expect(await screen.findByText("Started rerun from Run #1")).toBeInTheDocument();
  });

  it("loads a saved dataset without re-importing source files", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByLabelText("Saved dataset")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load Dataset" }));

    expect(apiMock.importDataset).not.toHaveBeenCalled();
    expect(apiMock.listImages).toHaveBeenCalledWith(1, {}, { limit: 50, offset: 0 });
    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
    const summaries = await screen.findAllByText((_, element) =>
      Boolean(
        element?.classList.contains("summary-line") &&
          element.textContent?.includes("Drone QA Project / camouflage-set"),
      ),
    );
    expect(summaries).toHaveLength(1);
  });

  it("surfaces duplicate box quality issues", async () => {
    const user = userEvent.setup();
    apiMock.getQuality.mockImplementation(async () => ({
      ...apiMock.defaultQuality(),
      duplicate_box_count: 1,
      ready_for_training: false,
      issues: ["1 box duplicates another box on the same image and class."],
    }));
    apiMock.listQualityIssues.mockImplementation(async () => ({
      ...apiMock.defaultQualityIssues(),
      total: 1,
      items: [
        {
          issue_type: "duplicate_box",
          severity: "error",
          message: "Box duplicates annotation 1 on the same image and class.",
          image_id: 11,
          image_path: "iris/frame002.jpg",
          image_url: "/api/images/11/file",
          annotation_id: 102,
          class_id: 1,
          class_name: "target",
          x_center: 0.6,
          y_center: 0.55,
          width: 0.25,
          height: 0.2,
        },
      ],
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));

    expect(await screen.findByText("Needs attention")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Quality metrics")).getByText("Duplicate boxes"),
    ).toBeInTheDocument();
    expect(await screen.findByText("duplicate box")).toBeInTheDocument();
    expect(
      await screen.findByText("Box duplicates annotation 1 on the same image and class."),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Quality issue type"), "duplicate_box");

    expect(apiMock.listQualityIssues).toHaveBeenLastCalledWith(1, "duplicate_box");

    await user.click(screen.getByRole("button", { name: "Apply Auto Tags" }));

    expect(apiMock.applyQualityTags).toHaveBeenCalledWith(1, "duplicate_box");
    expect(await screen.findByText("1 tags applied to 1 annotations")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open Issue" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
  });

  it("surfaces missing metadata quality issues", async () => {
    const user = userEvent.setup();
    apiMock.getQuality.mockImplementation(async () => ({
      ...apiMock.defaultQuality(),
      missing_metadata_count: 1,
      issues: ["1 image is missing platform, altitude, timestamp, or source metadata."],
    }));
    apiMock.listQualityIssues.mockImplementation(async () => ({
      ...apiMock.defaultQualityIssues(),
      total: 1,
      items: [
        {
          issue_type: "missing_metadata",
          severity: "warning",
          message: "Image is missing source metadata row, altitude, timestamp.",
          image_id: 11,
          image_path: "iris/frame002.jpg",
          image_url: "/api/images/11/file",
          annotation_id: null,
          class_id: null,
          class_name: null,
          x_center: null,
          y_center: null,
          width: null,
          height: null,
        },
      ],
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));

    expect(await screen.findByText("Ready to export")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Quality metrics")).getByText("Missing metadata"),
    ).toBeInTheDocument();
    expect(await screen.findByText("missing metadata")).toBeInTheDocument();
    expect(
      await screen.findByText("Image is missing source metadata row, altitude, timestamp."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open Issue" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
  });

  it("surfaces unreadable image dimension quality issues", async () => {
    const user = userEvent.setup();
    apiMock.getQuality.mockImplementation(async () => ({
      ...apiMock.defaultQuality(),
      missing_image_dimensions_count: 1,
      issues: ["1 image has unreadable image dimensions."],
    }));
    apiMock.listQualityIssues.mockImplementation(async () => ({
      ...apiMock.defaultQualityIssues(),
      total: 1,
      items: [
        {
          issue_type: "missing_image_dimensions",
          severity: "warning",
          message: "Image width or height could not be read.",
          image_id: 11,
          image_path: "iris/frame002.jpg",
          image_url: "/api/images/11/file",
          annotation_id: null,
          class_id: null,
          class_name: null,
          x_center: null,
          y_center: null,
          width: null,
          height: null,
        },
      ],
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));

    expect(await screen.findByText("Ready to export")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("Quality metrics")).getByText("Missing dimensions"),
    ).toBeInTheDocument();
    expect(await screen.findByText("missing image dimensions")).toBeInTheDocument();
    expect(
      await screen.findByText("Image width or height could not be read."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Open Issue" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
  });

  it("refreshes missing image dimensions from the quality panel", async () => {
    const user = userEvent.setup();
    apiMock.getQuality
      .mockImplementationOnce(async () => ({
        ...apiMock.defaultQuality(),
        missing_image_dimensions_count: 1,
        issues: ["1 image has unreadable image dimensions."],
      }))
      .mockImplementation(async () => apiMock.defaultQuality());

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Load Dataset" }));
    expect(
      within(screen.getByLabelText("Quality metrics")).getByText("Missing dimensions"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Refresh Dimensions" }));

    expect(apiMock.refreshImageDimensions).toHaveBeenCalledWith(1);
    expect(await screen.findByText("2 scanned, 1 updated, 0 still missing")).toBeInTheDocument();
    expect(apiMock.listImages).toHaveBeenCalledTimes(2);
    expect(apiMock.getQuality).toHaveBeenCalledTimes(2);
  });

  it("cancels an active training run from run history", async () => {
    apiMock.trainingRunsResponseQueue.push(
      { items: [apiMock.runningRun] },
      { items: [apiMock.cancelledRun] },
    );

    render(<App />);
    await flushPromises();

    fireEvent.click(screen.getByRole("button", { name: "Import Dataset" }));
    await flushPromises();

    expect(screen.getByText("Run #2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Cancel Run" }));
    await flushPromises();

    expect(apiMock.cancelTrainingRun).toHaveBeenCalledWith(2);
    expect(screen.getByText("cancelled")).toBeInTheDocument();
    expect(apiMock.getTrainingRunLogs).toHaveBeenCalledWith(2);
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
