import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { DatasetQualityIssueListResponse, DatasetQualitySummary } from "./api";

vi.mock("./auth", () => ({
  useAuth: () => ({
    user: {
      id: 1,
      username: "admin",
      role: "admin",
      is_active: true,
      created_at: "2026-06-30T00:00:00",
      updated_at: "2026-06-30T00:00:00",
    },
    token: "test-token",
    loading: false,
    isAdmin: true,
    login: vi.fn(),
    logout: vi.fn(),
    refreshMe: vi.fn(),
  }),
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
  roleLabel: (role: string) => (role === "admin" ? "管理员" : "标注员"),
}));

function pointerEvent(type: string, clientX: number, clientY: number, button = 0) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    clientX: { value: clientX },
    clientY: { value: clientY },
    pointerId: { value: 1 },
    button: { value: button },
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
    version_id: 1,
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
  const listTrainingModels = vi.fn(async () => ({
    items: [
      {
        model_ref: "base:yolov8n.pt",
        label: "内置模型 · YOLOv8n",
        kind: "base",
        run_id: null,
        weight_id: null,
        status: "可用",
      },
      {
        model_ref: "base:yolov8s.pt",
        label: "内置模型 · YOLOv8s",
        kind: "base",
        run_id: null,
        weight_id: null,
        status: "可用",
      },
      {
        model_ref: "base:yolov8m.pt",
        label: "内置模型 · YOLOv8m",
        kind: "base",
        run_id: null,
        weight_id: null,
        status: "可用",
      },
      {
        model_ref: "run:1",
        label: "训练任务 #1 · best.pt",
        kind: "trained",
        run_id: 1,
        weight_id: null,
        status: "completed",
      },
    ],
  }));
  const listModelWeights = vi.fn(async () => ({ items: [] }));
  const uploadModelWeight = vi.fn(async () => ({
    id: 9,
    project_id: 1,
    original_filename: "uploaded.pt",
    size_bytes: 12,
    created_at: "2026-06-30T00:05:00",
    updated_at: "2026-06-30T00:05:00",
  }));
  const deleteModelWeight = vi.fn(async () => undefined);
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
            annotated_image_count: 1,
            annotation_count: 3,
          },
        ],
      },
    ],
  }));
  const listDatasetSources = vi.fn(async () => ({
    items: [
      {
        source_ref: "server:/tmp/image_dataset.zip",
        label: "服务器文件 · image_dataset.zip",
        kind: "server",
        source_path: "/tmp/image_dataset.zip",
        original_filename: "image_dataset.zip",
        size_bytes: 1024,
        source_id: null,
      },
    ],
  }));
  const listDetectedClasses = vi.fn(async () => ({
    dataset_id: 1,
    total_images: 8,
    method: "filename_prefix_h_a",
    items: [
      { name: "fire_truck", image_count: 1, color: "#e45756", sample_filenames: ["fire_truck_h20_a045.jpg"] },
      { name: "person_white", image_count: 1, color: "#2f80ed", sample_filenames: ["person_white_h20_a000.jpg"] },
      { name: "prius_hybrid", image_count: 1, color: "#27ae60", sample_filenames: ["prius_hybrid_h20_a000.jpg"] },
      { name: "car_lexus", image_count: 1, color: "#f2994a", sample_filenames: ["car_lexus_h20_a000.jpg"] },
      { name: "prius_hybrid_camo", image_count: 1, color: "#9b51e0", sample_filenames: ["prius_hybrid_camo_h20_a000.jpg"] },
      { name: "suv_camo", image_count: 1, color: "#1f6f78", sample_filenames: ["suv_camo_h20_a000.jpg"] },
      { name: "car_opel", image_count: 1, color: "#d97706", sample_filenames: ["car_opel_h20_a000.jpg"] },
      { name: "person_red", image_count: 1, color: "#dc2626", sample_filenames: ["person_red_h20_a000.jpg"] },
    ],
  }));
  const uploadDatasetSource = vi.fn(async () => ({
    id: 1,
    original_filename: "uploaded.zip",
    size_bytes: 2048,
    source_path: "/tmp/uploaded.zip",
    created_at: "2026-06-30T00:00:00",
    updated_at: "2026-06-30T00:00:00",
  }));
  const deleteDatasetSource = vi.fn(async () => undefined);
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
  const createDatasetVersion = vi.fn(async (
    _datasetId: number,
    name?: string,
    classIds?: number[],
    imageScope: "annotated" | "all" = "annotated",
  ) => ({
    id: 2,
    project_id: 1,
    dataset_id: 1,
    name: name || "mvp-quality-pass",
    class_mapping: Object.fromEntries((classIds ?? [1]).map((classId, index) => [String(classId), index])),
    image_scope: imageScope,
    split_counts: { train: 1, val: 0, test: 0 },
    artifact_path: "/tmp/workspace/projects/1/versions/2",
    frozen: true,
    created_at: "2026-06-30T00:01:00",
  }));
  const listDatasetVersions = vi.fn(async () => ({
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
  }));
  const listProjectVersions = vi.fn(async () => ({
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
  const getTrainingRunArtifacts = vi.fn(async (runId: number) => ({
    run_id: runId,
    artifact_root: `/tmp/workspace/projects/1/runs/${runId}`,
    total_count: 4,
    items: [
      { relative_path: "config.json", category: "config", size_bytes: 320 },
      { relative_path: "logs.txt", category: "log", size_bytes: 128 },
      { relative_path: "metrics.jsonl", category: "metrics", size_bytes: 96 },
      {
        relative_path: "ultralytics/weights/best.pt",
        category: "weights",
        size_bytes: 4096,
      },
    ],
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
        version_id?: number;
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
      version_id: body.version_id ?? 1,
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
  const defaultCoverage = () => ({
    dataset_id: 1,
    image_count: 2,
    annotated_image_count: 1,
    annotation_count: 1,
    platforms: [
      {
        label: "iris",
        image_count: 1,
        annotated_image_count: 1,
        annotation_count: 1,
      },
      {
        label: "vtol",
        image_count: 1,
        annotated_image_count: 0,
        annotation_count: 0,
      },
    ],
    altitude_bands: [
      {
        label: "<20m",
        image_count: 1,
        annotated_image_count: 1,
        annotation_count: 1,
      },
      {
        label: "20-50m",
        image_count: 1,
        annotated_image_count: 0,
        annotation_count: 0,
      },
    ],
    classes: [
      {
        class_id: 1,
        class_name: "target",
        class_color: "#ef4444",
        image_count: 1,
        annotation_count: 1,
      },
      {
        class_id: 2,
        class_name: "vehicle",
        class_color: "#22c55e",
        image_count: 0,
        annotation_count: 0,
      },
    ],
    edge_tags: [
      {
        tag: "occluded",
        image_count: 1,
        annotation_count: 1,
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
      annotation_count: 0,
      version_count: 0,
    },
  ];
  const listClasses = vi.fn(async () => ({
    items: defaultClasses(),
  }));
  const createClass = vi.fn(async (_projectId: number, _body: { name: string; color: string }) => ({
    id: 2,
    project_id: 1,
    name: "vehicle",
    color: "#22c55e",
    description: null,
    active: true,
    annotation_count: 0,
    version_count: 0,
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
      annotation_count: 0,
      version_count: 0,
    }),
  );
  const getQuality = vi.fn(async () => defaultQuality());
  const getDatasetCoverage = vi.fn(async () => defaultCoverage());
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
      download_url: `/api/training/runs/${runId}/exports/${exportArtifacts.length + 1}/file`,
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
  const storageItemsResponse = {
    items: [
      {
        entity_type: "dataset" as const,
        entity_id: 1,
        display_name: "camouflage-set",
        project_id: 1,
        project_name: "Drone QA Project",
        dataset_id: 1,
        version_id: null,
        artifact_path: "/tmp/workspace/projects/1/datasets/1",
        size_bytes: 2048,
        image_count: 2,
        annotation_count: 2,
        split_counts: null,
        created_at: "2026-06-30T00:00:00",
        protected: false,
        blockers: [],
      },
      {
        entity_type: "dataset_version" as const,
        entity_id: 1,
        display_name: "smoke-export",
        project_id: 1,
        project_name: "Drone QA Project",
        dataset_id: 1,
        version_id: 1,
        artifact_path: "/tmp/workspace/projects/1/versions/1",
        size_bytes: 1024,
        image_count: 2,
        annotation_count: 2,
        split_counts: { train: 1, val: 1, test: 0 },
        created_at: "2026-06-30T00:01:00",
        protected: false,
        blockers: [],
      },
    ],
    total_size_bytes: 3072,
  };
  const listStorageItems = vi.fn(async () => storageItemsResponse);
  const getStorageItem = vi.fn(async (entityType: "dataset" | "dataset_version" | "training_run", entityId: number) => {
    const item = storageItemsResponse.items.find(
      (candidate) => candidate.entity_type === entityType && candidate.entity_id === entityId,
    );
    if (!item) throw new Error("数据不存在");
    return { ...item, class_names: ["target"], related_runs: [] };
  });
  const trashStorageItem = vi.fn(async (entityType: "dataset" | "dataset_version" | "training_run", entityId: number) => ({
    id: 100 + entityId,
    entity_type: entityType,
    entity_id: entityId,
    project_id: 1,
    dataset_id: 1,
    version_id: entityType === "dataset_version" ? entityId : null,
    display_name: entityType === "dataset" ? "camouflage-set" : "smoke-export",
    original_path: "/tmp/original",
    trash_path: "/tmp/trash",
    size_bytes: 1024,
    summary: {},
    status: "active",
    error_message: null,
    deleted_at: "2026-06-30T00:02:00",
    purge_after: "2026-07-30T00:02:00",
    created_at: "2026-06-30T00:02:00",
    updated_at: "2026-06-30T00:02:00",
  }));
  const listTrashItems = vi.fn(async () => ({ items: [], total_size_bytes: 0 }));
  const restoreTrashItem = vi.fn(async () => ({ status: "restored", message: "已恢复" }));
  const purgeTrashItem = vi.fn(async () => ({ status: "purged", message: "已删除" }));
  const purgeExpiredTrash = vi.fn(async () => ({ purged_count: 0, failed_count: 0, failures: [] }));

  return {
    completedRun,
    runningRun,
    cancelledRun,
    completedPredictionJob,
    runningPredictionJob,
    trainingRunsResponseQueue,
    predictionJobsResponseQueue,
    listTrainingRuns,
    listTrainingModels,
    listModelWeights,
    uploadModelWeight,
    deleteModelWeight,
    listPredictionJobs,
    importDataset,
    listProjects,
    listDatasetSources,
    listDetectedClasses,
    uploadDatasetSource,
    deleteDatasetSource,
    datasetImages,
    listImages,
    createDatasetVersion,
    listDatasetVersions,
    listProjectVersions,
    createTrainingRun,
    cancelTrainingRun,
    getAnnotations,
    replaceAnnotations,
    getTrainingRunLogs,
    getTrainingRunArtifacts,
    getPredictionJobLogs,
    createPredictionJob,
    createPredictionThresholdScan,
    listPredictions,
    getQuality,
    getDatasetCoverage,
    listQualityIssues,
    applyQualityTags,
    refreshImageDimensions,
    defaultQuality,
    defaultCoverage,
    defaultQualityIssues,
    defaultClasses,
    listClasses,
    createClass,
    updateClass,
    deleteClass: vi.fn(),
    getTrainingRunSummary,
    getProjectTrainingSummary,
    getExportCapabilities,
    listRunExports,
    createRunExport,
    exportArtifacts,
    storageItemsResponse,
    listStorageItems,
    getStorageItem,
    trashStorageItem,
    listTrashItems,
    restoreTrashItem,
    purgeTrashItem,
    purgeExpiredTrash,
  };
});

vi.mock("./api", () => ({
  getHealth: async () => ({
    status: "ok",
    app: "YoloStudio",
    workspace_root: "/tmp/workspace",
    database_path: "/tmp/workspace/app.db",
    devices: { selected: "cpu", available: ["cpu"], details: {} },
  }),
  setUnauthorizedHandler: vi.fn(),
  setAuthToken: vi.fn(),
  authedMediaUrl: (url: string) => url,
  getRunBestWeightsUrl: (runId: number) => `/api/training/runs/${runId}/weights/best`,
  getExportDownloadUrl: (runId: number, exportId: number) =>
    `/api/training/runs/${runId}/exports/${exportId}/file`,
  listProjects: apiMock.listProjects,
  listDatasetSources: apiMock.listDatasetSources,
  listDetectedClasses: apiMock.listDetectedClasses,
  uploadDatasetSource: apiMock.uploadDatasetSource,
  deleteDatasetSource: apiMock.deleteDatasetSource,
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
  deleteClass: apiMock.deleteClass,
  getAnnotations: apiMock.getAnnotations,
  replaceAnnotations: apiMock.replaceAnnotations,
  getQuality: apiMock.getQuality,
  getDatasetCoverage: apiMock.getDatasetCoverage,
  listQualityIssues: apiMock.listQualityIssues,
  applyQualityTags: apiMock.applyQualityTags,
  refreshImageDimensions: apiMock.refreshImageDimensions,
  listDatasetVersions: apiMock.listDatasetVersions,
  listProjectVersions: apiMock.listProjectVersions,
  createDatasetVersion: apiMock.createDatasetVersion,
  listTrainingRuns: apiMock.listTrainingRuns,
  listTrainingModels: apiMock.listTrainingModels,
  listModelWeights: apiMock.listModelWeights,
  uploadModelWeight: apiMock.uploadModelWeight,
  deleteModelWeight: apiMock.deleteModelWeight,
  createTrainingRun: apiMock.createTrainingRun,
  cancelTrainingRun: apiMock.cancelTrainingRun,
  getTrainingRunLogs: apiMock.getTrainingRunLogs,
  getTrainingRunArtifacts: apiMock.getTrainingRunArtifacts,
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
  listStorageItems: apiMock.listStorageItems,
  getStorageItem: apiMock.getStorageItem,
  trashStorageItem: apiMock.trashStorageItem,
  listTrashItems: apiMock.listTrashItems,
  restoreTrashItem: apiMock.restoreTrashItem,
  purgeTrashItem: apiMock.purgeTrashItem,
  purgeExpiredTrash: apiMock.purgeExpiredTrash,
}));

describe("App", () => {
  beforeEach(() => {
    class ResizeObserverMock {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", ResizeObserverMock);
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("annotation-canvas-viewport") ? 500 : 0;
    });
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("annotation-canvas-viewport") ? 300 : 0;
    });
    window.history.replaceState(null, "", window.location.pathname);
    apiMock.trainingRunsResponseQueue.length = 0;
    apiMock.predictionJobsResponseQueue.length = 0;
    apiMock.importDataset.mockClear();
    apiMock.listProjects.mockClear();
    apiMock.listDetectedClasses.mockClear();
    apiMock.listImages.mockClear();
    apiMock.listClasses.mockReset();
    apiMock.listClasses.mockImplementation(async () => ({
      items: apiMock.defaultClasses(),
    }));
    apiMock.createClass.mockClear();
    apiMock.deleteClass.mockReset();
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
        annotation_count: 0,
        version_count: 0,
      }),
    );
    apiMock.createDatasetVersion.mockClear();
    apiMock.listDatasetVersions.mockReset();
    apiMock.listDatasetVersions.mockImplementation(async () => ({
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
    }));
    apiMock.listProjectVersions.mockReset();
    apiMock.listProjectVersions.mockImplementation(async () => ({
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
    }));
    apiMock.createTrainingRun.mockClear();
    apiMock.cancelTrainingRun.mockClear();
    apiMock.getAnnotations.mockClear();
    apiMock.replaceAnnotations.mockClear();
    apiMock.listTrainingRuns.mockClear();
    apiMock.listTrainingModels.mockClear();
    apiMock.listModelWeights.mockClear();
    apiMock.uploadModelWeight.mockClear();
    apiMock.deleteModelWeight.mockClear();
    apiMock.listPredictionJobs.mockClear();
    apiMock.getTrainingRunLogs.mockClear();
    apiMock.getTrainingRunArtifacts.mockClear();
    apiMock.getPredictionJobLogs.mockClear();
    apiMock.createPredictionJob.mockClear();
    apiMock.createPredictionThresholdScan.mockClear();
    apiMock.listPredictions.mockClear();
    apiMock.getQuality.mockReset();
    apiMock.getQuality.mockImplementation(async () => apiMock.defaultQuality());
    apiMock.getDatasetCoverage.mockReset();
    apiMock.getDatasetCoverage.mockImplementation(async () => apiMock.defaultCoverage());
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
    apiMock.listStorageItems.mockClear();
    apiMock.getStorageItem.mockClear();
    apiMock.trashStorageItem.mockClear();
    apiMock.listTrashItems.mockClear();
    apiMock.restoreTrashItem.mockClear();
    apiMock.purgeTrashItem.mockClear();
    apiMock.purgeExpiredTrash.mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders local app status and dataset scan controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("YoloStudio")).toBeInTheDocument();
    expect(await screen.findByText("cpu")).toBeInTheDocument();
    expect(screen.getByTitle("/tmp/workspace")).toHaveTextContent("/tmp/workspace");
    expect(screen.getByTitle("/tmp/workspace/app.db")).toHaveTextContent(
      "/tmp/workspace/app.db",
    );
    expect(screen.getByLabelText("数据集来源")).toBeInTheDocument();
    expect(screen.getByLabelText("项目名称")).toHaveValue("YOLO 目标检测项目");
    expect(screen.getByLabelText("数据集名称")).toBeInTheDocument();
    expect(await screen.findByLabelText("已保存数据集")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("已保存数据集")).getByRole("option", {
        name: "Drone QA Project / camouflage-set（2 张图像 · 已标注 1 张 · 3 个边界框）",
      }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "加载数据集" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "扫描数据集" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "导入数据集" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "工作流步骤" })).toBeInTheDocument();
    expect(screen.queryByText("类别库")).not.toBeInTheDocument();
    expect(screen.queryByText("图像浏览器")).not.toBeInTheDocument();

    await user.clear(screen.getByLabelText("项目名称"));
    await user.type(screen.getByLabelText("项目名称"), "Drone QA Project");
    await user.clear(screen.getByLabelText("数据集名称"));
    await user.type(screen.getByLabelText("数据集名称"), "camouflage-set");
    await user.click(screen.getByRole("button", { name: "导入数据集" }));

    expect(apiMock.importDataset).toHaveBeenCalledWith(
      expect.stringMatching(/image_dataset\.zip$/),
      "Drone QA Project",
      "camouflage-set",
    );

    expect(window.location.hash).toBe("#classes");
    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    expect(screen.queryByText("图像浏览器")).not.toBeInTheDocument();

    await navigateToStep(user, "图像标注");
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
    expect(screen.getByLabelText("图像筛选器")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("标注状态"), "annotated");
    await user.click(screen.getByRole("button", { name: /高级筛选/ }));
    await user.selectOptions(screen.getByLabelText("识别结果"), "false_negative");
    fireEvent.change(screen.getByLabelText("边缘案例标签"), { target: { value: "occluded" } });
    await user.click(screen.getByRole("button", { name: "应用筛选" }));
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
    expect(screen.getByLabelText("图像分页")).toHaveTextContent("第 1-2 张，共 51 张");
    await user.click(screen.getByRole("button", { name: "下一页" }));
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
    expect(screen.getByLabelText("图像分页")).toHaveTextContent("第 51-51 张，共 51 张");
    await user.click(screen.getByRole("button", { name: "上一页" }));
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
    expect(await screen.findByRole("button", { name: "保存" })).toBeInTheDocument();

    await navigateToStep(user, "质量与版本");
    expect(screen.getByLabelText("版本类别子集")).toBeInTheDocument();
    expect(await screen.findByText("可以导出")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量指标")).getByText("重复边界框"),
    ).toBeInTheDocument();
    expect(apiMock.getDatasetCoverage).toHaveBeenCalledWith(1);
    const coveragePanel = screen.getByLabelText("数据集覆盖情况");
    expect(within(coveragePanel).getByText("数据集覆盖情况")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("1/2 张图像 | 1 个边界框")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("平台")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("高度")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("边缘案例标签")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("iris")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("20-50m")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("vehicle")).toBeInTheDocument();
    expect(within(coveragePanel).getByText("遮挡")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量问题样本")).getByText("极小边界框"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "打开问题图像" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(window.location.hash).toBe("#annotation");
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
    await navigateToStep(user, "质量与版本");
    expect(await screen.findByText("smoke-export")).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("checkbox", { name: "target" }));
    await user.click(screen.getByRole("button", { name: "创建数据集版本" }));
    expect(apiMock.createDatasetVersion).toHaveBeenCalledWith(1, undefined, [1], "annotated", [1]);

    await navigateToStep(user, "模型训练");
    expect(await screen.findByText("训练任务 #1")).toBeInTheDocument();
    expect(await screen.findByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    expect(apiMock.getTrainingRunArtifacts).toHaveBeenCalledWith(1);
    const runRow = screen.getByText("训练任务 #1").closest(".run-row") as HTMLElement;
    expect(within(runRow).getByText("训练产物")).toBeInTheDocument();
    expect(within(runRow).getByText("ultralytics/weights/best.pt")).toBeInTheDocument();
    expect(within(runRow).getByText("4.0 KB")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("MixUp"), { target: { value: "0.2" } });
    fireEvent.change(screen.getByLabelText("Copy-Paste"), { target: { value: "0.35" } });
    await user.click(screen.getByRole("checkbox", { name: "GridMask" }));
    await user.click(screen.getByRole("checkbox", { name: "自动阈值扫描" }));
    await user.click(screen.getByRole("button", { name: "开始训练" }));
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
    expect(screen.queryByText("实验看板")).not.toBeInTheDocument();
    await navigateToStep(user, "评估与导出");
    expect(await screen.findByLabelText("评估权重")).toBeInTheDocument();
    expect(screen.getByLabelText("评估数据版本")).toBeInTheDocument();
    expect(screen.getByLabelText("图像范围")).toBeInTheDocument();
    expect(screen.queryByText("执行阈值扫描")).not.toBeInTheDocument();
    expect(screen.queryByText("使用图像筛选条件")).not.toBeInTheDocument();
    expect(screen.queryByText("ONNX")).not.toBeInTheDocument();
    expect(screen.queryByText("TensorRT")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "展开高级：实验看板" }));
    expect(await screen.findByText("实验看板")).toBeInTheDocument();
    expect(apiMock.getProjectTrainingSummary).toHaveBeenCalledWith(1);
    const comparisonPanel = (await screen.findByText("任务对比")).closest(
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
    expect(await screen.findByText("类别检测结果")).toBeInTheDocument();
    expect(await screen.findByText("阈值扫描")).toBeInTheDocument();
    expect(await screen.findByText("最佳阈值")).toBeInTheDocument();
    expect((await screen.findAllByText("0.25")).length).toBeGreaterThanOrEqual(2);
    expect(await screen.findByText("F1")).toBeInTheDocument();
    expect(await screen.findByText("F1 67% | P 67% | R 67%")).toBeInTheDocument();
    expect((await screen.findAllByText("67%")).length).toBeGreaterThanOrEqual(3);
    expect((await screen.findAllByText("误报")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("类别混淆")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("匹配正确")).length).toBeGreaterThanOrEqual(2);
    await user.selectOptions(screen.getByLabelText("结果类型"), "class_confusion");
    await user.selectOptions(screen.getByLabelText("预测类别"), "1");
    fireEvent.change(screen.getByLabelText("最低置信度"), { target: { value: "0.5" } });
    await user.click(screen.getByRole("button", { name: "应用样本筛选" }));
    expect(apiMock.listPredictions).toHaveBeenLastCalledWith(1, {
      failure_type: "class_confusion",
      class_id: 1,
      confidence_min: 0.5,
      confidence_max: undefined,
    });
    await user.click(screen.getByRole("button", { name: "开始预测分析" }));
    expect(apiMock.createPredictionJob).toHaveBeenCalledWith(1, {
      image_scope: "all",
      confidence_threshold: 0.25,
      version_id: 1,
    });
    expect(await screen.findByText("预测任务 #2")).toBeInTheDocument();
    expect(await screen.findByText("模型导出")).toBeInTheDocument();
    expect(await screen.findByText(".pt 权重")).toBeInTheDocument();
    expect(screen.queryByText("ONNX")).not.toBeInTheDocument();
    expect(screen.queryByText("TensorRT")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "导出 PT" }));

    expect(apiMock.createRunExport).toHaveBeenCalledWith(1, "pt");
    expect(await screen.findByText("PT 导出任务 #1")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /下载训练权重 best\.pt/ })).toHaveAttribute(
      "href",
      "/api/training/runs/1/weights/best",
    );
    expect(
      within(screen.getByLabelText("导出产物")).getByRole("link", { name: /下载/ }),
    ).toHaveAttribute("href", "/api/training/runs/1/exports/1/file");

    await user.click(within(screen.getByLabelText("预测样本")).getAllByRole("button")[0]);

    expect(window.location.hash).toBe("#evaluation");
    expect(await screen.findByText("筛选样本查看")).toBeInTheDocument();
    expect(screen.getByLabelText("问题样本查看器")).toBeInTheDocument();
    expect(screen.getByLabelText("评估审查图层")).toBeInTheDocument();
    const predictionLegend = screen.getByLabelText("预测结果图例");
    expect(within(predictionLegend).getByText("误报")).toBeInTheDocument();
    expect(within(predictionLegend).getByText("漏报")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上一张" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "下一张" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "下一张" }));
    expect(await screen.findByText(/2 \/ 3/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "去标注页修正" }));

    expect(window.location.hash).toBe("#annotation");
    expect(await screen.findByText("预测结果叠加")).toBeInTheDocument();
    expect(screen.getByLabelText("标注审查图层")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("预测修正操作")).getByText("类别混淆"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "添加为标注" }));

    expect(screen.getByDisplayValue("false_positive, reviewed_prediction")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "标记为已审查" })[0]);

    expect(
      await screen.findByDisplayValue("occluded, false_negative, reviewed_prediction"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "更多标注操作" }));
    await user.click(screen.getByRole("menuitem", { name: "复制下一张标注" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "展开坐标参数" }));
    fireEvent.change(screen.getByLabelText("X"), { target: { value: "0.42" } });
    await user.click(screen.getByRole("button", { name: "伪装" }));
    await user.click(screen.getByRole("button", { name: "困难负样本" }));
    await user.click(screen.getByRole("button", { name: "保存" }));
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

  it("在项目与数据页面切换导入数据和数据管理", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole("tab", { name: "导入数据" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("数据集来源")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "数据管理" }));

    expect(screen.getByRole("tab", { name: "数据管理" })).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByRole("table", { name: "现有数据" })).toBeInTheDocument();
    expect(screen.queryByLabelText("数据集来源")).not.toBeInTheDocument();
    expect(window.location.hash).toBe("#dataset");
  });

  it("当前数据集移入回收站后清空数据集状态并避免重新请求已删除数据", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    apiMock.createClass.mockRejectedValueOnce(new Error("旧类别错误"));
    await user.click(screen.getByRole("button", { name: "确认启用所选类别" }));
    expect(await screen.findByText("旧类别错误")).toBeInTheDocument();

    await navigateToStep(user, "图像标注");
    await user.type(screen.getByLabelText("文件名搜索"), "frame002");
    await user.selectOptions(screen.getByLabelText("标注状态"), "unannotated");
    await user.click(screen.getByRole("button", { name: /高级筛选/ }));
    fireEvent.change(screen.getByLabelText("平台"), { target: { value: "iris" } });
    await user.click(screen.getByRole("button", { name: "应用筛选" }));

    await navigateToStep(user, "质量与版本");
    apiMock.refreshImageDimensions.mockRejectedValueOnce(new Error("旧质量错误"));
    await user.click(screen.getByRole("button", { name: "刷新图像尺寸" }));
    expect(await screen.findByText("旧质量错误")).toBeInTheDocument();

    await navigateToStep(user, "模型训练");
    await user.click(screen.getByRole("button", { name: "加载配置" }));
    expect(await screen.findByText("已加载训练任务 #1 的配置")).toBeInTheDocument();

    await navigateToStep(user, "项目与数据");
    await user.click(screen.getByRole("tab", { name: "数据管理" }));
    apiMock.listClasses.mockClear();
    apiMock.listImages.mockClear();
    await user.click(await screen.findByRole("button", { name: "移入回收站 camouflage-set" }));
    await user.click(screen.getByRole("button", { name: "确认移入回收站" }));

    expect(apiMock.trashStorageItem).toHaveBeenCalledWith("dataset", 1);
    expect(apiMock.listClasses).not.toHaveBeenCalled();
    expect(apiMock.listImages).not.toHaveBeenCalled();
    expect(screen.getByRole("tab", { name: "数据管理" })).toHaveAttribute("aria-selected", "true");
    expect(
      within(screen.getByRole("navigation", { name: "工作流步骤" })).getByRole("button", { name: /类别管理/ }),
    ).toHaveAttribute("data-availability", "locked");

    await user.click(screen.getByRole("tab", { name: "导入数据" }));
    expect(screen.queryByText(/已将 1 张图像导入/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    expect(screen.getByRole("checkbox", { name: "fire_truck" })).toBeChecked();
    expect(screen.queryByText("旧类别错误")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("编辑类别名称 target")).not.toBeInTheDocument();

    await navigateToStep(user, "图像标注");
    expect(screen.getByLabelText("文件名搜索")).toHaveValue("");
    expect(screen.getByLabelText("标注状态")).toHaveValue("all");
    expect(screen.getByRole("button", { name: "高级筛选" })).toBeInTheDocument();
    expect(screen.queryByLabelText("图像高级筛选")).not.toBeInTheDocument();

    await navigateToStep(user, "质量与版本");
    expect(screen.queryByText("旧质量错误")).not.toBeInTheDocument();
    expect(screen.getByLabelText("质量问题类型")).toHaveValue("all");
    expect(screen.getByText("可以导出")).toBeInTheDocument();

    await navigateToStep(user, "模型训练");
    expect(screen.queryByText("已加载训练任务 #1 的配置")).not.toBeInTheDocument();
    expect(screen.getByLabelText("起始模型")).toHaveValue("base:yolov8n.pt");
  });

  it("删除其他存储对象时完整重新加载当前数据集工作区", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "项目与数据");
    await user.click(screen.getByRole("tab", { name: "数据管理" }));
    apiMock.listClasses.mockClear();
    apiMock.listImages.mockClear();
    apiMock.getQuality.mockClear();
    apiMock.getDatasetCoverage.mockClear();
    apiMock.listQualityIssues.mockClear();
    apiMock.listDatasetVersions.mockClear();
    apiMock.listTrainingRuns.mockClear();
    apiMock.getTrainingRunArtifacts.mockClear();
    apiMock.getTrainingRunSummary.mockClear();
    apiMock.getProjectTrainingSummary.mockClear();
    apiMock.listPredictionJobs.mockClear();
    apiMock.listPredictions.mockClear();
    apiMock.getExportCapabilities.mockClear();
    apiMock.listRunExports.mockClear();
    await user.click(await screen.findByRole("button", { name: "移入回收站 smoke-export" }));
    await user.click(screen.getByRole("button", { name: "确认移入回收站" }));

    expect(apiMock.trashStorageItem).toHaveBeenCalledWith("dataset_version", 1);
    await waitFor(() => {
      expect(apiMock.listClasses).toHaveBeenCalledWith(1);
      expect(apiMock.listImages).toHaveBeenCalledWith(1, {}, { limit: 50, offset: 0 });
      expect(apiMock.getQuality).toHaveBeenCalledWith(1);
      expect(apiMock.getDatasetCoverage).toHaveBeenCalledWith(1);
      expect(apiMock.listQualityIssues).toHaveBeenCalledWith(1, "all");
      expect(apiMock.listDatasetVersions).toHaveBeenCalledWith(1);
      expect(apiMock.listTrainingRuns).toHaveBeenCalledWith(1);
      expect(apiMock.getTrainingRunArtifacts).toHaveBeenCalledWith(1);
      expect(apiMock.getTrainingRunSummary).toHaveBeenCalledWith(1);
      expect(apiMock.getProjectTrainingSummary).toHaveBeenCalledWith(1);
      expect(apiMock.listPredictionJobs).toHaveBeenCalledWith(1);
      expect(apiMock.listPredictions).toHaveBeenCalled();
      expect(apiMock.getExportCapabilities).toHaveBeenCalledWith(1);
      expect(apiMock.listRunExports).toHaveBeenCalledWith(1);
    });
    expect(
      within(screen.getByRole("navigation", { name: "工作流步骤" })).getByRole("button", { name: /类别管理/ }),
    ).not.toHaveAttribute("data-availability", "locked");

    await user.click(screen.getByRole("tab", { name: "导入数据" }));
    expect(screen.getByText(/已将 2 张图像导入/)).toBeInTheDocument();
  });

  it("shows annotation readiness before a dataset is loaded", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("YoloStudio")).toBeInTheDocument();
    await navigateToStep(user, "图像标注");
    expect(screen.getByText("请先导入或加载数据集")).toBeInTheDocument();
    expect(screen.getByLabelText("数据集来源")).toBeInTheDocument();
    expect(screen.queryByLabelText("标注就绪状态")).not.toBeInTheDocument();
  });

  it("以核心工具栏和高级筛选抽屉组织标注操作", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");

    expect(screen.getByLabelText("标注工具栏")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存并下一张" })).toBeInTheDocument();
    expect(screen.queryByLabelText("标注就绪状态")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("图像高级筛选")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /高级筛选/ }));
    expect(screen.getByLabelText("图像高级筛选")).toBeInTheDocument();
    expect(screen.getByLabelText("平台")).toBeInTheDocument();
  });

  it("切换图像后恢复适应窗口倍率", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");

    expect(await screen.findByRole("button", { name: "当前倍率 63%" })).toBeInTheDocument();
    const image = screen.getByRole("img", { name: "iris/frame001.jpg" });
    const canvas = screen.getByLabelText("标注画布");
    expect(image.parentElement).toBe(canvas.parentElement);
    expect(image.parentElement).toHaveClass("annotation-transform-layer");

    await user.click(screen.getByRole("button", { name: "放大图像" }));
    expect(screen.getByRole("button", { name: "当前倍率 75%" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "缩小图像" }));
    expect(screen.getByRole("button", { name: "当前倍率 50%" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "适应窗口" }));
    expect(screen.getByRole("button", { name: "当前倍率 63%" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "显示原始大小" }));
    expect(screen.getByRole("button", { name: "当前倍率 100%" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "下一张图像" }));
    expect(await screen.findByRole("img", { name: "iris/frame002.jpg" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "当前倍率 63%" })).toBeInTheDocument();
  });

  it("以光标为中心滚轮缩放并阻止画布区域滚动", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    const viewport = screen.getByLabelText("标注画布视口");
    Object.defineProperty(viewport, "getBoundingClientRect", {
      configurable: true,
      value: () => ({ left: 20, top: 30, width: 500, height: 300, right: 520, bottom: 330 }),
    });

    const wheel = new WheelEvent("wheel", {
      bubbles: true,
      cancelable: true,
      clientX: 270,
      clientY: 180,
      deltaY: -100,
    });
    fireEvent(viewport, wheel);

    expect(wheel.defaultPrevented).toBe(true);
    expect(screen.getByRole("button", { name: "当前倍率 73%" })).toBeInTheDocument();
    expect(screen.getByTestId("annotation-transform-layer").getAttribute("style")).toContain(
      "--canvas-zoom: 0.726",
    );
  });

  it("空格加左键平移且不会画框或标记未保存", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: "显示原始大小" }));
    const viewport = screen.getByLabelText("标注画布视口");
    const canvas = screen.getByLabelText("标注画布");

    fireEvent.keyDown(window, { key: " ", code: "Space" });
    expect(viewport).toHaveClass("is-space-ready");
    fireEvent(canvas, pointerEvent("pointerdown", 200, 120));
    fireEvent(viewport, pointerEvent("pointermove", 160, 90));

    expect(viewport).toHaveClass("is-panning");
    expect(screen.getByTestId("annotation-transform-layer")).toHaveStyle({
      "--canvas-pan-x": "-110px",
      "--canvas-pan-y": "-120px",
      "--canvas-zoom": "1",
    });
    expect(screen.getByText("已保存", { selector: ".annotation-save-status" })).toBeInTheDocument();

    fireEvent(viewport, pointerEvent("pointercancel", 160, 90));
    expect(viewport).not.toHaveClass("is-panning");
    fireEvent.keyUp(window, { key: " ", code: "Space" });
    expect(viewport).not.toHaveClass("is-space-ready");
    expect(screen.queryByRole("button", { name: /画布边界框/ })).not.toBeInTheDocument();
  });

  it("可编辑控件内的空格不进入平移，窗口失焦会清理平移准备状态", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    const viewport = screen.getByLabelText("标注画布视口");
    const editableElements = [
      screen.getByLabelText("文件名搜索"),
      screen.getByLabelText("当前绘制类别"),
      screen.getByRole("button", { name: "保存" }),
      document.createElement("textarea"),
      document.createElement("div"),
    ];
    editableElements[4].setAttribute("contenteditable", "true");
    viewport.append(editableElements[3], editableElements[4]);

    for (const element of editableElements) {
      fireEvent.keyDown(element, { key: " ", code: "Space" });
      expect(viewport).not.toHaveClass("is-space-ready");
    }
    fireEvent.keyDown(window, { key: " ", code: "Space" });
    expect(viewport).toHaveClass("is-space-ready");
    fireEvent(window, new Event("blur"));
    expect(viewport).not.toHaveClass("is-space-ready");
  });

  it("保存成功后切换到下一张图像", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    expect(await screen.findByRole("img", { name: "iris/frame001.jpg" })).toBeInTheDocument();

    apiMock.listProjects.mockClear();
    await user.click(screen.getByRole("button", { name: "保存并下一张" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(10, []);
    expect(apiMock.listProjects).toHaveBeenCalledTimes(1);
    expect(await screen.findByRole("img", { name: "iris/frame002.jpg" })).toBeInTheDocument();
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
  });

  it("保存失败时保留当前图像和草稿", async () => {
    const user = userEvent.setup();
    apiMock.replaceAnnotations.mockRejectedValueOnce(new Error("磁盘写入失败"));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    expect(await screen.findByRole("img", { name: "iris/frame001.jpg" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存并下一张" }));

    expect((await screen.findAllByText("磁盘写入失败")).length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: "iris/frame001.jpg" })).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "iris/frame002.jpg" })).not.toBeInTheDocument();
  });

  it("保存最后一张图像后提示当前列表已经完成", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: /iris\/frame002\.jpg/ }));
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存并下一张" }));

    expect(await screen.findByText("已完成当前图像列表")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "iris/frame002.jpg" })).toBeInTheDocument();
  });

  it("刷新图像列表时优先选择第一张未标注图像", async () => {
    apiMock.listImages.mockResolvedValueOnce({
      items: [
        { ...apiMock.datasetImages[0], annotation_count: 2 },
        { ...apiMock.datasetImages[2], annotation_count: 0 },
      ],
      limit: 50,
      offset: 0,
      total: 2,
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");

    expect(await screen.findByRole("img", { name: "iris/frame051.jpg" })).toBeInTheDocument();
  });

  it("切换图像前通过应用内确认保护未保存草稿", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: /iris\/frame002\.jpg/ }));
    const trackId = await screen.findByLabelText("目标轨迹 ID");
    await user.type(trackId, "-edited");

    await user.click(screen.getByRole("button", { name: "上一张图像" }));
    expect(screen.getByRole("alert")).toHaveTextContent("切换图像后，这些修改将被放弃");
    expect(screen.getByRole("img", { name: "iris/frame002.jpg" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "留在当前图像" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByDisplayValue("copy-source-edited")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /iris\/frame001\.jpg/ }));
    await user.click(screen.getByRole("button", { name: "放弃修改并切换" }));
    expect(await screen.findByRole("img", { name: "iris/frame001.jpg" })).toBeInTheDocument();
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(10);
  });

  it("下一张按钮也不会跳过未保存草稿确认", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: "更多标注操作" }));
    await user.click(screen.getByRole("menuitem", { name: "复制下一张标注" }));
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "下一张图像" }));

    expect(screen.getByRole("alert")).toHaveTextContent("切换图像后，这些修改将被放弃");
    expect(screen.getByRole("img", { name: "iris/frame001.jpg" })).toBeInTheDocument();
  });

  it("移动端标签切换画布、图像和属性 pane 时保留编辑状态", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    const tabs = screen.getByRole("tablist", { name: "移动端标注视图" });
    expect(within(tabs).getByRole("tab", { name: "画布" })).toHaveAttribute("aria-selected", "true");

    await user.click(within(tabs).getByRole("tab", { name: "图像" }));
    expect(screen.getByLabelText("图像浏览器")).toHaveClass("active");
    await user.click(screen.getByRole("button", { name: /iris\/frame002\.jpg/ }));

    await user.click(within(tabs).getByRole("tab", { name: "属性" }));
    const trackId = await screen.findByLabelText("目标轨迹 ID");
    await user.type(trackId, "-mobile");
    await user.click(within(tabs).getByRole("tab", { name: "画布" }));
    await user.click(within(tabs).getByRole("tab", { name: "属性" }));

    expect(screen.getByDisplayValue("copy-source-mobile")).toBeInTheDocument();
    expect(screen.getByLabelText("边界框检查器")).toHaveClass("active");
  });

  it("移动端保留全部缩放控件并使用滚动画布浏览放大图像", async () => {
    vi.stubGlobal("matchMedia", vi.fn().mockImplementation((query: string) => ({
      matches: query === "(max-width: 820px)",
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })));
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");

    const toolbar = screen.getByRole("toolbar", { name: "画布缩放" });
    expect(within(toolbar).getByRole("button", { name: "缩小图像" })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: /当前倍率/ })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: "放大图像" })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: "显示原始大小" })).toBeInTheDocument();
    expect(within(toolbar).getByRole("button", { name: "适应窗口" })).toBeInTheDocument();

    const viewport = screen.getByLabelText("标注画布视口");
    fireEvent.keyDown(window, { key: " ", code: "Space" });
    expect(viewport).not.toHaveClass("is-space-ready");

    await user.click(within(toolbar).getByRole("button", { name: "显示原始大小" }));
    expect(screen.getByTestId("annotation-canvas-scroll-content")).toHaveStyle({
      width: "640px",
      height: "480px",
    });

    const wheel = new WheelEvent("wheel", { bubbles: true, cancelable: true, deltaY: -100 });
    fireEvent(viewport, wheel);
    expect(wheel.defaultPrevented).toBe(false);
    expect(screen.queryByText(/双指/)).not.toBeInTheDocument();
  });

  it("guides empty class libraries and selects a created class", async () => {
    const user = userEvent.setup();
    apiMock.listClasses.mockImplementation(async () => ({ items: [] }));
    apiMock.createClass.mockImplementationOnce(async (_projectId, body) => ({
      id: 2,
      project_id: 1,
      name: body.name,
      color: body.color,
      description: null,
      active: true,
      annotation_count: 0,
      version_count: 0,
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    expect(window.location.hash).toBe("#classes");

    for (const name of ["person_white", "prius_hybrid", "car_lexus", "prius_hybrid_camo", "suv_camo", "car_opel", "person_red"]) {
      await user.click(screen.getByRole("checkbox", { name }));
    }
    await user.click(screen.getByRole("button", { name: "确认确认启用所选类别" }));

    expect(apiMock.createClass).toHaveBeenCalledWith(1, {
      name: "fire_truck",
      color: "#e45756",
    });
    expect(window.location.hash).toBe("#annotation");
    expect(screen.queryByLabelText("标注就绪状态")).not.toBeInTheDocument();
    expect(screen.getByText("在图像上拖动以添加边界框。")).toBeInTheDocument();
  });

  it("扫描类别默认全部选择并批量启用", async () => {
    const user = userEvent.setup();
    apiMock.listClasses.mockImplementation(async () => ({ items: [] }));
    let nextClassId = 10;
    apiMock.createClass.mockImplementation(async (_projectId, body) => ({
      id: nextClassId++,
      project_id: 1,
      name: body.name,
      color: body.color,
      description: null,
      active: true,
      annotation_count: 0,
      version_count: 0,
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");

    expect(apiMock.listDetectedClasses).toHaveBeenCalledWith(1);
    for (const name of ["fire_truck", "person_white", "prius_hybrid", "car_lexus", "prius_hybrid_camo", "suv_camo", "car_opel", "person_red"]) {
      expect(screen.getByRole("checkbox", { name })).toBeChecked();
    }

    await user.click(screen.getByRole("checkbox", { name: "suv_camo" }));
    await user.click(screen.getByRole("button", { name: "确认确认启用所选类别" }));

    await waitFor(() => expect(apiMock.createClass).toHaveBeenCalledTimes(7));
    expect(apiMock.createClass).not.toHaveBeenCalledWith(1, {
      name: "suv_camo",
      color: "#1f6f78",
    });
    expect(apiMock.createClass).toHaveBeenCalledWith(1, {
      name: "fire_truck",
      color: "#e45756",
    });
    expect(window.location.hash).toBe("#annotation");
  });

  it("扫描类别不会重复创建项目中已有的类别", async () => {
    const user = userEvent.setup();
    let nextClassId = 10;
    apiMock.listClasses.mockImplementation(async () => ({
      items: [
        {
          id: 8,
          project_id: 1,
          name: "fire_truck",
          color: "#e45756",
          description: null,
          active: true,
          annotation_count: 0,
          version_count: 0,
        },
      ],
    }));
    apiMock.createClass.mockImplementation(async (_projectId, body) => ({
      id: nextClassId++,
      project_id: 1,
      name: body.name,
      color: body.color,
      description: null,
      active: true,
      annotation_count: 0,
      version_count: 0,
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    expect(screen.getByText("已在类别库")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "确认启用所选类别" }));

    await waitFor(() => expect(apiMock.createClass).toHaveBeenCalledTimes(7));
    expect(apiMock.createClass).not.toHaveBeenCalledWith(1, {
      name: "fire_truck",
      color: "#e45756",
    });
  });

  it("类别删除需要确认，确认后从类别库移除", async () => {
    const user = userEvent.setup();
    apiMock.deleteClass.mockResolvedValueOnce(apiMock.defaultClasses()[0]);
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    await user.click(screen.getByRole("button", { name: "删除 target" }));

    const dialog = screen.getByRole("dialog", { name: "确认删除类别" });
    expect(dialog).toHaveTextContent("确定删除类别“target”吗");
    await user.click(within(dialog).getByRole("button", { name: "取消" }));
    expect(screen.queryByRole("dialog", { name: "确认删除类别" })).not.toBeInTheDocument();
    expect(apiMock.deleteClass).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "删除 target" }));
    await user.click(
      within(screen.getByRole("dialog", { name: "确认删除类别" })).getByRole("button", {
        name: "确认删除",
      }),
    );

    expect(apiMock.deleteClass).toHaveBeenCalledWith(1, 1);
    expect(screen.queryByRole("button", { name: "target" })).not.toBeInTheDocument();
  });

  it("已被标注或数据集版本使用的类别不能删除", async () => {
    const user = userEvent.setup();
    apiMock.listClasses.mockImplementation(async () => ({
      items: [
        {
          ...apiMock.defaultClasses()[0],
          annotation_count: 3,
          version_count: 1,
        },
      ],
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");

    const deleteButton = screen.getByRole("button", { name: "删除 target" });
    expect(deleteButton).toBeDisabled();
    expect(deleteButton).toHaveAttribute(
      "title",
      "已被 3 条标注或 1 个数据集版本使用，不能删除",
    );
  });

  it("导入数据集后进入类别管理，创建首个类别后进入图像标注", async () => {
    const user = userEvent.setup();
    apiMock.listClasses.mockImplementation(async () => ({ items: [] }));
    apiMock.createClass.mockImplementationOnce(async (_projectId, body) => ({
      id: 2,
      project_id: 1,
      name: body.name,
      color: body.color,
      description: null,
      active: true,
      annotation_count: 0,
      version_count: 0,
    }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "导入数据集" }));
    expect(await screen.findByRole("heading", { name: "类别库" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#classes");

    for (const name of ["person_white", "prius_hybrid", "car_lexus", "prius_hybrid_camo", "suv_camo", "car_opel", "person_red"]) {
      await user.click(screen.getByRole("checkbox", { name }));
    }
    await user.click(screen.getByRole("button", { name: "确认启用所选类别" }));
    expect(await screen.findByRole("heading", { name: "标注" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#annotation");
  });

  it("通过 hash 恢复开放步骤，拒绝锁定步骤并回退无效 hash", async () => {
    const user = userEvent.setup();
    apiMock.listDatasetVersions.mockImplementationOnce(async () => ({ items: [] }));
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    expect(window.location.hash).toBe("#classes");

    window.history.replaceState(null, "", "#quality");
    fireEvent(window, new Event("hashchange"));
    expect(await screen.findByRole("heading", { name: "质量审查" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#quality");

    window.location.hash = "#training";
    fireEvent(window, new Event("hashchange"));
    expect(screen.getByRole("heading", { name: "质量审查" })).toBeInTheDocument();
    expect(screen.getByText("请先创建冻结数据集版本")).toBeInTheDocument();
    expect(window.location.hash).toBe("#quality");

    window.location.hash = "#unknown";
    fireEvent(window, new Event("hashchange"));
    expect(await screen.findByLabelText("数据集来源")).toBeInTheDocument();
    expect(window.location.hash).toBe("#dataset");
  });

  it("离开标注页前确认未保存修改，并保护 hash 导航", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: /iris\/frame002\.jpg/ }));
    expect(await screen.findByDisplayValue("copy-source")).toBeInTheDocument();

    await user.type(screen.getByLabelText("目标轨迹 ID"), "-edited");
    await navigateToStep(user, "质量与版本");

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("当前图像的标注尚未保存");
    expect(screen.getByRole("heading", { name: "标注" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#annotation");

    await user.click(screen.getByRole("button", { name: "留在标注页" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    window.history.replaceState(null, "", "#quality");
    fireEvent(window, new Event("hashchange"));
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(window.location.hash).toBe("#annotation");

    await user.click(screen.getByRole("button", { name: "放弃修改并离开" }));
    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(await screen.findByRole("heading", { name: "质量审查" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#quality");
  });

  it("刷新到锁定步骤时回退项目与数据", async () => {
    window.history.replaceState(null, "", "#annotation");
    render(<App />);

    expect(await screen.findByLabelText("数据集来源")).toBeInTheDocument();
    expect(screen.getByText("请先导入或加载数据集")).toBeInTheDocument();
    expect(window.location.hash).toBe("#dataset");
  });

  it("edits a project class and refreshes dependent annotation labels", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "编辑 target" }));
    await user.clear(screen.getByLabelText("编辑类别名称 target"));
    await user.type(screen.getByLabelText("编辑类别名称 target"), "vehicle");
    fireEvent.change(screen.getByLabelText("编辑类别颜色 target"), {
      target: { value: "#22c55e" },
    });
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(apiMock.updateClass).toHaveBeenCalledWith(1, 1, {
      name: "vehicle",
      color: "#22c55e",
    });

    const classLibrary = screen.getByLabelText("可用类别");
    expect(await within(classLibrary).findByRole("button", { name: "vehicle" })).toHaveClass(
      "selected",
    );
    expect(screen.queryByRole("button", { name: "target" })).not.toBeInTheDocument();

    await navigateToStep(user, "质量与版本");
    const versionSubset = screen.getByLabelText("版本类别子集");
    expect(within(versionSubset).getByRole("checkbox", { name: "vehicle" })).toBeChecked();

    await navigateToStep(user, "图像标注");
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      screen.getByText("vehicle", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();
  });

  it("changes an existing annotation box class before saving", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "类别管理");
    apiMock.createClass.mockImplementationOnce(async (_projectId, body) => ({
      id: 2,
      project_id: 1,
      name: body.name,
      color: body.color,
      description: null,
      active: true,
      annotation_count: 0,
      version_count: 0,
    }));
    for (const name of ["person_white", "prius_hybrid", "car_lexus", "prius_hybrid_camo", "suv_camo", "car_opel", "person_red"]) {
      await user.click(screen.getByRole("checkbox", { name }));
    }
    await user.click(screen.getByRole("button", { name: "确认启用所选类别" }));
    expect(await screen.findByRole("button", { name: "fire_truck" })).toBeInTheDocument();

    await navigateToStep(user, "图像标注");
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("边界框 1 类别"), "2");
    expect(
      screen.getByText("fire_truck", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存" }));

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

  it("只在检查器中展开当前边界框", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(await screen.findByText("iris/frame002.jpg"));

    expect(screen.getByLabelText("边界框检查器")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /选择边界框/ })).toHaveLength(1);
    expect(screen.queryByLabelText("归一化坐标")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /选择边界框/ }));
    await user.click(screen.getByRole("button", { name: "展开坐标参数" }));
    expect(screen.getByLabelText("归一化坐标")).toBeInTheDocument();
  });

  it("moves an existing annotation box from the canvas and nudge controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    const canvas = screen.getByLabelText("标注画布");
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
    await user.click(screen.getByRole("button", { name: "向左移动边界框 1" }));

    await user.click(screen.getByRole("button", { name: "保存" }));

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

  it("resizes an existing annotation box from the canvas handles", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(await screen.findByText("iris/frame002.jpg"));
    expect(
      await screen.findByText("target", { selector: ".box-editor-title strong" }),
    ).toBeInTheDocument();

    const canvas = screen.getByLabelText("标注画布");
    Object.defineProperty(canvas, "getBoundingClientRect", {
      configurable: true,
      value: () => ({ left: 0, top: 0, width: 1000, height: 1000, right: 1000, bottom: 1000 }),
    });

    const canvasBox = screen.getByTestId("annotation-box-annotation-101");
    fireEvent(canvasBox, pointerEvent("pointerdown", 600, 550));
    await act(async () => {
      fireEvent(window, pointerEvent("pointerup", 600, 550));
    });

    const resizeHandle = await screen.findByTestId("resize-handle-annotation-101-bottom-right");
    const annotationFrame = screen.getByTestId("annotation-frame-annotation-101");
    expect(annotationFrame).toHaveAttribute("stroke-width", "3");
    expect(Number(resizeHandle.getAttribute("width")) * 640 * 0.625).toBeCloseTo(10);
    expect(Number(resizeHandle.getAttribute("height")) * 480 * 0.625).toBeCloseTo(10);
    expect(resizeHandle).toHaveAttribute("fill", "#ef4444");
    expect(resizeHandle).toHaveAttribute("stroke", "#ffffff");
    expect(resizeHandle).toHaveAttribute("stroke-width", "2");
    fireEvent(resizeHandle, pointerEvent("pointerdown", 725, 650));
    await act(async () => {
      fireEvent(window, pointerEvent("pointermove", 875, 750));
      fireEvent(window, pointerEvent("pointerup", 875, 750));
    });

    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      11,
      expect.arrayContaining([
        expect.objectContaining({
          class_id: 1,
          x_center: 0.675,
          y_center: 0.6,
          width: 0.4,
          height: 0.3,
          track_id: "copy-source",
          edge_tags: ["occluded"],
        }),
      ]),
    );
  });

  it("缩放后新建边界框仍保存正确的归一化坐标", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: "显示原始大小" }));
    const canvas = screen.getByLabelText("标注画布");
    Object.defineProperty(canvas, "getBoundingClientRect", {
      configurable: true,
      value: () => ({ left: -70, top: -90, width: 640, height: 480, right: 570, bottom: 390 }),
    });

    fireEvent(canvas, pointerEvent("pointerdown", 122, 54));
    fireEvent(canvas, pointerEvent("pointermove", 250, 150));
    fireEvent(canvas, pointerEvent("pointerup", 250, 150));
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      10,
      expect.arrayContaining([
        expect.objectContaining({
          x_center: 0.4,
          y_center: 0.4,
          width: 0.2,
          height: 0.2,
        }),
      ]),
    );
  });

  it("缩放后移动和调整边界框仍使用图像归一化坐标", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "图像标注");
    await user.click(screen.getByRole("button", { name: /iris\/frame002\.jpg/ }));
    await screen.findByText("target", { selector: ".box-editor-title strong" });
    await user.click(screen.getByRole("button", { name: "显示原始大小" }));
    await user.click(screen.getByRole("button", { name: "放大图像" }));
    const canvas = screen.getByLabelText("标注画布");
    Object.defineProperty(canvas, "getBoundingClientRect", {
      configurable: true,
      value: () => ({ left: -230, top: -210, width: 960, height: 720, right: 730, bottom: 510 }),
    });

    const canvasBox = screen.getByTestId("annotation-box-annotation-101");
    fireEvent(canvasBox, pointerEvent("pointerdown", 346, 186));
    fireEvent(window, pointerEvent("pointermove", 442, 258));
    fireEvent(window, pointerEvent("pointerup", 442, 258));

    const resizeHandle = await screen.findByTestId("resize-handle-annotation-101-bottom-right");
    expect(Number(resizeHandle.getAttribute("width")) * 640 * 1.5).toBeCloseTo(10);
    expect(Number(resizeHandle.getAttribute("height")) * 480 * 1.5).toBeCloseTo(10);
    fireEvent(resizeHandle, pointerEvent("pointerdown", 562, 330));
    fireEvent(window, pointerEvent("pointermove", 658, 402));
    fireEvent(window, pointerEvent("pointerup", 658, 402));
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(apiMock.replaceAnnotations).toHaveBeenLastCalledWith(
      11,
      expect.arrayContaining([
        expect.objectContaining({
          x_center: 0.75,
          y_center: 0.7,
          width: 0.35,
          height: 0.3,
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

    fireEvent.click(screen.getByRole("button", { name: "导入数据集" }));
    await flushPromises();

    fireEvent.click(screen.getByRole("button", { name: /模型训练/ }));
    expect(screen.getByText("训练任务 #2")).toBeInTheDocument();
    expect(screen.getByText("自动刷新中")).toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(2500);
      await Promise.resolve();
    });
    await flushPromises();

    expect(screen.getByText("训练任务 #1")).toBeInTheDocument();
    expect(screen.getByText("metrics/mAP50(B): 0.420")).toBeInTheDocument();
    expect(apiMock.getTrainingRunLogs).toHaveBeenCalledWith(2);
    expect(apiMock.getPredictionJobLogs).toHaveBeenCalledWith(2);
    expect(screen.getByText("空闲")).toBeInTheDocument();
  });

  it("loads a historical run config and reruns it on the latest version", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "模型训练");
    expect(await screen.findByText("训练任务 #1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "加载配置" }));

    expect(screen.getByLabelText("起始模型")).toHaveValue("custom-drone.pt");
    expect(screen.getByLabelText("训练轮数")).toHaveValue(12);
    expect(screen.getByLabelText("图像尺寸")).toHaveValue(512);
    expect(screen.getByLabelText("批大小")).toHaveValue(4);
    expect(screen.getByLabelText("计算设备")).toHaveValue("mps");
    expect(screen.getByLabelText("策略名称")).toHaveValue("edge-case");
    expect(screen.getByLabelText("Mosaic")).toHaveValue(0.8);
    expect(screen.getByLabelText("MixUp")).toHaveValue(0.15);
    expect(screen.getByRole("checkbox", { name: "GridMask" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "TTA" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "自动阈值扫描" })).toBeChecked();
    expect(await screen.findByText("已加载训练任务 #1 的配置")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "重新训练" }));

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
    expect(await screen.findByText("已基于训练任务 #1 开始重新训练")).toBeInTheDocument();
  });

  it("loads a saved dataset without re-importing source files", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByLabelText("已保存数据集")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "加载数据集" }));

    expect(apiMock.importDataset).not.toHaveBeenCalled();
    expect(apiMock.listImages).toHaveBeenCalledWith(1, {}, { limit: 50, offset: 0 });
    expect(window.location.hash).toBe("#classes");
    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    await navigateToStep(user, "项目与数据");
    const summaries = await screen.findAllByText((_, element) =>
      Boolean(
        element?.classList.contains("summary-line") &&
          element.textContent?.includes("Drone QA Project / camouflage-set"),
      ),
    );
    expect(summaries).toHaveLength(1);
    await navigateToStep(user, "图像标注");
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
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

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "质量与版本");

    expect(await screen.findByText("需要处理")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量指标")).getByText("重复边界框"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量问题样本")).getByText("重复边界框"),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("边界框与同一图像、同一类别中的标注 #1 重复。"),
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("质量问题类型"), "duplicate_box");

    expect(apiMock.listQualityIssues).toHaveBeenLastCalledWith(1, "duplicate_box");

    await user.click(screen.getByRole("button", { name: "应用自动标签" }));

    expect(apiMock.applyQualityTags).toHaveBeenCalledWith(1, "duplicate_box");
    expect(await screen.findByText("已向 1 个标注应用 1 个标签")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "打开问题图像" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(window.location.hash).toBe("#annotation");
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

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "质量与版本");

    expect(await screen.findByText("可以导出")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量指标")).getByText("缺少元数据"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量问题样本")).getByText("缺少元数据"),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("图像缺少来源元数据行、高度、时间戳。"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "打开问题图像" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(window.location.hash).toBe("#annotation");
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

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "质量与版本");

    expect(await screen.findByText("可以导出")).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量指标")).getByText("缺少图像尺寸"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByLabelText("质量问题样本")).getByText("缺少图像尺寸"),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("无法读取图像宽度或高度。"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "打开问题图像" }));

    expect(apiMock.getAnnotations).toHaveBeenLastCalledWith(11);
    expect(window.location.hash).toBe("#annotation");
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

    await user.click(await screen.findByRole("button", { name: "加载数据集" }));
    await navigateToStep(user, "质量与版本");
    expect(
      within(screen.getByLabelText("质量指标")).getByText("缺少图像尺寸"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "刷新图像尺寸" }));

    expect(apiMock.refreshImageDimensions).toHaveBeenCalledWith(1);
    expect(await screen.findByText("已扫描 2 张，已更新 1 张，仍缺失 0 张")).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("button", { name: "导入数据集" }));
    await flushPromises();

    fireEvent.click(screen.getByRole("button", { name: /模型训练/ }));
    expect(screen.getByText("训练任务 #2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "取消训练" }));
    await flushPromises();

    expect(apiMock.cancelTrainingRun).toHaveBeenCalledWith(2);
    expect(screen.getByText("已取消")).toBeInTheDocument();
    expect(apiMock.getTrainingRunLogs).toHaveBeenCalledWith(2);
  });
});

async function navigateToStep(
  user: ReturnType<typeof userEvent.setup>,
  label: string,
) {
  const workflowNavigation = screen.getByRole("navigation", { name: "工作流步骤" });
  await user.click(
    within(workflowNavigation).getByRole("button", { name: new RegExp(label) }),
  );
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}
