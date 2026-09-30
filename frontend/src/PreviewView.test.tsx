import "@testing-library/jest-dom/vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PreviewView } from "./PreviewView";

const apiMock = vi.hoisted(() => ({
  listPreviewModels: vi.fn(async () => ({
    items: [
      { model_ref: "base:yolov8n.pt", label: "内置模型 · YOLOv8n", kind: "base", run_id: null, status: "可用" },
      { model_ref: "run:7", label: "训练任务 #7 · best.pt", kind: "trained", run_id: 7, status: "completed" },
    ],
  })),
  listAllImages: vi.fn(async () => [
    {
      id: 101,
      relative_path: "frame001.jpg",
      width: 640,
      height: 480,
      platform: "test",
      altitude: null,
      timestamp: null,
      annotation_count: 1,
      annotation_status: "annotated",
      image_url: "/api/images/101/file",
    },
    {
      id: 102,
      relative_path: "frame002.jpg",
      width: 640,
      height: 480,
      platform: "test",
      altitude: null,
      timestamp: null,
      annotation_count: 0,
      annotation_status: "pending",
      image_url: "/api/images/102/file",
    },
    {
      id: 103,
      relative_path: "frame003.jpg",
      width: 640,
      height: 480,
      platform: "test",
      altitude: null,
      timestamp: null,
      annotation_count: 0,
      annotation_status: "pending",
      image_url: "/api/images/103/file",
    },
  ]),
  previewImage: vi.fn(async (imageId: number) => ({
    image_id: imageId,
    filename: `frame00${imageId - 100}.jpg`,
    image_url: `/api/images/${imageId}/file`,
    width: 640,
    height: 480,
    model_ref: "base:yolov8n.pt",
    confidence_threshold: 0.25,
    annotations: [{ class_id: 1, class_name: "target", color: "#22a06b", x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2, confidence: null }],
    predictions: [{ class_id: 0, class_name: "person", color: "#f59e0b", x_center: 0.5, y_center: 0.5, width: 0.3, height: 0.25, confidence: 0.91 }],
  })),
  listPreviewVideos: vi.fn(async () => ({
    items: [
      {
        id: 9,
        project_id: 1,
        original_filename: "drone-clip.mp4",
        size_bytes: 2048,
        created_at: "2026-07-24T02:00:00Z",
      },
    ],
  })),
  listVideoPreviewJobs: vi.fn(async () => ({ items: [] })),
  uploadPreviewVideo: vi.fn(),
  createVideoPreview: vi.fn(async (_body: FormData) => ({
    id: 44,
    project_id: 1,
    dataset_id: null,
    run_id: null,
    video_id: 9,
    model_ref: "run:7",
    kind: "video",
    source_filename: "drone-clip.mp4",
    status: "queued",
    confidence_threshold: 0.25,
    frame_step: 1,
    fps: null,
    total_frames: 0,
    processed_frames: 0,
    error_message: null,
    result_url: null,
    stream_url: "/api/preview/video-jobs/44/stream",
    created_at: "2026-07-24T03:00:00Z",
    started_at: null,
    ended_at: null,
  })),
  getVideoPreviewStatus: vi.fn(),
  deletePreviewVideo: vi.fn(),
  deleteVideoPreview: vi.fn(),
}));

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return { ...actual, ...apiMock };
});

describe("PreviewView", () => {
  beforeEach(() => {
    apiMock.previewImage.mockClear();
    apiMock.createVideoPreview.mockClear();
    apiMock.listPreviewVideos.mockReset();
    apiMock.listVideoPreviewJobs.mockReset();
    apiMock.listPreviewVideos.mockResolvedValue({
      items: [
        {
          id: 9,
          project_id: 1,
          original_filename: "drone-clip.mp4",
          size_bytes: 2048,
          created_at: "2026-07-24T02:00:00Z",
        },
      ],
    });
    apiMock.listVideoPreviewJobs.mockResolvedValue({ items: [] });
    apiMock.createVideoPreview.mockResolvedValue({
      id: 44,
      project_id: 1,
      dataset_id: null,
      run_id: null,
      video_id: 9,
      model_ref: "run:7",
      kind: "video",
      source_filename: "drone-clip.mp4",
      status: "queued",
      confidence_threshold: 0.25,
      frame_step: 1,
      fps: null,
      total_frames: 0,
      processed_frames: 0,
      error_message: null,
      result_url: null,
      stream_url: "/api/preview/video-jobs/44/stream",
      created_at: "2026-07-24T03:00:00Z",
      started_at: null,
      ended_at: null,
    });
  });

  it("选择数据集后加载第一张图像和预测框", async () => {
    const user = userEvent.setup();
    render(
      <PreviewView
        projectId={1}
        datasets={[{
          id: 3,
          project_id: 1,
          name: "演示集",
          source_type: "zip",
          import_status: "imported",
          image_count: 1,
          annotated_image_count: 1,
          annotation_count: 1,
        }]}
        runs={[]}
        classes={[]}
      />,
    );

    expect(await screen.findByRole("img", { name: "frame001.jpg" })).toBeInTheDocument();
    expect(apiMock.previewImage).toHaveBeenCalledWith(101, {
      model_ref: "base:yolov8n.pt",
      confidence_threshold: 0.25,
    });
    expect(screen.getByRole("img", { name: "frame001.jpg" })).toBeVisible();

    expect(screen.getByLabelText("标注：target")).toHaveStyle({ "--box-color": "#22a06b" });
    expect(screen.getByLabelText("预测：person")).toHaveStyle({ "--box-color": "#f59e0b" });

    const predictionToggle = screen.getAllByRole("checkbox")[0];
    await user.click(predictionToggle);
    await waitFor(() => expect(predictionToggle).not.toBeChecked());
  });

  it("可以切换到训练后的模型", async () => {
    const user = userEvent.setup();
    render(<PreviewView projectId={1} datasets={[]} runs={[]} classes={[]} />);
    await user.selectOptions(await screen.findByLabelText("预览模型"), "run:7");
    expect(apiMock.previewImage).not.toHaveBeenCalled();
  });

  it("可以直接输入张号跳转", async () => {
    const user = userEvent.setup();
    render(
      <PreviewView
        projectId={1}
        datasets={[{
          id: 3,
          project_id: 1,
          name: "演示集",
          source_type: "zip",
          import_status: "imported",
          image_count: 3,
          annotated_image_count: 1,
          annotation_count: 1,
        }]}
        runs={[]}
        classes={[]}
      />,
    );

    expect(await screen.findByRole("img", { name: "frame001.jpg" })).toBeInTheDocument();
    const jumpInput = screen.getByLabelText("跳转到第几张");
    await user.clear(jumpInput);
    await user.type(jumpInput, "3{Enter}");
    expect(await screen.findByRole("img", { name: "frame003.jpg" })).toBeInTheDocument();
    expect(apiMock.previewImage).toHaveBeenCalledWith(103, {
      model_ref: "base:yolov8n.pt",
      confidence_threshold: 0.25,
    });
    expect(jumpInput).toHaveValue(3);
  });

  it("可从视频库选择已保存视频开始回测", async () => {
    const user = userEvent.setup();
    render(
      <PreviewView
        projectId={1}
        datasets={[]}
        runs={[{
          id: 7,
          project_id: 1,
          version_id: 1,
          status: "completed",
          device: "cuda",
          config: {},
          artifact_path: "",
          log_path: "",
          error_message: null,
          latest_metrics: {},
          started_at: null,
          ended_at: null,
          created_at: "2026-07-24T01:00:00Z",
          updated_at: "2026-07-24T01:00:00Z",
        }]}
        classes={[]}
      />,
    );

    await user.click(await screen.findByRole("tab", { name: /视频实时预览/ }));
    await waitFor(() => expect(apiMock.listPreviewVideos).toHaveBeenCalledWith(1));
    expect(await screen.findByRole("cell", { name: /2\.00 KB/ })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /drone-clip\.mp4/ })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("已保存视频"), "9");
    await user.selectOptions(screen.getByLabelText("预览模型"), "run:7");
    await user.click(screen.getByRole("button", { name: "开始回测" }));

    await waitFor(() => expect(apiMock.createVideoPreview).toHaveBeenCalledTimes(1));
    const form = apiMock.createVideoPreview.mock.calls[0][0];
    expect(form.get("video_id")).toBe("9");
    expect(form.get("model_ref")).toBe("run:7");
    expect(form.get("project_id")).toBe("1");
  });
});
