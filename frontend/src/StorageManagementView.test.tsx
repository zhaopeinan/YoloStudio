import "@testing-library/jest-dom/vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { StorageManagementView } from "./StorageManagementView";

const apiMock = vi.hoisted(() => ({
  getStorageItem: vi.fn(),
  listStorageItems: vi.fn(),
  listTrashItems: vi.fn(),
  purgeExpiredTrash: vi.fn(),
  purgeTrashItem: vi.fn(),
  restoreTrashItem: vi.fn(),
  trashStorageItem: vi.fn(),
}));

vi.mock("./api", () => apiMock);

const dataset = {
  entity_type: "dataset" as const,
  entity_id: 1,
  display_name: "camouflage-set",
  project_id: 1,
  project_name: "无人机项目",
  dataset_id: 1,
  version_id: null,
  artifact_path: "/workspace/projects/1/datasets/1",
  size_bytes: 2048,
  image_count: 2,
  annotation_count: 2,
  split_counts: null,
  created_at: "2026-07-15T08:00:00Z",
  protected: true,
  blockers: [
    {
      entity_type: "dataset_version" as const,
      entity_id: 4,
      display_name: "标注版本 4",
      status: null,
    },
  ],
};

const version = {
  entity_type: "dataset_version" as const,
  entity_id: 4,
  display_name: "标注版本 4",
  project_id: 1,
  project_name: "无人机项目",
  dataset_id: 1,
  version_id: 4,
  artifact_path: "/workspace/projects/1/versions/4",
  size_bytes: 4096,
  image_count: 2,
  annotation_count: 2,
  split_counts: { train: 1, val: 1, test: 0 },
  created_at: "2026-07-15T09:00:00Z",
  protected: true,
  blockers: [
    {
      entity_type: "training_run" as const,
      entity_id: 9,
      display_name: "训练任务 #9",
      status: "completed",
    },
  ],
};

const versionDetail = {
  ...version,
  class_names: ["伪装无人机", "车辆"],
  related_runs: [
    {
      id: 9,
      status: "completed",
      model: "yolov8n.pt",
      size_bytes: 8192,
      prediction_job_count: 2,
      export_count: 1,
      created_at: "2026-07-15T10:00:00Z",
    },
    {
      id: 10,
      status: "running",
      model: "yolov10n.pt",
      size_bytes: 1024,
      prediction_job_count: 0,
      export_count: 0,
      created_at: "2026-07-15T11:00:00Z",
    },
  ],
};

const trashItem = {
  id: 21,
  entity_type: "dataset_version" as const,
  entity_id: 3,
  display_name: "旧标注版本",
  project_id: 1,
  dataset_id: 1,
  version_id: 3,
  original_path: "/workspace/projects/1/versions/3",
  trash_path: "/workspace/.trash/21",
  size_bytes: 6144,
  summary: {
    image_count: 12,
    annotation_count: 19,
    split_counts: { train: 9, val: 2, test: 1 },
    class_names: ["伪装无人机"],
  },
  status: "active",
  error_message: null,
  deleted_at: "2026-07-15T08:00:00Z",
  purge_after: "2026-08-14T08:00:00Z",
};

const errorTrashItem = {
  ...trashItem,
  id: 22,
  entity_id: 4,
  display_name: "异常标注版本",
  trash_path: "/workspace/.trash/22",
  status: "error",
  error_message: "回收站目录与数据库记录不一致，请检查磁盘目录。",
};

const pendingMoveTrashItem = {
  ...trashItem,
  id: 23,
  entity_id: 5,
  display_name: "正在移入的版本",
  trash_path: "/workspace/.trash/23",
  status: "pending_move",
};

const pendingRestoreTrashItem = {
  ...trashItem,
  id: 24,
  entity_id: 6,
  display_name: "正在恢复的版本",
  trash_path: "/workspace/.trash/24",
  status: "pending_restore",
};

function arrangeApi() {
  apiMock.listStorageItems.mockResolvedValue({
    items: [dataset, version],
    total_size_bytes: dataset.size_bytes + version.size_bytes,
  });
  apiMock.listTrashItems.mockResolvedValue({ items: [trashItem], total_size_bytes: 6144 });
  apiMock.getStorageItem.mockResolvedValue(versionDetail);
  apiMock.trashStorageItem.mockResolvedValue(trashItem);
  apiMock.restoreTrashItem.mockResolvedValue({
    trash_id: 21,
    entity_type: "dataset_version",
    entity_id: 3,
    status: "restored",
    message: "已恢复",
  });
  apiMock.purgeTrashItem.mockResolvedValue({
    trash_id: 21,
    entity_type: "dataset_version",
    entity_id: 3,
    status: "purged",
    message: "已彻底删除",
  });
  apiMock.purgeExpiredTrash.mockResolvedValue({ purged_count: 0, failed_count: 0 });
}

describe("StorageManagementView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    arrangeApi();
  });

  it("展示现有数据、保护原因和详细信息", async () => {
    const user = userEvent.setup();
    render(
      <StorageManagementView
        loadedDatasetId={1}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    const table = await screen.findByRole("table", { name: "现有数据" });
    expect(within(table).getByText("camouflage-set")).toBeInTheDocument();
    expect(within(table).getAllByText("2 张图像 / 2 个标注")).toHaveLength(2);
    expect(within(table).getAllByText("受保护")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "移入回收站 camouflage-set" })).toBeDisabled();
    expect(screen.getAllByText("标注版本 4")).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "查看 标注版本 4" }));
    const drawer = await screen.findByRole("dialog", { name: "数据详情" });
    expect(within(drawer).getByText("伪装无人机")).toBeInTheDocument();
    expect(within(drawer).getByText("训练 1 / 验证 1 / 测试 0")).toBeInTheDocument();
    expect(within(drawer).getByTitle(version.artifact_path)).toHaveTextContent(version.artifact_path);
    expect(within(drawer).getByRole("button", { name: "移入回收站 训练任务 #10" })).toBeDisabled();
  });

  it("可以从详情抽屉清理已结束训练任务", async () => {
    const user = userEvent.setup();
    const onStorageChanged = vi.fn();
    render(
      <StorageManagementView
        loadedDatasetId={1}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={onStorageChanged}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "查看 标注版本 4" }));
    await user.click(screen.getByRole("button", { name: "移入回收站 训练任务 #9" }));
    const confirm = screen.getByRole("dialog", { name: "移入回收站" });
    await user.click(within(confirm).getByRole("button", { name: "确认移入回收站" }));

    expect(apiMock.trashStorageItem).toHaveBeenCalledWith("training_run", 9);
    expect(onStorageChanged).toHaveBeenCalledOnce();
  });

  it("支持恢复与精确名称确认后彻底删除", async () => {
    const user = userEvent.setup();
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("tab", { name: "回收站" }));
    expect(await screen.findByRole("table", { name: "回收站数据" })).toBeInTheDocument();
    expect(screen.getByText(/天后清理/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "恢复 旧标注版本" }));
    expect(apiMock.restoreTrashItem).toHaveBeenCalledWith(21);

    await user.click(screen.getByRole("button", { name: "彻底删除 旧标注版本" }));
    const purgeDialog = screen.getByRole("dialog", { name: "彻底删除" });
    const purgeButton = within(purgeDialog).getByRole("button", { name: "永久删除" });
    expect(purgeButton).toBeDisabled();
    await user.type(within(purgeDialog).getByLabelText("输入名称确认"), "旧标注版本");
    expect(purgeButton).toBeEnabled();
    await user.click(purgeButton);

    expect(apiMock.purgeTrashItem).toHaveBeenCalledWith(21, "旧标注版本");
  });

  it("按回收站状态展示诊断信息并限制操作", async () => {
    const user = userEvent.setup();
    apiMock.listTrashItems.mockResolvedValue({
      items: [trashItem, errorTrashItem, pendingMoveTrashItem, pendingRestoreTrashItem],
      total_size_bytes: 24_576,
    });
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("tab", { name: "回收站" }));
    const table = await screen.findByRole("table", { name: "回收站数据" });

    const activeRow = within(table).getByRole("row", { name: /旧标注版本/ });
    expect(within(activeRow).getByText("可恢复")).toBeInTheDocument();
    expect(within(activeRow).getByRole("button", { name: "恢复 旧标注版本" })).toBeEnabled();
    expect(within(activeRow).getByRole("button", { name: "彻底删除 旧标注版本" })).toBeEnabled();

    const errorRow = within(table).getByRole("row", { name: /异常标注版本/ });
    expect(within(errorRow).getByText("需要检查")).toBeInTheDocument();
    expect(within(errorRow).getByText(errorTrashItem.error_message)).toBeInTheDocument();
    expect(within(errorRow).getByText("彻底删除前请确认磁盘数据状态")).toBeInTheDocument();
    expect(within(errorRow).getByRole("button", { name: "恢复 异常标注版本" })).toBeDisabled();
    expect(within(errorRow).getByRole("button", { name: "彻底删除 异常标注版本" })).toBeEnabled();

    for (const name of ["正在移入的版本", "正在恢复的版本"]) {
      const pendingRow = within(table).getByRole("row", { name: new RegExp(name) });
      expect(within(pendingRow).getByText("处理中")).toBeInTheDocument();
      expect(within(pendingRow).getByRole("button", { name: `恢复 ${name}` })).toBeDisabled();
      expect(within(pendingRow).getByRole("button", { name: `彻底删除 ${name}` })).toBeDisabled();
    }
  });

  it("删除当前数据集后通知上层清空工作区", async () => {
    const user = userEvent.setup();
    const onDatasetTrashed = vi.fn();
    apiMock.listStorageItems.mockResolvedValue({
      items: [{ ...dataset, protected: false, blockers: [] }],
      total_size_bytes: dataset.size_bytes,
    });

    render(
      <StorageManagementView
        loadedDatasetId={1}
        onDatasetTrashed={onDatasetTrashed}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "移入回收站 camouflage-set" }));
    await user.click(screen.getByRole("button", { name: "确认移入回收站" }));
    expect(onDatasetTrashed).toHaveBeenCalledWith(1);
  });

  it("接口失败时保留弹窗和抽屉并显示可读错误", async () => {
    const user = userEvent.setup();
    apiMock.trashStorageItem.mockRejectedValue(new Error("该版本仍有关联训练任务"));
    render(
      <StorageManagementView
        loadedDatasetId={1}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "查看 标注版本 4" }));
    await user.click(screen.getByRole("button", { name: "移入回收站 训练任务 #9" }));
    await user.click(screen.getByRole("button", { name: "确认移入回收站" }));

    expect(screen.getByRole("dialog", { name: "数据详情" })).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "移入回收站" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("该版本仍有关联训练任务");
  });

  it("彻底删除失败时保留精确名称确认弹窗", async () => {
    const user = userEvent.setup();
    apiMock.purgeTrashItem.mockRejectedValue(new Error("恢复父级对象后才能删除"));
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("tab", { name: "回收站" }));
    await user.click(await screen.findByRole("button", { name: "彻底删除 旧标注版本" }));
    const dialog = screen.getByRole("dialog", { name: "彻底删除" });
    await user.type(within(dialog).getByLabelText("输入名称确认"), "旧标注版本");
    await user.click(within(dialog).getByRole("button", { name: "永久删除" }));

    expect(screen.getByRole("dialog", { name: "彻底删除" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("恢复父级对象后才能删除");
  });

  it("到期清理成功使用状态通知而不是错误警报", async () => {
    const user = userEvent.setup();
    apiMock.purgeExpiredTrash.mockResolvedValue({ purged_count: 2, failed_count: 0 });
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("tab", { name: "回收站" }));
    await screen.findByRole("table", { name: "回收站数据" });
    await user.click(screen.getByRole("button", { name: "清理到期项目" }));

    expect(await screen.findByRole("status")).toHaveTextContent("已清理 2 项，失败 0 项");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("现有数据和回收站均展示明确空状态", async () => {
    const user = userEvent.setup();
    apiMock.listStorageItems.mockResolvedValue({ items: [], total_size_bytes: 0 });
    apiMock.listTrashItems.mockResolvedValue({ items: [], total_size_bytes: 0 });
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    expect(await screen.findByText("暂无现有数据")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "回收站" }));
    expect(await screen.findByText("回收站为空")).toBeInTheDocument();
  });

  it("紧凑操作菜单可以执行现有数据操作", async () => {
    const user = userEvent.setup();
    apiMock.listStorageItems.mockResolvedValue({
      items: [{ ...dataset, protected: false, blockers: [] }],
      total_size_bytes: dataset.size_bytes,
    });
    render(
      <StorageManagementView
        loadedDatasetId={null}
        onDatasetTrashed={vi.fn()}
        onStorageChanged={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "更多操作 camouflage-set" }));
    const menu = screen.getByRole("menu", { name: "camouflage-set 操作" });
    await user.click(within(menu).getByRole("menuitem", { name: "移入回收站" }));

    expect(screen.getByRole("dialog", { name: "移入回收站" })).toBeInTheDocument();
  });
});

describe("storage API error parsing", () => {
  it("提取 FastAPI detail 字符串和结构化消息", async () => {
    vi.resetModules();
    vi.doUnmock("./api");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "数据集不存在" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: { message: "仍有关联训练任务", blockers: [] } }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const realApi = await import("./api");

    await expect(realApi.listStorageItems()).rejects.toThrow("数据集不存在");
    await expect(realApi.listTrashItems()).rejects.toThrow("仍有关联训练任务");
    vi.unstubAllGlobals();
  });
});
