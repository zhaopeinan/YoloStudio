import { useCallback, useEffect, useState } from "react";
import {
  Database,
  Eye,
  FolderOpen,
  FolderX,
  HardDrive,
  Info,
  MoreHorizontal,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import {
  getStorageItem,
  cascadePurgeTrashItem,
  listStorageItems,
  listTrashItems,
  purgeExpiredTrash,
  purgeTrashItem,
  restoreTrashItem,
  trashStorageItem,
  type StorageEntityType,
  type StorageItem,
  type StorageItemDetail,
  type StorageRelatedRun,
  type TrashItem,
} from "./api";

export type OpenableDatasetRef = {
  project_id: number;
  project_name: string;
  dataset_id: number;
  dataset_name: string;
  image_count: number;
};

type Props = {
  loadedDatasetId: number | null;
  openingDatasetId?: number | null;
  onOpenDataset: (dataset: OpenableDatasetRef) => void | Promise<void>;
  onDatasetTrashed: (datasetId: number) => void;
  onStorageChanged: () => void | Promise<void>;
};

type TrashTarget = {
  entityType: StorageEntityType;
  entityId: number;
  displayName: string;
};

const ACTIVE_RUN_STATUSES = new Set(["queued", "preparing", "running"]);

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[index]}`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function entityLabel(entityType: StorageEntityType): string {
  if (entityType === "dataset") return "原始数据集";
  if (entityType === "dataset_version") return "标注版本";
  return "训练任务";
}

function runStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: "排队中",
    preparing: "准备中",
    running: "训练中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
  };
  return labels[status] ?? status;
}

function contentSummary(item: StorageItem): string {
  return `${item.image_count ?? 0} 张图像 / ${item.annotation_count ?? 0} 个标注`;
}

function resolveOpenableDataset(item: StorageItem, siblingItems: StorageItem[] = []): OpenableDatasetRef | null {
  const datasetId = item.entity_type === "dataset" ? item.entity_id : item.dataset_id;
  if (!datasetId) {
    return null;
  }
  const parentName =
    item.entity_type === "dataset"
      ? item.display_name
      : siblingItems.find(
          (candidate) => candidate.entity_type === "dataset" && candidate.entity_id === datasetId,
        )?.display_name ?? item.display_name;
  return {
    project_id: item.project_id,
    project_name: item.project_name,
    dataset_id: datasetId,
    dataset_name: parentName,
    image_count: item.image_count ?? 0,
  };
}

function splitSummary(split: { train: number; val: number; test: number } | null | undefined) {
  if (!split) return null;
  return `训练 ${split.train} / 验证 ${split.val} / 测试 ${split.test}`;
}

function daysUntil(value: string): number {
  return Math.max(0, Math.ceil((new Date(value).getTime() - Date.now()) / 86_400_000));
}

function trashStatus(item: TrashItem) {
  if (item.status === "active") {
    return {
      label: "可恢复",
      className: "active",
      detail: null,
      canRestore: true,
      canPurge: true,
    };
  }
  if (item.status === "error") {
    return {
      label: "需要检查",
      className: "error",
      detail: item.error_message || "回收站记录状态异常，请检查磁盘数据。",
      canRestore: false,
      canPurge: true,
    };
  }
  if (item.status === "pending_move" || item.status === "pending_restore") {
    return {
      label: "处理中",
      className: "pending",
      detail: item.status === "pending_move" ? "正在移入回收站" : "正在恢复数据",
      canRestore: false,
      canPurge: false,
    };
  }
  return {
    label: item.status || "状态未知",
    className: "unknown",
    detail: "请刷新页面后重试；若状态未恢复，请检查后端日志。",
    canRestore: false,
    canPurge: false,
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "操作失败，请稍后重试";
}

function IconButton({
  label,
  disabled = false,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      className="icon-button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function StorageManagementView({
  loadedDatasetId,
  openingDatasetId = null,
  onOpenDataset,
  onDatasetTrashed,
  onStorageChanged,
}: Props) {
  const [tab, setTab] = useState<"active" | "trash">("active");
  const [items, setItems] = useState<StorageItem[]>([]);
  const [trashItems, setTrashItems] = useState<TrashItem[]>([]);
  const [totalSize, setTotalSize] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [detail, setDetail] = useState<StorageItemDetail | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [trashTarget, setTrashTarget] = useState<TrashTarget | null>(null);
  const [purgeTarget, setPurgeTarget] = useState<TrashItem | null>(null);
  const [cascadePurgeTarget, setCascadePurgeTarget] = useState<TrashItem | null>(null);

  const loadActive = useCallback(async () => {
    const response = await listStorageItems();
    setItems(response.items);
    setTotalSize(response.total_size_bytes);
  }, []);

  const loadTrash = useCallback(async () => {
    const response = await listTrashItems();
    setTrashItems(response.items);
    setTotalSize(response.total_size_bytes);
  }, []);

  const loadCurrentTab = useCallback(async () => {
    setLoading(true);
    setMessage(null);
    setNotice(null);
    try {
      if (tab === "active") await loadActive();
      else await loadTrash();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, [loadActive, loadTrash, tab]);

  useEffect(() => {
    void loadCurrentTab();
  }, [loadCurrentTab]);

  const refreshAfterMutation = async () => {
    await onStorageChanged();
    await loadCurrentTab();
  };

  const openDetail = async (item: StorageItem) => {
    setDetailOpen(true);
    setDetail(null);
    setMessage(null);
    setNotice(null);
    try {
      setDetail(await getStorageItem(item.entity_type, item.entity_id));
    } catch (error) {
      setMessage(errorMessage(error));
    }
  };

  const confirmTrash = async () => {
    if (!trashTarget) return;
    setBusy(true);
    setMessage(null);
    setNotice(null);
    try {
      await trashStorageItem(trashTarget.entityType, trashTarget.entityId);
      if (
        trashTarget.entityType === "dataset" &&
        trashTarget.entityId === loadedDatasetId
      ) {
        onDatasetTrashed(trashTarget.entityId);
      }
      setTrashTarget(null);
      await refreshAfterMutation();
      if (detailOpen && detail) {
        try {
          setDetail(await getStorageItem(detail.entity_type, detail.entity_id));
        } catch {
          setDetailOpen(false);
          setDetail(null);
        }
      }
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const restore = async (item: TrashItem) => {
    setBusy(true);
    setMessage(null);
    setNotice(null);
    try {
      await restoreTrashItem(item.id);
      await refreshAfterMutation();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const purge = async () => {
    if (!purgeTarget) return;
    setBusy(true);
    setMessage(null);
    setNotice(null);
    try {
      await purgeTrashItem(purgeTarget.id, purgeTarget.display_name);
      setPurgeTarget(null);
      await refreshAfterMutation();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const cascadePurge = async () => {
    if (!cascadePurgeTarget) return;
    setBusy(true);
    setMessage(null);
    setNotice(null);
    try {
      await cascadePurgeTrashItem(
        cascadePurgeTarget.id,
        cascadePurgeTarget.display_name,
      );
      if (cascadePurgeTarget.entity_id === loadedDatasetId) {
        onDatasetTrashed(cascadePurgeTarget.entity_id);
      }
      setCascadePurgeTarget(null);
      await refreshAfterMutation();
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const purgeExpired = async () => {
    setBusy(true);
    setMessage(null);
    setNotice(null);
    try {
      const result = await purgeExpiredTrash();
      await onStorageChanged();
      await loadTrash();
      setNotice(`已清理 ${result.purged_count} 项，失败 ${result.failed_count} 项`);
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="storage-management-view" aria-label="数据管理">
      <header className="storage-management-header">
        <div className="segmented-control" role="tablist" aria-label="数据管理视图">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "active"}
            onClick={() => setTab("active")}
          >
            <Database size={16} />
            现有数据
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "trash"}
            onClick={() => setTab("trash")}
          >
            <Trash2 size={16} />
            回收站
          </button>
        </div>
        <div className="storage-total" title="当前列表占用空间">
          <HardDrive size={16} />
          {formatBytes(totalSize)}
        </div>
        {tab === "active" && (
          <p className="storage-open-hint muted">
            可对原始数据集或标注版本点击「打开并标注」，在已有标注基础上继续标注。
          </p>
        )}
        {tab === "trash" && (
          <button type="button" disabled={busy} onClick={() => void purgeExpired()}>
            清理到期项目
          </button>
        )}
      </header>

      {message && <div role="alert" className="status-message error-message">{message}</div>}
      {notice && <div role="status" className="status-message success-message">{notice}</div>}
      {loading ? (
        <p role="status">正在加载数据...</p>
      ) : (
        <div className="storage-table-scroll">
          {tab === "active" ? (
            <ActiveTable
              items={items}
              loadedDatasetId={loadedDatasetId}
              openingDatasetId={openingDatasetId}
              onDetail={openDetail}
              onOpen={(item) => {
                const target = resolveOpenableDataset(item, items);
                if (!target) {
                  setMessage("无法解析该条目对应的原始数据集");
                  return;
                }
                void onOpenDataset(target);
              }}
              onTrash={setTrashTarget}
            />
          ) : (
            <TrashTable
              items={trashItems}
              busy={busy}
              onRestore={restore}
              onPurge={(item) => {
                setPurgeTarget(item);
              }}
              onCascadePurge={(item) => {
                setCascadePurgeTarget(item);
              }}
            />
          )}
        </div>
      )}

      {detailOpen && (
        <DetailDrawer
          detail={detail}
          opening={
            detail != null &&
            resolveOpenableDataset(detail, items)?.dataset_id === openingDatasetId
          }
          onClose={() => {
            setDetailOpen(false);
            setDetail(null);
          }}
          onOpen={() => {
            if (!detail) return;
            const target = resolveOpenableDataset(detail, items);
            if (!target) {
              setMessage("无法解析该条目对应的原始数据集");
              return;
            }
            void onOpenDataset(target);
          }}
          onTrashRun={(run) =>
            setTrashTarget({
              entityType: "training_run",
              entityId: run.id,
              displayName: `训练任务 #${run.id}`,
            })
          }
        />
      )}

      {trashTarget && (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-dialog" role="dialog" aria-modal="true" aria-label="移入回收站">
            <header>
              <h2>移入回收站</h2>
              <IconButton label="关闭移入回收站弹窗" onClick={() => setTrashTarget(null)}>
                <X size={18} />
              </IconButton>
            </header>
            <p>“{trashTarget.displayName}”将保留 30 天，期间可以恢复。</p>
            <footer>
              <button type="button" onClick={() => setTrashTarget(null)}>取消</button>
              <button type="button" disabled={busy} onClick={() => void confirmTrash()}>
                确认移入回收站
              </button>
            </footer>
          </div>
        </div>
      )}

      {purgeTarget && (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-dialog" role="dialog" aria-modal="true" aria-label="彻底删除">
            <header>
              <h2>彻底删除</h2>
              <IconButton label="关闭彻底删除弹窗" onClick={() => setPurgeTarget(null)}>
                <X size={18} />
              </IconButton>
            </header>
            <p>此操作不可恢复。确定要彻底删除“{purgeTarget.display_name}”吗？</p>
            {purgeTarget.status === "error" && (
              <p className="purge-risk-warning">
                该记录处于异常状态。彻底删除前请确认磁盘数据状态，避免遗漏或误删文件。
              </p>
            )}
            <footer>
              <button type="button" onClick={() => setPurgeTarget(null)}>取消</button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void purge()}
              >
                永久删除
              </button>
            </footer>
          </div>
        </div>
      )}

      {cascadePurgeTarget && (
        <div className="modal-backdrop" role="presentation">
          <div
            className="modal-dialog cascade-purge-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="级联彻底删除"
          >
            <header>
              <h2>级联彻底删除</h2>
              <IconButton
                label="关闭级联彻底删除弹窗"
                onClick={() => setCascadePurgeTarget(null)}
              >
                <X size={18} />
              </IconButton>
            </header>
            <p>此操作不可恢复，将一并删除以下内容：</p>
            <ul className="cascade-purge-list">
              <li>原始数据集、图像与标注</li>
              <li>该数据集的全部标注版本</li>
              <li>关联训练任务、预测结果与模型导出产物</li>
            </ul>
            <p>
              确定要删除“{cascadePurgeTarget.display_name}”及其全部关联数据吗？
            </p>
            {cascadePurgeTarget.status === "error" && (
              <p className="purge-risk-warning">
                该记录处于异常状态。执行前请确认磁盘数据状态，避免遗漏或误删文件。
              </p>
            )}
            <footer>
              <button type="button" onClick={() => setCascadePurgeTarget(null)}>
                取消
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void cascadePurge()}
              >
                确认级联删除
              </button>
            </footer>
          </div>
        </div>
      )}
    </section>
  );
}

function ActiveTable({
  items,
  loadedDatasetId,
  openingDatasetId,
  onDetail,
  onOpen,
  onTrash,
}: {
  items: StorageItem[];
  loadedDatasetId: number | null;
  openingDatasetId: number | null;
  onDetail: (item: StorageItem) => void;
  onOpen: (item: StorageItem) => void;
  onTrash: (target: TrashTarget) => void;
}) {
  const [openMenu, setOpenMenu] = useState<string | null>(null);

  return (
    <table className="storage-table" aria-label="现有数据">
      <thead>
        <tr>
          <th scope="col">名称</th>
          <th scope="col">类型</th>
          <th scope="col">项目</th>
          <th scope="col">内容</th>
          <th scope="col">占用空间</th>
          <th scope="col">创建时间</th>
          <th scope="col">状态</th>
          <th scope="col">操作</th>
        </tr>
      </thead>
      <tbody>
        {items.length === 0 && (
          <tr className="storage-empty-row">
            <td colSpan={8}>暂无现有数据</td>
          </tr>
        )}
        {items.map((item) => {
          const rowKey = `${item.entity_type}-${item.entity_id}`;
          const blockerText = item.blockers.map((blocker) => blocker.display_name).join("、");
          const openable = resolveOpenableDataset(item, items);
          const isLoaded = openable != null && openable.dataset_id === loadedDatasetId;
          const isOpening = openable != null && openable.dataset_id === openingDatasetId;
          const openLabel =
            item.entity_type === "dataset"
              ? isLoaded
                ? `继续标注 ${item.display_name}`
                : `打开并标注 ${item.display_name}`
              : `打开原始数据集并标注 ${item.display_name}`;
          return (
            <tr key={rowKey} className={isLoaded ? "storage-row-loaded" : undefined}>
              <th scope="row">
                {item.display_name}
                {isLoaded ? <span className="loaded-chip">已加载</span> : null}
              </th>
              <td>{entityLabel(item.entity_type)}</td>
              <td>{item.project_name}</td>
              <td>
                <span>{contentSummary(item)}</span>
                {item.split_counts && <small>{splitSummary(item.split_counts)}</small>}
              </td>
              <td>{formatBytes(item.size_bytes)}</td>
              <td>{formatDate(item.created_at)}</td>
              <td>
                <span className={`status-badge ${item.protected ? "protected" : "deletable"}`}>
                  {item.protected ? "受保护" : "可删除"}
                </span>
                {item.protected && (
                  <span className="blocker-summary" title={`关联项：${blockerText}`}>
                    <Info size={14} />
                    {blockerText}
                  </span>
                )}
              </td>
              <td>
                <div className="table-actions">
                  <IconButton
                    label={openLabel}
                    disabled={!openable || isOpening}
                    onClick={() => onOpen(item)}
                  >
                    <FolderOpen size={16} />
                  </IconButton>
                  <IconButton label={`查看 ${item.display_name}`} onClick={() => void onDetail(item)}>
                    <Eye size={16} />
                  </IconButton>
                  <IconButton
                    label={`移入回收站 ${item.display_name}`}
                    disabled={item.protected}
                    onClick={() =>
                      onTrash({
                        entityType: item.entity_type,
                        entityId: item.entity_id,
                        displayName: item.display_name,
                      })
                    }
                  >
                    <Trash2 size={16} />
                  </IconButton>
                </div>
                <div className="compact-actions">
                  <IconButton
                    label={`更多操作 ${item.display_name}`}
                    onClick={() => setOpenMenu(openMenu === rowKey ? null : rowKey)}
                  >
                    <MoreHorizontal size={16} />
                  </IconButton>
                  {openMenu === rowKey && (
                    <div className="compact-actions-menu" role="menu" aria-label={`${item.display_name} 操作`}>
                      <button
                        type="button"
                        role="menuitem"
                        disabled={!openable || isOpening}
                        onClick={() => {
                          setOpenMenu(null);
                          onOpen(item);
                        }}
                      >
                        <FolderOpen size={16} />
                        {item.entity_type === "dataset"
                          ? isLoaded
                            ? "继续标注"
                            : "打开并标注"
                          : "打开原始数据集并标注"}
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        onClick={() => {
                          setOpenMenu(null);
                          onDetail(item);
                        }}
                      >
                        <Eye size={16} />
                        查看详情
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        disabled={item.protected}
                        title={item.protected ? `受保护：${blockerText}` : "移入回收站"}
                        onClick={() => {
                          setOpenMenu(null);
                          onTrash({
                            entityType: item.entity_type,
                            entityId: item.entity_id,
                            displayName: item.display_name,
                          });
                        }}
                      >
                        <Trash2 size={16} />
                        移入回收站
                      </button>
                    </div>
                  )}
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function TrashTable({
  items,
  busy,
  onRestore,
  onPurge,
  onCascadePurge,
}: {
  items: TrashItem[];
  busy: boolean;
  onRestore: (item: TrashItem) => void;
  onPurge: (item: TrashItem) => void;
  onCascadePurge: (item: TrashItem) => void;
}) {
  const [openMenu, setOpenMenu] = useState<number | null>(null);

  return (
    <table className="storage-table trash-table" aria-label="回收站数据">
      <thead>
        <tr>
          <th scope="col">名称</th>
          <th scope="col">类型</th>
          <th scope="col">内容</th>
          <th scope="col">占用空间</th>
          <th scope="col">删除时间</th>
          <th scope="col">自动清理</th>
          <th scope="col">状态</th>
          <th scope="col">操作</th>
        </tr>
      </thead>
      <tbody>
        {items.length === 0 && (
          <tr className="storage-empty-row">
            <td colSpan={8}>回收站为空</td>
          </tr>
        )}
        {items.map((item) => {
          const status = trashStatus(item);
          const canCascade = item.entity_type === "dataset" && status.canPurge;
          const actionsDisabled = !status.canRestore && !status.canPurge && !canCascade;
          return (
            <tr key={item.id} className={`trash-row ${status.className}`}>
              <th scope="row">{item.display_name}</th>
              <td>{entityLabel(item.entity_type)}</td>
              <td>
                {item.summary.image_count !== undefined
                  ? `${item.summary.image_count} 张图像 / ${item.summary.annotation_count ?? 0} 个标注`
                  : item.summary.model || runStatusLabel(item.summary.status ?? "")}
              </td>
              <td>{formatBytes(item.size_bytes)}</td>
              <td>{formatDate(item.deleted_at)}</td>
              <td>{daysUntil(item.purge_after)} 天后清理</td>
              <td className="trash-status-cell">
                <span className={`trash-status-badge ${status.className}`}>{status.label}</span>
                {status.detail && <small>{status.detail}</small>}
                {item.status === "error" && (
                  <small className="purge-risk-text">彻底删除前请确认磁盘数据状态</small>
                )}
              </td>
              <td>
                <div className="table-actions">
                  <IconButton
                    label={`恢复 ${item.display_name}`}
                    disabled={busy || !status.canRestore}
                    onClick={() => void onRestore(item)}
                  >
                    <RotateCcw size={16} />
                  </IconButton>
                  <IconButton
                    label={`彻底删除 ${item.display_name}`}
                    disabled={busy || !status.canPurge}
                    onClick={() => onPurge(item)}
                  >
                    <Trash2 size={16} />
                  </IconButton>
                  {canCascade && (
                    <IconButton
                      label={`级联彻底删除 ${item.display_name}`}
                      disabled={busy}
                      onClick={() => onCascadePurge(item)}
                    >
                      <FolderX size={16} />
                    </IconButton>
                  )}
                </div>
                <div className="compact-actions">
                  <IconButton
                    label={`更多操作 ${item.display_name}`}
                    disabled={busy || actionsDisabled}
                    onClick={() => setOpenMenu(openMenu === item.id ? null : item.id)}
                  >
                    <MoreHorizontal size={16} />
                  </IconButton>
                  {openMenu === item.id && (
                    <div className="compact-actions-menu" role="menu" aria-label={`${item.display_name} 操作`}>
                      <button
                        type="button"
                        role="menuitem"
                        disabled={busy || !status.canRestore}
                        onClick={() => {
                          setOpenMenu(null);
                          onRestore(item);
                        }}
                      >
                        <RotateCcw size={16} />
                        恢复
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        disabled={busy || !status.canPurge}
                        onClick={() => {
                          setOpenMenu(null);
                          onPurge(item);
                        }}
                      >
                        <Trash2 size={16} />
                        彻底删除
                      </button>
                      {canCascade && (
                        <button
                          type="button"
                          role="menuitem"
                          disabled={busy}
                          onClick={() => {
                            setOpenMenu(null);
                            onCascadePurge(item);
                          }}
                        >
                          <FolderX size={16} />
                          级联彻底删除
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function DetailDrawer({
  detail,
  opening,
  onClose,
  onOpen,
  onTrashRun,
}: {
  detail: StorageItemDetail | null;
  opening: boolean;
  onClose: () => void;
  onOpen: () => void;
  onTrashRun: (run: StorageRelatedRun) => void;
}) {
  const canOpen = detail != null && resolveOpenableDataset(detail) != null;
  return (
    <div className="drawer-backdrop">
      <aside className="storage-detail-drawer" role="dialog" aria-modal="true" aria-label="数据详情">
        <header>
          <div>
            <span>{detail ? entityLabel(detail.entity_type) : "正在加载"}</span>
            <h2>{detail?.display_name ?? "数据详情"}</h2>
          </div>
          <IconButton label="关闭数据详情" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </header>
        {detail && (
          <div className="storage-detail-content">
            <div className="storage-detail-actions">
              <button type="button" disabled={!canOpen || opening} onClick={onOpen}>
                <FolderOpen size={16} />
                {detail.entity_type === "dataset" ? "打开并标注" : "打开原始数据集并标注"}
              </button>
            </div>
            <dl>
              <dt>所属项目</dt><dd>{detail.project_name}</dd>
              <dt>图像与标注</dt><dd>{contentSummary(detail)}</dd>
              {detail.split_counts && <><dt>数据切分</dt><dd>{splitSummary(detail.split_counts)}</dd></>}
              <dt>占用空间</dt><dd>{formatBytes(detail.size_bytes)}</dd>
              <dt>存储路径</dt><dd className="path-value" title={detail.artifact_path}>{detail.artifact_path}</dd>
            </dl>
            <section>
              <h3>类别</h3>
              <div className="class-name-list">
                {detail.class_names.length > 0 ? detail.class_names.map((name) => <span key={name}>{name}</span>) : "无"}
              </div>
            </section>
            {detail.blockers.length > 0 && (
              <section>
                <h3>删除阻塞项</h3>
                <ul>
                  {detail.blockers.map((blocker) => (
                    <li key={`${blocker.entity_type}-${blocker.entity_id}`}>
                      {blocker.display_name}{blocker.status ? ` · ${runStatusLabel(blocker.status)}` : ""}
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <section>
              <h3>关联训练任务</h3>
              {detail.related_runs.length === 0 ? (
                <p>无关联训练任务</p>
              ) : (
                <table className="storage-related-runs" aria-label="关联训练任务">
                  <thead><tr><th>任务</th><th>模型</th><th>状态</th><th>产物</th><th>操作</th></tr></thead>
                  <tbody>
                    {detail.related_runs.map((run) => {
                      const active = ACTIVE_RUN_STATUSES.has(run.status);
                      return (
                        <tr key={run.id}>
                          <th scope="row">#{run.id}</th>
                          <td>{run.model || "未指定"}</td>
                          <td>{runStatusLabel(run.status)}</td>
                          <td>{run.prediction_job_count} 个预测 / {run.export_count} 个导出</td>
                          <td>
                            <IconButton
                              label={`移入回收站 训练任务 #${run.id}`}
                              disabled={active}
                              onClick={() => onTrashRun(run)}
                            >
                              <Trash2 size={16} />
                            </IconButton>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </section>
          </div>
        )}
      </aside>
    </div>
  );
}
