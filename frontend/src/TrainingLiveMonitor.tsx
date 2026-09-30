import { Activity, Gauge, Thermometer, X, Zap } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getTrainingRunLive,
  type TrainingLiveSnapshot,
} from "./api";

const LIVE_REFRESH_MS = 1000;
const HIDDEN_REFRESH_MS = 4000;

type ChartTab = "loss" | "map" | "lr";

const LOSS_KEYS = [
  "train/box_loss",
  "train/cls_loss",
  "train/dfl_loss",
  "val/box_loss",
  "val/cls_loss",
  "val/dfl_loss",
] as const;

const MAP_KEYS = [
  "metrics/mAP50(B)",
  "metrics/mAP50-95(B)",
  "metrics/precision(B)",
  "metrics/recall(B)",
] as const;

const LR_KEYS = ["lr/pg0", "lr/pg1", "lr/pg2"] as const;

const SERIES_COLORS = ["#0f766e", "#c2410c", "#1d4ed8", "#a16207", "#7c3aed", "#be123c"];

function formatDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return "—";
  const total = Math.round(sec);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function shortName(name: string): string {
  return name
    .replace("metrics/", "")
    .replace("train/", "tr/")
    .replace("val/", "va/")
    .replace("(B)", "");
}

function MultiSeriesChart(props: {
  epochs: Array<number | null>;
  series: Record<string, Array<number | null>>;
  keys: readonly string[];
}) {
  const { epochs, series, keys } = props;
  const active = keys.filter((key) => (series[key] || []).some((v) => v != null));
  if (active.length === 0 || epochs.length === 0) {
    return <p className="empty-state">等待 epoch 指标写入 results.csv…</p>;
  }

  const allValues = active.flatMap((key) => (series[key] || []).filter((v): v is number => v != null));
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;
  const n = epochs.length;

  return (
    <div className="live-chart-wrap">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="live-chart-svg" aria-hidden="true">
        <line x1="0" y1="88" x2="100" y2="88" />
        <line x1="0" y1="16" x2="100" y2="16" className="live-chart-grid" />
        {active.map((key, seriesIndex) => {
          const values = series[key] || [];
          const points = values
            .map((value, index) => {
              if (value == null) return null;
              const x = n <= 1 ? 50 : (index / (n - 1)) * 100;
              const y = 88 - ((value - min) / range) * 72;
              return `${x},${y}`;
            })
            .filter(Boolean)
            .join(" ");
          if (!points) return null;
          return (
            <polyline
              key={key}
              points={points}
              style={{ stroke: SERIES_COLORS[seriesIndex % SERIES_COLORS.length] }}
            />
          );
        })}
      </svg>
      <div className="live-chart-legend">
        {active.map((key, index) => {
          const values = series[key] || [];
          const latest = [...values].reverse().find((v) => v != null);
          return (
            <span key={key} style={{ color: SERIES_COLORS[index % SERIES_COLORS.length] }}>
              {shortName(key)}
              {latest == null ? "" : ` ${latest.toFixed(3)}`}
            </span>
          );
        })}
      </div>
      <div className="live-chart-range">
        <span>{max.toFixed(3)}</span>
        <span>{min.toFixed(3)}</span>
      </div>
    </div>
  );
}

export function TrainingLiveMonitor(props: {
  runId: number;
  onClose: () => void;
}) {
  const { runId, onClose } = props;
  const [snapshot, setSnapshot] = useState<TrainingLiveSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<ChartTab>("loss");
  const [paused, setPaused] = useState(false);
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const data = await getTrainingRunLive(runId, true);
      setSnapshot(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载监控失败");
    } finally {
      inFlight.current = false;
    }
  }, [runId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const tick = () => {
      if (paused) return;
      if (document.hidden) return;
      void refresh();
    };
    const visibleId = window.setInterval(tick, LIVE_REFRESH_MS);
    const hiddenId = window.setInterval(() => {
      if (!document.hidden || paused) return;
      void refresh();
    }, HIDDEN_REFRESH_MS);

    const onVisibility = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      window.clearInterval(visibleId);
      window.clearInterval(hiddenId);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [paused, refresh]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const percent = snapshot?.progress.percent ?? 0;
  const chartKeys = useMemo(() => {
    if (tab === "map") return MAP_KEYS;
    if (tab === "lr") return LR_KEYS;
    return LOSS_KEYS;
  }, [tab]);

  const highlightMetrics = useMemo(() => {
    const latest = snapshot?.latest || {};
    const keys = [
      "train/box_loss",
      "train/cls_loss",
      "train/dfl_loss",
      "val/box_loss",
      "metrics/mAP50(B)",
      "metrics/mAP50-95(B)",
    ];
    return keys
      .filter((key) => latest[key] != null)
      .map((key) => ({ name: key, value: latest[key] }));
  }, [snapshot]);

  return (
    <div className="live-monitor-backdrop" role="presentation" onClick={onClose}>
      <div
        className="live-monitor-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="训练实时监控"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="live-monitor-header">
          <div>
            <p className="eyebrow">实时监控 · 1s 刷新</p>
            <h2>
              训练任务 #{runId}
              {snapshot ? (
                <span className={`run-status ${snapshot.status}`}>{snapshot.status_label}</span>
              ) : null}
            </h2>
            <p className="live-monitor-subtitle">
              {snapshot
                ? `${String(snapshot.config.model ?? "—")} · ${String(snapshot.config.epochs ?? "?")} epochs · batch ${String(snapshot.config.batch_size ?? "—")} · ${snapshot.device}`
                : "加载中…"}
            </p>
          </div>
          <div className="live-monitor-header-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() => setPaused((value) => !value)}
            >
              {paused ? "继续刷新" : "暂停刷新"}
            </button>
            <button type="button" className="icon-button" aria-label="关闭" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="live-monitor-body">
          <section className="live-kpi-grid">
            <div className="live-kpi-card">
              <div className="live-kpi-label">
                <Activity size={16} /> 进度
              </div>
              <div className="live-progress-ring" style={{ ["--pct" as string]: `${percent}%` }}>
                <strong>{percent.toFixed(0)}%</strong>
              </div>
              <p>
                Epoch {snapshot?.progress.epoch ?? "—"} / {snapshot?.progress.total_epochs ?? "—"}
              </p>
              <p className="muted">
                已用 {formatDuration(snapshot?.progress.elapsed_sec)} · ETA{" "}
                {formatDuration(snapshot?.progress.eta_sec)}
              </p>
            </div>

            <div className="live-kpi-card live-kpi-wide">
              <div className="live-kpi-label">
                <Gauge size={16} /> GPU
              </div>
              {!snapshot?.gpu?.available ? (
                <p className="empty-state">{snapshot?.gpu?.error || "暂无 GPU 数据"}</p>
              ) : (
                <div className="live-gpu-list">
                  {snapshot.gpu.gpus.map((gpu) => {
                    const memPct =
                      gpu.memory_used_mb != null && gpu.memory_total_mb
                        ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
                        : 0;
                    return (
                      <div className="live-gpu-card" key={gpu.index}>
                        <strong>
                          GPU{gpu.index} · {gpu.name}
                        </strong>
                        <div className="live-gpu-meters">
                          <div>
                            <span>利用率</span>
                            <div className="live-meter">
                              <i style={{ width: `${gpu.utilization_gpu ?? 0}%` }} />
                            </div>
                            <em>{gpu.utilization_gpu?.toFixed(0) ?? "—"}%</em>
                          </div>
                          <div>
                            <span>显存</span>
                            <div className="live-meter">
                              <i style={{ width: `${memPct}%` }} />
                            </div>
                            <em>
                              {gpu.memory_used_mb?.toFixed(0) ?? "—"} /{" "}
                              {gpu.memory_total_mb?.toFixed(0) ?? "—"} MB
                            </em>
                          </div>
                        </div>
                        <div className="live-gpu-stats">
                          <span>
                            <Thermometer size={14} /> {gpu.temperature_c?.toFixed(0) ?? "—"}°C
                          </span>
                          <span>
                            <Zap size={14} /> {gpu.power_w?.toFixed(0) ?? "—"} /{" "}
                            {gpu.power_limit_w?.toFixed(0) ?? "—"} W
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="live-kpi-card">
              <div className="live-kpi-label">关键指标</div>
              <div className="live-metric-list">
                {highlightMetrics.length === 0 ? (
                  <p className="empty-state">训练开始后显示</p>
                ) : (
                  highlightMetrics.map((item) => (
                    <div key={item.name}>
                      <span>{shortName(item.name)}</span>
                      <strong>{item.value.toFixed(4)}</strong>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          <section className="live-chart-panel">
            <div className="live-chart-tabs" role="tablist">
              {(
                [
                  ["loss", "Loss"],
                  ["map", "mAP / P-R"],
                  ["lr", "学习率"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  role="tab"
                  aria-selected={tab === id}
                  className={tab === id ? "is-active" : undefined}
                  onClick={() => setTab(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <MultiSeriesChart
              epochs={snapshot?.series.epoch || []}
              series={snapshot?.series || {}}
              keys={chartKeys}
            />
          </section>

          <section className="live-log-panel">
            <div className="live-kpi-label">日志尾部</div>
            <pre className="live-log-tail">
              {(snapshot?.log_tail || []).join("\n") || "暂无日志"}
            </pre>
            {snapshot?.error_message ? (
              <p className="run-error">{snapshot.error_message}</p>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}

export default TrainingLiveMonitor;
