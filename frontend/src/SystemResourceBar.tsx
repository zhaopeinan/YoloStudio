import { Activity, Cpu, HardDrive, MemoryStick } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { getSystemResources, type SystemResources } from "./api";

function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${Math.round(value)}%`;
}

function meterTone(value: number | null | undefined): "ok" | "warn" | "hot" | "unknown" {
  if (value == null || Number.isNaN(value)) return "unknown";
  if (value >= 90) return "hot";
  if (value >= 75) return "warn";
  return "ok";
}

function ResourceChip({
  icon,
  label,
  value,
  detail,
  percent,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail?: string;
  percent: number | null | undefined;
}) {
  return (
    <div className={`resource-chip tone-${meterTone(percent)}`} title={detail}>
      <span className="resource-chip-icon" aria-hidden="true">
        {icon}
      </span>
      <span className="resource-chip-label">{label}</span>
      <span className="resource-chip-value">{value}</span>
      <span className="resource-chip-meter" aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(100, percent ?? 0))}%` }} />
      </span>
    </div>
  );
}

export function SystemResourceBar() {
  const [stats, setStats] = useState<SystemResources | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      void getSystemResources()
        .then((response) => {
          if (!cancelled) setStats(response);
        })
        .catch(() => {
          /* keep last good sample */
        });
    };
    load();
    const timer = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const gpu = stats?.gpu.gpus[0] ?? null;
  const gpuPercent = gpu?.utilization_gpu ?? null;
  const gpuMemPercent =
    gpu?.memory_used_mb != null && gpu?.memory_total_mb
      ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
      : null;

  return (
    <div className="resource-bar" aria-label="系统资源占用" role="status">
      <ResourceChip
        icon={<Cpu size={14} />}
        label="CPU"
        value={formatPercent(stats?.cpu_percent)}
        percent={stats?.cpu_percent}
        detail={
          stats?.load_average
            ? `load ${stats.load_average.join(" / ")} · ${stats.cpu_count ?? "?"} 核`
            : undefined
        }
      />
      <ResourceChip
        icon={<MemoryStick size={14} />}
        label="内存"
        value={formatPercent(stats?.memory.percent)}
        percent={stats?.memory.percent}
        detail={
          stats?.memory.used_mb != null && stats.memory.total_mb != null
            ? `${Math.round(stats.memory.used_mb)} / ${Math.round(stats.memory.total_mb)} MB`
            : undefined
        }
      />
      <ResourceChip
        icon={<HardDrive size={14} />}
        label="磁盘"
        value={formatPercent(stats?.disk.percent)}
        percent={stats?.disk.percent}
        detail={
          stats?.disk.used_gb != null && stats.disk.total_gb != null
            ? `${stats.disk.used_gb.toFixed(1)} / ${stats.disk.total_gb.toFixed(1)} GB`
            : undefined
        }
      />
      <ResourceChip
        icon={<Activity size={14} />}
        label="GPU"
        value={
          stats?.gpu.available === false
            ? "无"
            : gpu
              ? `${formatPercent(gpuPercent)}`
              : "—"
        }
        percent={gpuPercent}
        detail={
          gpu
            ? `${gpu.name} · 显存 ${Math.round(gpu.memory_used_mb ?? 0)}/${Math.round(gpu.memory_total_mb ?? 0)} MB (${formatPercent(gpuMemPercent)}) · ${gpu.temperature_c ?? "—"}°C`
            : stats?.gpu.error ?? "等待 GPU 状态"
        }
      />
    </div>
  );
}
