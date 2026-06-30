import { Activity, AlertTriangle, Database, FolderSearch, HardDrive } from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type { DatasetScanSummary, HealthResponse } from "./api";
import { getHealth, scanDataset } from "./api";

const defaultDatasetPath = "/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error: Error) => setHealthError(error.message));
  }, []);

  const totalGroupImages = useMemo(
    () => scan?.groups.reduce((total, group) => total + group.image_count, 0) ?? 0,
    [scan],
  );

  async function handleScan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsScanning(true);
    setScanError(null);

    try {
      setScan(await scanDataset(datasetPath));
    } catch (error) {
      setScanError(error instanceof Error ? error.message : "Dataset scan failed");
    } finally {
      setIsScanning(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Application status">
        <div>
          <p className="eyebrow">Local Detection Workbench</p>
          <h1>YOLO Trainer</h1>
        </div>
        <div className="device-pill">
          <Activity size={16} />
          <span>{health?.devices.selected ?? "connecting"}</span>
        </div>
      </section>

      <section className="status-grid" aria-label="Backend details">
        <StatusTile
          icon={<HardDrive size={20} />}
          label="Workspace"
          value={health?.workspace_root ?? "Waiting for backend"}
        />
        <StatusTile
          icon={<Database size={20} />}
          label="Database"
          value={health?.database_path ?? "SQLite will initialize on startup"}
        />
        <StatusTile
          icon={<Activity size={20} />}
          label="Devices"
          value={health ? `${health.devices.available.length} available` : "Detecting"}
        />
      </section>

      {healthError ? <div className="error-banner">{healthError}</div> : null}

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Dataset Intake</p>
            <h2>Scan Local Zip</h2>
          </div>
          <FolderSearch size={22} />
        </div>

        <form className="scan-form" onSubmit={handleScan}>
          <label htmlFor="dataset-path">Dataset zip path</label>
          <div className="input-row">
            <input
              id="dataset-path"
              value={datasetPath}
              onChange={(event) => setDatasetPath(event.target.value)}
            />
            <button type="submit" disabled={isScanning || datasetPath.trim().length === 0}>
              {isScanning ? "Scanning" : "Scan Dataset"}
            </button>
          </div>
        </form>

        {scanError ? <div className="error-banner">{scanError}</div> : null}

        {scan ? (
          <div className="scan-results">
            <div className="metrics-row">
              <Metric label="Images" value={scan.total_images.toLocaleString()} />
              <Metric label="Metadata rows" value={scan.total_metadata_rows.toLocaleString()} />
              <Metric label="YOLO labels" value={scan.total_yolo_labels.toLocaleString()} />
              <Metric label="Class config" value={scan.has_data_yaml ? "Found" : "Missing"} />
            </div>

            <div className="group-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Group</th>
                    <th>Images</th>
                    <th>Metadata</th>
                    <th>Altitude</th>
                  </tr>
                </thead>
                <tbody>
                  {scan.groups.map((group) => (
                    <tr key={group.name}>
                      <td>{group.name}</td>
                      <td>{group.image_count}</td>
                      <td>{group.metadata_rows}</td>
                      <td>{formatAltitude(group.altitude_min, group.altitude_max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="summary-line">
              Grouped images: {totalGroupImages.toLocaleString()} from {scan.archive_name}
            </div>

            {scan.warnings.length > 0 ? (
              <div className="warnings">
                <AlertTriangle size={18} />
                <div>
                  {scan.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </main>
  );
}

function formatAltitude(minimum: number | null, maximum: number | null) {
  if (minimum === null || maximum === null) {
    return "Not available";
  }

  return `${minimum.toFixed(1)}-${maximum.toFixed(1)}m`;
}

function StatusTile(props: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="status-tile">
      {props.icon}
      <div>
        <span>{props.label}</span>
        <strong>{props.value}</strong>
      </div>
    </div>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}
