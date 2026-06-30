import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Database,
  FolderSearch,
  HardDrive,
  Image as ImageIcon,
  Library,
  PackageCheck,
  Play,
  Save,
  Trash2,
  Upload,
} from "lucide-react";
import type { FormEvent, PointerEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type {
  Annotation,
  DatasetImage,
  DatasetImportResponse,
  DatasetQualitySummary,
  DatasetScanSummary,
  DatasetVersion,
  HealthResponse,
  ProjectClass,
  AnnotationWrite,
  TrainingRun,
} from "./api";
import {
  createClass,
  createDatasetVersion,
  createTrainingRun,
  getAnnotations,
  getHealth,
  getQuality,
  getTrainingRunLogs,
  importDataset,
  listClasses,
  listDatasetVersions,
  listImages,
  listTrainingRuns,
  replaceAnnotations,
  scanDataset,
} from "./api";

const defaultDatasetPath = "/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip";
const defaultClassColor = "#ef4444";

type DraftBox = Annotation & {
  local_id: string;
};

type DragState = {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
};

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [datasetPath, setDatasetPath] = useState(defaultDatasetPath);
  const [scan, setScan] = useState<DatasetScanSummary | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [importedDataset, setImportedDataset] = useState<DatasetImportResponse | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [classes, setClasses] = useState<ProjectClass[]>([]);
  const [className, setClassName] = useState("");
  const [classColor, setClassColor] = useState(defaultClassColor);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [classError, setClassError] = useState<string | null>(null);
  const [isCreatingClass, setIsCreatingClass] = useState(false);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<number | null>(null);
  const [annotations, setAnnotations] = useState<DraftBox[]>([]);
  const [annotationError, setAnnotationError] = useState<string | null>(null);
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [quality, setQuality] = useState<DatasetQualitySummary | null>(null);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [isLoadingQuality, setIsLoadingQuality] = useState(false);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [versionName, setVersionName] = useState("");
  const [versionError, setVersionError] = useState<string | null>(null);
  const [isCreatingVersion, setIsCreatingVersion] = useState(false);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [runLogs, setRunLogs] = useState<Record<number, string>>({});
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [isStartingRun, setIsStartingRun] = useState(false);
  const [trainingModel, setTrainingModel] = useState("yolov8n.pt");
  const [trainingEpochs, setTrainingEpochs] = useState(50);
  const [trainingImageSize, setTrainingImageSize] = useState(640);
  const [trainingBatchSize, setTrainingBatchSize] = useState(8);
  const [trainingDevice, setTrainingDevice] = useState("");
  const [augmentationPreset, setAugmentationPreset] = useState("balanced");
  const [trainingTta, setTrainingTta] = useState(false);
  const [thresholdScan, setThresholdScan] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error: Error) => setHealthError(error.message));
  }, []);

  const totalGroupImages = useMemo(
    () => scan?.groups.reduce((total, group) => total + group.image_count, 0) ?? 0,
    [scan],
  );

  const selectedImage = useMemo(
    () => images.find((image) => image.id === selectedImageId) ?? null,
    [images, selectedImageId],
  );

  const selectedClass = useMemo(
    () => classes.find((classItem) => classItem.id === selectedClassId) ?? null,
    [classes, selectedClassId],
  );

  const classById = useMemo(() => {
    return new Map(classes.map((classItem) => [classItem.id, classItem]));
  }, [classes]);

  useEffect(() => {
    if (!selectedImageId) {
      setAnnotations([]);
      return;
    }

    setAnnotationError(null);
    getAnnotations(selectedImageId)
      .then((response) => {
        setAnnotations(response.items.map(toDraftBox));
      })
      .catch((error: Error) => setAnnotationError(error.message));
  }, [selectedImageId]);

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

  async function handleImportDataset() {
    setIsImporting(true);
    setImportError(null);
    setClassError(null);
    setAnnotationError(null);
    setQualityError(null);
    setVersionError(null);

    try {
      const imported = await importDataset(datasetPath);
      setImportedDataset(imported);

      const [classResponse, imageResponse, qualityResponse, versionResponse] = await Promise.all([
        listClasses(imported.project_id),
        listImages(imported.dataset_id),
        getQuality(imported.dataset_id),
        listDatasetVersions(imported.dataset_id),
      ]);
      const runResponse = await listTrainingRuns(imported.project_id);

      setClasses(classResponse.items);
      setSelectedClassId(classResponse.items[0]?.id ?? null);
      setImages(imageResponse.items);
      setSelectedImageId(imageResponse.items[0]?.id ?? null);
      setQuality(qualityResponse);
      setVersions(versionResponse.items);
      setRuns(runResponse.items);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Dataset import failed");
    } finally {
      setIsImporting(false);
    }
  }

  async function refreshTrainingPrep(datasetId = importedDataset?.dataset_id) {
    if (!datasetId) {
      return;
    }

    setIsLoadingQuality(true);
    setQualityError(null);

    try {
      const [qualityResponse, versionResponse] = await Promise.all([
        getQuality(datasetId),
        listDatasetVersions(datasetId),
      ]);
      setQuality(qualityResponse);
      setVersions(versionResponse.items);
      if (importedDataset) {
        const runResponse = await listTrainingRuns(importedDataset.project_id);
        setRuns(runResponse.items);
      }
    } catch (error) {
      setQualityError(error instanceof Error ? error.message : "Quality refresh failed");
    } finally {
      setIsLoadingQuality(false);
    }
  }

  async function refreshTrainingRuns(projectId = importedDataset?.project_id) {
    if (!projectId) {
      return;
    }

    try {
      const runResponse = await listTrainingRuns(projectId);
      setRuns(runResponse.items);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run refresh failed");
    }
  }

  async function handleCreateClass(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!importedDataset || className.trim().length === 0) {
      return;
    }

    setIsCreatingClass(true);
    setClassError(null);

    try {
      const created = await createClass(importedDataset.project_id, {
        name: className.trim(),
        color: classColor,
      });
      setClasses((current) => [...current, created]);
      setSelectedClassId(created.id);
      setClassName("");
      void refreshTrainingPrep();
    } catch (error) {
      setClassError(error instanceof Error ? error.message : "Class creation failed");
    } finally {
      setIsCreatingClass(false);
    }
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if (!selectedClass || !selectedImage) {
      return;
    }

    const point = getRelativePoint(event);
    setDragState({
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!dragState) {
      return;
    }

    const point = getRelativePoint(event);
    setDragState((current) =>
      current ? { ...current, currentX: point.x, currentY: point.y } : current,
    );
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    if (!dragState || !selectedClass) {
      setDragState(null);
      return;
    }

    const point = getRelativePoint(event);
    const box = rectangleToAnnotation(dragState.startX, dragState.startY, point.x, point.y);
    setDragState(null);

    if (!box) {
      return;
    }

    setAnnotations((current) => [
      ...current,
      {
        ...box,
        class_id: selectedClass.id,
        class_name: selectedClass.name,
        class_color: selectedClass.color,
        local_id: `draft-${Date.now()}-${current.length}`,
        track_id: "",
        edge_tags: [],
      },
    ]);
  }

  function updateAnnotation(localId: string, patch: Partial<DraftBox>) {
    setAnnotations((current) =>
      current.map((annotation) =>
        annotation.local_id === localId ? { ...annotation, ...patch } : annotation,
      ),
    );
  }

  function deleteAnnotation(localId: string) {
    setAnnotations((current) => current.filter((annotation) => annotation.local_id !== localId));
  }

  async function handleSaveAnnotations() {
    if (!selectedImageId) {
      return;
    }

    setIsSavingAnnotations(true);
    setAnnotationError(null);

    try {
      const response = await replaceAnnotations(selectedImageId, annotations.map(toAnnotationWrite));
      setAnnotations(response.items.map(toDraftBox));
      setImages((current) =>
        current.map((image) =>
          image.id === selectedImageId
            ? { ...image, annotation_count: response.items.length }
            : image,
        ),
      );
      void refreshTrainingPrep();
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "Saving annotations failed");
    } finally {
      setIsSavingAnnotations(false);
    }
  }

  async function handleCreateDatasetVersion() {
    if (!importedDataset || !quality?.ready_for_training) {
      return;
    }

    setIsCreatingVersion(true);
    setVersionError(null);

    try {
      const created = await createDatasetVersion(
        importedDataset.dataset_id,
        versionName.trim() || undefined,
      );
      setVersions((current) => [created, ...current]);
      setVersionName("");
      await refreshTrainingPrep(importedDataset.dataset_id);
    } catch (error) {
      setVersionError(error instanceof Error ? error.message : "Version export failed");
    } finally {
      setIsCreatingVersion(false);
    }
  }

  async function handleStartTrainingRun() {
    const version = versions[0];
    if (!version || !importedDataset) {
      return;
    }

    setIsStartingRun(true);
    setTrainingError(null);

    try {
      const run = await createTrainingRun({
        version_id: version.id,
        model: trainingModel.trim() || "yolov8n.pt",
        epochs: trainingEpochs,
        image_size: trainingImageSize,
        batch_size: trainingBatchSize,
        device: trainingDevice.trim() || undefined,
        augmentation_preset: augmentationPreset,
        tta: trainingTta,
        threshold_scan: thresholdScan,
      });
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      await refreshTrainingRuns(importedDataset.project_id);
      await loadRunLogs(run.id);
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Training run failed to start");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function loadRunLogs(runId: number) {
    try {
      const response = await getTrainingRunLogs(runId);
      setRunLogs((current) => ({ ...current, [runId]: response.text }));
    } catch (error) {
      setTrainingError(error instanceof Error ? error.message : "Run logs failed to load");
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
            <button
              type="button"
              className="secondary-button"
              disabled={isImporting || datasetPath.trim().length === 0}
              onClick={handleImportDataset}
            >
              <Upload size={16} />
              {isImporting ? "Importing" : "Import Dataset"}
            </button>
          </div>
        </form>

        {scanError ? <div className="error-banner">{scanError}</div> : null}
        {importError ? <div className="error-banner">{importError}</div> : null}

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

        {importedDataset ? (
          <div className="summary-line">
            Imported {importedDataset.image_count.toLocaleString()} images into{" "}
            {importedDataset.project_name} / {importedDataset.dataset_name}
          </div>
        ) : null}
      </section>

      <section className="prep-grid" aria-label="Training readiness">
        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Training Prep</p>
              <h2>Quality Review</h2>
            </div>
            {quality?.ready_for_training ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
          </div>

          {quality ? (
            <>
              <div className="readiness-row">
                <strong>{quality.ready_for_training ? "Ready to export" : "Needs attention"}</strong>
                <span>{isLoadingQuality ? "Refreshing" : `${quality.annotation_count} boxes`}</span>
              </div>

              <div className="metrics-row quality-metrics">
                <Metric label="Images" value={quality.image_count.toLocaleString()} />
                <Metric
                  label="Annotated"
                  value={quality.annotated_image_count.toLocaleString()}
                />
                <Metric label="Classes" value={quality.class_count.toLocaleString()} />
                <Metric label="Tiny boxes" value={quality.tiny_box_count.toLocaleString()} />
              </div>

              {quality.issues.length > 0 ? (
                <div className="issue-list">
                  {quality.issues.map((issue) => (
                    <p key={issue}>{issue}</p>
                  ))}
                </div>
              ) : (
                <p className="empty-state">No blocking quality issues detected.</p>
              )}
            </>
          ) : (
            <p className="empty-state">Import a dataset to compute label quality and export readiness.</p>
          )}

          {qualityError ? <div className="error-banner">{qualityError}</div> : null}
        </section>

        <section className="panel prep-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Frozen Dataset</p>
              <h2>Version Export</h2>
            </div>
            <PackageCheck size={20} />
          </div>

          <div className="version-controls">
            <label htmlFor="version-name">Version name</label>
            <div className="input-row">
              <input
                id="version-name"
                value={versionName}
                disabled={!importedDataset}
                onChange={(event) => setVersionName(event.target.value)}
                placeholder="mvp-quality-pass"
              />
              <button
                type="button"
                disabled={!importedDataset || !quality?.ready_for_training || isCreatingVersion}
                onClick={handleCreateDatasetVersion}
              >
                {isCreatingVersion ? "Exporting" : "Create Dataset Version"}
              </button>
            </div>
          </div>

          {versionError ? <div className="error-banner">{versionError}</div> : null}

          <div className="version-list" aria-label="Dataset versions">
            {versions.length === 0 ? (
              <p className="empty-state">Exported YOLO versions will appear here.</p>
            ) : (
              versions.map((version) => (
                <div className="version-row" key={version.id}>
                  <div>
                    <strong>{version.name}</strong>
                    <span>{version.artifact_path}</span>
                  </div>
                  <span>
                    {version.split_counts.train}/{version.split_counts.val}/
                    {version.split_counts.test}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>
      </section>

      <section className="training-grid" aria-label="Training setup and runs">
        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Model Training</p>
              <h2>Training Setup</h2>
            </div>
            <Play size={20} />
          </div>

          <div className="training-form">
            <label htmlFor="training-model">Model preset or local weights</label>
            <input
              id="training-model"
              value={trainingModel}
              onChange={(event) => setTrainingModel(event.target.value)}
              disabled={versions.length === 0}
            />

            <div className="training-number-grid">
              <label htmlFor="training-epochs">
                Epochs
                <input
                  id="training-epochs"
                  type="number"
                  min={1}
                  max={1000}
                  value={trainingEpochs}
                  onChange={(event) => setTrainingEpochs(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
              <label htmlFor="training-imgsz">
                Image size
                <input
                  id="training-imgsz"
                  type="number"
                  min={32}
                  max={4096}
                  value={trainingImageSize}
                  onChange={(event) => setTrainingImageSize(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
              <label htmlFor="training-batch">
                Batch
                <input
                  id="training-batch"
                  type="number"
                  min={1}
                  max={256}
                  value={trainingBatchSize}
                  onChange={(event) => setTrainingBatchSize(Number(event.target.value))}
                  disabled={versions.length === 0}
                />
              </label>
            </div>

            <div className="training-number-grid">
              <label htmlFor="training-device">
                Device
                <input
                  id="training-device"
                  value={trainingDevice}
                  onChange={(event) => setTrainingDevice(event.target.value)}
                  placeholder={health?.devices.selected ?? "cpu"}
                  disabled={versions.length === 0}
                />
              </label>
              <label htmlFor="augmentation-preset">
                Augmentation
                <input
                  id="augmentation-preset"
                  value={augmentationPreset}
                  onChange={(event) => setAugmentationPreset(event.target.value)}
                  disabled={versions.length === 0}
                />
              </label>
            </div>

            <div className="toggle-row">
              <label>
                <input
                  type="checkbox"
                  checked={trainingTta}
                  onChange={(event) => setTrainingTta(event.target.checked)}
                  disabled={versions.length === 0}
                />
                TTA
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={thresholdScan}
                  onChange={(event) => setThresholdScan(event.target.checked)}
                  disabled={versions.length === 0}
                />
                Threshold scan
              </label>
            </div>

            <button
              type="button"
              disabled={versions.length === 0 || isStartingRun}
              onClick={handleStartTrainingRun}
            >
              <Play size={16} />
              {isStartingRun ? "Starting" : "Start Training Run"}
            </button>
          </div>

          {trainingError ? <div className="error-banner">{trainingError}</div> : null}
        </section>

        <section className="panel training-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Experiments</p>
              <h2>Run History</h2>
            </div>
            <Activity size={20} />
          </div>

          <div className="run-list" aria-label="Training runs">
            {runs.length === 0 ? (
              <p className="empty-state">Create a dataset version, then start a training run.</p>
            ) : (
              runs.map((run) => (
                <div className="run-row" key={run.id}>
                  <div className="run-row-heading">
                    <strong>Run #{run.id}</strong>
                    <span className={`run-status ${run.status}`}>{run.status}</span>
                  </div>
                  <span>{run.artifact_path}</span>
                  <span>
                    {String(run.config.model ?? "model")} | {String(run.config.epochs ?? "?")} epochs |{" "}
                    {run.device}
                  </span>
                  {Object.keys(run.latest_metrics).length > 0 ? (
                    <div className="metric-chips">
                      {Object.entries(run.latest_metrics).map(([name, value]) => (
                        <span key={name}>
                          {name}: {value.toFixed(3)}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {run.error_message ? <p className="run-error">{run.error_message}</p> : null}
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => loadRunLogs(run.id)}
                  >
                    Load Logs
                  </button>
                  {runLogs[run.id] ? <pre className="log-preview">{runLogs[run.id]}</pre> : null}
                </div>
              ))
            )}
          </div>
        </section>
      </section>

      <section className="workbench-grid" aria-label="Annotation workbench">
        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Project Labels</p>
              <h2>Class Library</h2>
            </div>
            <Library size={20} />
          </div>

          <form className="class-form" onSubmit={handleCreateClass}>
            <label htmlFor="class-name">Class name</label>
            <input
              id="class-name"
              value={className}
              disabled={!importedDataset}
              onChange={(event) => setClassName(event.target.value)}
              placeholder="target"
            />
            <label htmlFor="class-color">Class color</label>
            <div className="color-row">
              <input
                id="class-color"
                type="color"
                value={classColor}
                disabled={!importedDataset}
                onChange={(event) => setClassColor(event.target.value)}
                aria-label="Class color"
              />
              <button
                type="submit"
                disabled={!importedDataset || isCreatingClass || className.trim().length === 0}
              >
                Create Class
              </button>
            </div>
          </form>

          {classError ? <div className="error-banner">{classError}</div> : null}

          <div className="class-list" aria-label="Available classes">
            {classes.length === 0 ? (
              <p className="empty-state">Import a dataset, then create a class to draw boxes.</p>
            ) : (
              classes.map((classItem) => (
                <button
                  key={classItem.id}
                  type="button"
                  className={
                    classItem.id === selectedClassId ? "class-chip selected" : "class-chip"
                  }
                  onClick={() => setSelectedClassId(classItem.id)}
                >
                  <span style={{ background: classItem.color }} />
                  {classItem.name}
                </button>
              ))
            )}
          </div>
        </aside>

        <aside className="panel side-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Dataset Frames</p>
              <h2>Image Browser</h2>
            </div>
            <ImageIcon size={20} />
          </div>

          <div className="image-list" aria-label="Imported images">
            {images.length === 0 ? (
              <p className="empty-state">Imported images will appear here.</p>
            ) : (
              images.map((image) => (
                <button
                  key={image.id}
                  type="button"
                  className={image.id === selectedImageId ? "image-row selected" : "image-row"}
                  onClick={() => setSelectedImageId(image.id)}
                >
                  <strong>{image.relative_path}</strong>
                  <span>
                    {image.platform ?? "unknown"} | {formatImageAltitude(image.altitude)} |{" "}
                    {image.annotation_count} boxes
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="panel annotation-panel">
          <div className="panel-heading compact-heading">
            <div>
              <p className="eyebrow">Draw And Review</p>
              <h2>Annotation</h2>
            </div>
            <Box size={20} />
          </div>

          {selectedImage ? (
            <div className="annotation-layout">
              <div className="viewer-wrap">
                <div className="image-stage">
                  <img src={selectedImage.image_url} alt={selectedImage.relative_path} />
                  <svg
                    aria-label="Annotation canvas"
                    className={selectedClass ? "annotation-overlay drawable" : "annotation-overlay"}
                    viewBox="0 0 1 1"
                    preserveAspectRatio="none"
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerCancel={() => setDragState(null)}
                  >
                    {annotations.map((annotation) => (
                      <BoxRect
                        key={annotation.local_id}
                        annotation={annotation}
                        color={resolveClassColor(annotation, classById)}
                      />
                    ))}
                    {dragState ? <DragRect dragState={dragState} color={selectedClass?.color} /> : null}
                  </svg>
                </div>
              </div>

              <div className="box-list">
                <div className="box-list-heading">
                  <strong>Boxes</strong>
                  <button
                    type="button"
                    onClick={handleSaveAnnotations}
                    disabled={isSavingAnnotations || !selectedImage}
                  >
                    <Save size={16} />
                    {isSavingAnnotations ? "Saving" : "Save Annotations"}
                  </button>
                </div>

                {annotationError ? <div className="error-banner">{annotationError}</div> : null}

                {annotations.length === 0 ? (
                  <p className="empty-state">
                    Select a class, then drag over the image to add a bounding box.
                  </p>
                ) : (
                  annotations.map((annotation, index) => (
                    <div className="box-editor" key={annotation.local_id}>
                      <div className="box-editor-title">
                        <span
                          style={{ background: resolveClassColor(annotation, classById) }}
                        />
                        <strong>
                          {annotation.class_name ??
                            classById.get(annotation.class_id)?.name ??
                            `Class ${annotation.class_id}`}
                        </strong>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={`Delete box ${index + 1}`}
                          onClick={() => deleteAnnotation(annotation.local_id)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>

                      <label htmlFor={`track-${annotation.local_id}`}>Track ID</label>
                      <input
                        id={`track-${annotation.local_id}`}
                        value={annotation.track_id ?? ""}
                        onChange={(event) =>
                          updateAnnotation(annotation.local_id, { track_id: event.target.value })
                        }
                      />

                      <label htmlFor={`tags-${annotation.local_id}`}>Edge tags</label>
                      <input
                        id={`tags-${annotation.local_id}`}
                        value={(annotation.edge_tags ?? []).join(", ")}
                        onChange={(event) =>
                          updateAnnotation(annotation.local_id, {
                            edge_tags: parseTags(event.target.value),
                          })
                        }
                        placeholder="occluded, small"
                      />
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <p className="empty-state">Import a dataset and select an image to begin annotation.</p>
          )}
        </section>
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

function formatImageAltitude(altitude: number | null) {
  return altitude === null ? "altitude n/a" : `${altitude.toFixed(1)}m`;
}

function parseTags(value: string) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function toDraftBox(annotation: Annotation, index: number): DraftBox {
  return {
    ...annotation,
    local_id: annotation.id ? `annotation-${annotation.id}` : `loaded-${index}`,
    track_id: annotation.track_id ?? "",
    edge_tags: annotation.edge_tags ?? [],
  };
}

function toAnnotationWrite(annotation: DraftBox): AnnotationWrite {
  return {
    class_id: annotation.class_id,
    x_center: annotation.x_center,
    y_center: annotation.y_center,
    width: annotation.width,
    height: annotation.height,
    track_id: annotation.track_id?.trim() ? annotation.track_id.trim() : null,
    edge_tags: annotation.edge_tags ?? [],
  };
}

function getRelativePoint(event: PointerEvent<SVGSVGElement>) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
  };
}

function rectangleToAnnotation(startX: number, startY: number, endX: number, endY: number) {
  const left = Math.min(startX, endX);
  const right = Math.max(startX, endX);
  const top = Math.min(startY, endY);
  const bottom = Math.max(startY, endY);
  const width = right - left;
  const height = bottom - top;

  if (width < 0.005 || height < 0.005) {
    return null;
  }

  return {
    x_center: left + width / 2,
    y_center: top + height / 2,
    width,
    height,
  };
}

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

function resolveClassColor(annotation: DraftBox, classById: Map<number, ProjectClass>) {
  return annotation.class_color ?? classById.get(annotation.class_id)?.color ?? defaultClassColor;
}

function BoxRect(props: { annotation: DraftBox; color: string }) {
  const { annotation, color } = props;
  return (
    <rect
      x={annotation.x_center - annotation.width / 2}
      y={annotation.y_center - annotation.height / 2}
      width={annotation.width}
      height={annotation.height}
      fill="transparent"
      stroke={color}
      strokeWidth={0.004}
      vectorEffect="non-scaling-stroke"
    />
  );
}

function DragRect(props: { dragState: DragState; color?: string }) {
  const { dragState, color = defaultClassColor } = props;
  const draft = rectangleToAnnotation(
    dragState.startX,
    dragState.startY,
    dragState.currentX,
    dragState.currentY,
  );

  if (!draft) {
    return null;
  }

  return (
    <rect
      x={draft.x_center - draft.width / 2}
      y={draft.y_center - draft.height / 2}
      width={draft.width}
      height={draft.height}
      fill="rgba(255, 255, 255, 0.16)"
      stroke={color}
      strokeDasharray="0.018 0.012"
      strokeWidth={0.004}
      vectorEffect="non-scaling-stroke"
    />
  );
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
