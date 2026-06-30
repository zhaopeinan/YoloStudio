import {
  Activity,
  AlertTriangle,
  Box,
  Database,
  FolderSearch,
  HardDrive,
  Image as ImageIcon,
  Library,
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
  DatasetScanSummary,
  HealthResponse,
  ProjectClass,
  AnnotationWrite,
} from "./api";
import {
  createClass,
  getAnnotations,
  getHealth,
  importDataset,
  listClasses,
  listImages,
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

    try {
      const imported = await importDataset(datasetPath);
      setImportedDataset(imported);

      const [classResponse, imageResponse] = await Promise.all([
        listClasses(imported.project_id),
        listImages(imported.dataset_id),
      ]);

      setClasses(classResponse.items);
      setSelectedClassId(classResponse.items[0]?.id ?? null);
      setImages(imageResponse.items);
      setSelectedImageId(imageResponse.items[0]?.id ?? null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Dataset import failed");
    } finally {
      setIsImporting(false);
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
    } catch (error) {
      setAnnotationError(error instanceof Error ? error.message : "Saving annotations failed");
    } finally {
      setIsSavingAnnotations(false);
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
