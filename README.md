# YOLO Trainer

Local single-user YOLO training workbench.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -v
python -m uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Dataset scan check:

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/scan \
  -H "Content-Type: application/json" \
  -d '{"source_path":"/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip"}'
```

Expected scan facts for the provided archive:

- `total_images` is `1099`.
- `groups` contains `iris` and `vtol`.
- `has_yolo_labels` is `false`.
- `has_data_yaml` is `false`.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, keep the backend running, and scan:

```text
/Users/peinan/DevProjects/YOLO_Trainer/image_dataset.zip
```

## Annotation smoke workflow

With both servers running:

1. Open `http://127.0.0.1:5173`.
2. Click `Scan Dataset` to preview the zip.
3. Click `Import Dataset` to copy the archive images into the managed workspace and create SQLite rows.
4. In `Class Library`, create a class such as `target`.
5. Select an image in `Image Browser`.
6. Drag on the image in `Annotation` to create a bounding box.
7. Optionally fill `Track ID` and `Edge tags`.
8. Click `Save Annotations`.
9. Reselect or reload the image and confirm the saved box is still listed.

## Quality review and dataset version export

After importing a dataset and saving at least one annotation:

1. Check `Quality Review` for image, annotated image, class, box, tiny-box, and issue counts.
2. Confirm the panel says `Ready to export`.
3. Enter an optional version name in `Version Export`.
4. Click `Create Dataset Version`.
5. Confirm the new version appears with train/val/test counts and an artifact path.

The backend writes frozen YOLO artifacts under:

```text
workspace/projects/<project_id>/versions/<version_id>/
  images/train|val|test/
  labels/train|val|test/
  data.yaml
  manifest.json
```

Version export includes only annotated images. It freezes the project class library into a
zero-based YOLO class map sorted by class ID, writes normalized `class x_center y_center width height`
labels, and uses a deterministic 80/10/10 split with at least one validation image when there are
two or more annotated images.

## Training run lifecycle

After a dataset version exists:

1. Open `Training Setup`.
2. Choose a model preset such as `yolov8n.pt`, or enter a local `.pt` path.
3. Set epochs, image size, batch size, device, augmentation preset, TTA, and threshold scan flags.
4. Click `Start Training Run`.
5. Check `Run History` for status, artifact path, latest metrics, errors, and logs.

The backend creates persistent run artifacts under:

```text
workspace/projects/<project_id>/runs/<run_id>/
  config.json
  logs.txt
  metrics.jsonl
```

Only one run can be active for a project at a time. Run status can be `queued`, `preparing`,
`running`, `completed`, or `failed`.

Real training uses Ultralytics when it is installed in the backend Python environment:

```bash
cd backend
python -m pip install ultralytics
```

If Ultralytics is not installed, the run is still persisted and moves to `failed` with a clear log
message. This lets the UI and run history be tested without downloading model weights.

API smoke after creating a dataset version:

```bash
curl -X POST http://127.0.0.1:8000/api/training/runs \
  -H "Content-Type: application/json" \
  -d '{"version_id":1,"model":"yolov8n.pt","epochs":1,"image_size":320,"batch_size":1}'

curl http://127.0.0.1:8000/api/projects/1/training/runs
curl http://127.0.0.1:8000/api/training/runs/1/logs
```

## Prediction analysis and failure samples

After a training run exists:

1. Open `Prediction Analysis`.
2. Choose an image scope: `all`, `train`, `val`, or `test`.
3. Set a confidence threshold.
4. Click `Run Prediction Analysis`.
5. Review matched, false-positive, and false-negative counts.
6. Inspect the prediction/failure sample list; each row keeps the source image ID for annotator follow-up.
7. Click `Open Image` on a sample to jump to that image in the annotator.
8. Use the `GT` and `Pred` layer toggles to isolate saved boxes or model outputs.
9. Prediction boxes show failure type, class, and confidence on the image overlay.
10. Use `Add as annotation` to promote a false-positive prediction into an editable annotation draft.
11. Use `Mark reviewed` on false-negative rows to tag the matched ground-truth box for follow-up.
12. Click `Save Annotations` when the correction draft looks right.

Prediction artifacts are written under:

```text
workspace/projects/<project_id>/runs/<run_id>/predictions/<job_id>/
  logs.txt
  predictions.json
```

Real prediction uses `workspace/projects/<project_id>/runs/<run_id>/ultralytics/weights/best.pt`.
If weights or Ultralytics are missing, the prediction job is persisted as `failed` with a clear log
message.

API smoke after a run exists:

```bash
curl -X POST http://127.0.0.1:8000/api/training/runs/1/prediction-jobs \
  -H "Content-Type: application/json" \
  -d '{"image_scope":"all","confidence_threshold":0.25}'

curl http://127.0.0.1:8000/api/training/runs/1/prediction-jobs
curl http://127.0.0.1:8000/api/prediction-jobs/1/predictions
curl http://127.0.0.1:8000/api/prediction-jobs/1/images/1/review
curl http://127.0.0.1:8000/api/prediction-jobs/1/logs
```
