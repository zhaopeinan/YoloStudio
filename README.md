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

Dataset scan check. `source_path` accepts either a `.zip` archive or a local dataset folder:

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/scan \
  -H "Content-Type: application/json" \
  -d '{"source_path":"/path/to/dataset.zip"}'
```

The response includes image counts, collection group names, and whether YOLO labels or `data.yaml` are present.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, keep the backend running, and scan a zip or folder path on your machine, for example:

```text
/path/to/dataset.zip
```

Set `Project name` and `Dataset name` before import. Imports that use the same project name reuse
the project-level class library, so related datasets can share labels while each dataset keeps its
own image list, annotations, quality review, and exported versions.

After a restart or browser refresh, use `Saved dataset` and `Load Dataset` in Dataset Intake to
reload an already imported dataset without copying the source files again.

## Annotation smoke workflow

With both servers running:

1. Open `http://127.0.0.1:5173`.
2. Click `Scan Dataset` to preview the zip or folder.
3. Click `Import Dataset` to copy the source images into the managed workspace and create SQLite rows.
4. In `Class Library`, create a class such as `target`.
5. Select an image in `Image Browser`.
6. Drag on the image in `Annotation` to create a bounding box.
7. Select an existing box on the image to move it, drag a selected corner to resize it, or use the
   arrow buttons in the box editor to nudge it precisely.
8. Optionally fill `Track ID` and `Edge tags`.
9. Click `Save Annotations`.
10. Reselect or reload the image and confirm the saved box is still listed.

## Dataset browser filters

After import, `Image Browser` can filter the current dataset by:

- platform, matching a collection folder name in the dataset
- label status: all, annotated, or unannotated
- project class
- edge tag, such as `occluded`
- altitude range

API smoke:

```bash
curl "http://127.0.0.1:8000/api/datasets/1/images?platform=group_a&label_status=annotated"
curl "http://127.0.0.1:8000/api/datasets/1/images?class_id=1&edge_tag=occluded"
curl "http://127.0.0.1:8000/api/datasets/1/images?altitude_min=20&altitude_max=40"
```

## Quality review and dataset version export

After importing a dataset and saving at least one annotation:

1. Check `Quality Review` for image, annotated image, class, box, tiny-box, and issue counts.
2. Use `Quality issue type` to inspect actionable tiny, duplicate, invalid, metadata, and unannotated samples.
3. Check `Dataset Coverage` to see platform, altitude-band, class, and edge-tag coverage before
   freezing a version.
4. Click `Apply Auto Tags` to write annotation-level quality tags such as `tiny_box`,
   `duplicate_box`, and `invalid_box` into each affected annotation's `edge_tags`.
5. Use the Image Browser `Edge tag` filter to create follow-up correction queues from those tags.
6. Confirm the panel says `Ready to export`.
7. Enter an optional version name in `Version Export`.
8. Select the class subset to freeze for this version. Leaving all classes selected exports the full
   active project class library; selecting a subset exports only images and labels that contain those
   classes.
9. Click `Create Dataset Version`.
10. Confirm the new version appears with train/val/test counts and an artifact path.

The backend writes frozen YOLO artifacts under:

```text
workspace/projects/<project_id>/versions/<version_id>/
  images/train|val|test/
  labels/train|val|test/
  data.yaml
  manifest.json
```

Version export includes only annotated images for the selected classes. It freezes the selected
project classes into a zero-based YOLO class map sorted by class ID, writes normalized
`class x_center y_center width height` labels, and uses a deterministic 80/10/10 split with at least
one validation image when there are two or more annotated images.
The version `manifest.json` also freezes each exported image's platform, altitude, timestamp,
dimensions, source metadata, annotation `track_id`, and edge tags so later prediction review or
situation-map alignment can trace results back to the exact training snapshot.

## Training run lifecycle

After a dataset version exists:

1. Open `Training Setup`.
2. Choose a model preset such as `yolov8n.pt`, or enter a local `.pt` path.
3. Set epochs, image size, batch size, device, augmentation strategy, TTA, and threshold scan flags.
4. Click `Start Training Run`.
5. Check `Run History` for status, artifact path, latest metrics, generated artifacts, errors,
   and logs.
6. While a run is `queued`, `preparing`, or `running`, the frontend shows `Auto refresh on` and
   refreshes run status plus logs automatically.
7. Click `Load Config` on a historical run to restore its model, hyperparameters, augmentation,
   TTA, and threshold-scan settings into Training Setup.
8. Click `Rerun` to start a new run from that historical config on the latest dataset version.
9. Click `Cancel Run` on an active run to mark it `cancelled`, write a cancellation log, and allow
   another run to be queued.

The backend creates persistent run artifacts under:

```text
workspace/projects/<project_id>/runs/<run_id>/
  config.json
  logs.txt
  metrics.jsonl
```

The Run History `Run Artifacts` panel and API endpoint list generated configs, logs, metrics,
weights, plots, prediction outputs, GridMask derivatives, and exports:

```bash
curl http://127.0.0.1:8000/api/training/runs/1/artifacts
```

Only one run can be active for a project at a time. Run status can be `queued`, `preparing`,
`running`, `completed`, `failed`, or `cancelled`.

Real training uses Ultralytics when it is installed in the backend Python environment:

```bash
cd backend
python -m pip install ultralytics
```

The training config persists both a strategy name and structured augmentation values:

- `mosaic`, `mixup`, and `copy_paste` for small/dense targets
- `hsv_h`, `hsv_s`, `hsv_v`, `translate`, `scale`, `fliplr`, and `erasing`
- `gridmask` to generate a run-local masked training dataset for occlusion robustness

Ultralytics-supported augmentation fields are passed into `model.train()`. When `gridmask` is
enabled, the backend derives `runs/<run_id>/gridmask_dataset/`, applies deterministic masks to the
copied images, preserves YOLO label files, rewrites that dataset's `data.yaml`, and trains from the
derived path. The frozen dataset version stays unchanged.

If Ultralytics is not installed, the run is still persisted and moves to `failed` with a clear log
message. This lets the UI and run history be tested without downloading model weights.

Enable `Auto threshold scan` in Training Setup to run the default thresholds
`0.15, 0.25, 0.35, 0.5, 0.65` after a successful training run. Each threshold creates a
normal prediction job, so the Prediction Analysis list, logs, and Experiment Dashboard threshold
table use the same review workflow as manual prediction jobs.

Enable `TTA` to pass Ultralytics `augment=True` into prediction jobs. This affects manual
prediction analysis, manual threshold scans, and automatic post-training threshold scans for the run.

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
4. Optionally set Image Browser filters, then enable `Use image filters` to run only that filtered
   subset inside the selected scope.
5. Click `Run Prediction Analysis`.
6. Review matched, false-positive, false-negative, and class-confusion counts.
7. Filter prediction/failure samples by failure type, class, confidence, platform, altitude, or timestamp.
8. Inspect the prediction/failure sample list; each row keeps the source image ID for annotator follow-up.
9. Click `Open Image` on a sample to jump to that image in the annotator.
10. Use the `GT` and `Pred` layer toggles to isolate saved boxes or model outputs.
11. Prediction boxes show failure type, class, and confidence on the image overlay.
12. Use `Add as annotation` to promote a false-positive prediction into an editable annotation draft.
13. Use `Mark reviewed` on false-negative rows to tag the matched ground-truth box for follow-up.
14. Click `Save Annotations` when the correction draft looks right.

Prediction jobs automatically add the `false_negative` edge tag to missed ground-truth boxes and
`class_confusion` to ground-truth boxes that were localized with high IoU but assigned the wrong
class. Use the Image Browser `Edge tag` filter with either tag to build follow-up annotation queues.
The manual `reviewed_prediction` tag is still only added when an operator clicks `Mark reviewed` and
saves the annotation changes.

For threshold tuning, enter comma-separated confidence values in `Scan thresholds`, such as
`0.15, 0.25, 0.35, 0.5, 0.65`, then click `Run Threshold Scan`. The backend creates one
prediction job for each value and the Experiment Dashboard automatically refreshes the
`Threshold Scan` table with precision and recall for each point.

Prediction jobs also show `Auto refresh on` while queued or running. The UI refreshes the latest job,
its prediction rows, and active job logs until the job reaches a terminal state.

API smoke for threshold scans:

```bash
curl -X POST http://127.0.0.1:8000/api/training/runs/1/prediction-threshold-scan \
  -H "Content-Type: application/json" \
  -d '{"image_scope":"all","thresholds":[0.15,0.25,0.35,0.5,0.65]}'

curl "http://127.0.0.1:8000/api/prediction-jobs/1/predictions?failure_type=false_positive&class_id=1&confidence_min=0.5&platform=group_a"
```

Prediction artifacts are written under:

```text
workspace/projects/<project_id>/runs/<run_id>/predictions/<job_id>/
  logs.txt
  predictions.json
```

`predictions.json` records the selected image scope, normalized image filters, image IDs, and
prediction rows so filtered experiments can be reproduced later.

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

## Experiment dashboard

`Run History` includes an `Experiment Dashboard` for the latest run. It summarizes:

1. Training metric series stored in `run_metrics`.
2. Per-class matched, false-positive, false-negative, and class-confusion counts from the latest completed prediction job.
3. A class-level confusion matrix for matched and class-confusion predictions.
4. Threshold scan rows across completed prediction jobs at different confidence thresholds.
5. A `Run Comparison` table across recent project runs with status, model, epochs, mAP50, box loss,
   latest prediction counts, best threshold/F1, and artifact path.

The dashboard is populated through:

```bash
curl http://127.0.0.1:8000/api/training/runs/1/summary
curl http://127.0.0.1:8000/api/projects/1/training/summary
```

## Model export

After a completed training run exists:

1. Open `Model Export`.
2. Check whether the latest run has `ultralytics/weights/best.pt`.
3. Use `Export PT` to copy/register the trained `.pt` weights under the run export folder.
4. Use `Export ONNX` when Ultralytics is installed and the weights file exists.
5. TensorRT is shown only when backend capability detection finds TensorRT support; otherwise the UI shows
   the unsupported reason.

Export artifacts are persisted in SQLite and written under:

```text
workspace/projects/<project_id>/runs/<run_id>/exports/
  run-<run_id>.pt
  run-<run_id>.onnx
  run-<run_id>.engine
```

API smoke:

```bash
curl http://127.0.0.1:8000/api/training/runs/1/exports/capabilities
curl -X POST http://127.0.0.1:8000/api/training/runs/1/exports \
  -H "Content-Type: application/json" \
  -d '{"format":"pt"}'
curl http://127.0.0.1:8000/api/training/runs/1/exports
```

## Model effect preview

After at least one training run exists, open the `效果预览` workflow step. Image mode lets you
choose the built-in `yolov8n.pt` or a completed run's `best.pt`, select a managed dataset, and step
through images one by one. Orange solid boxes are model predictions; green dashed boxes are saved
human annotations. Changing the confidence threshold only changes the preview request and never
writes annotations or formal prediction jobs.

Video mode accepts `mp4`, `mov`, `avi`, `mkv`, and `webm` uploads. Frames are inferred in the
background and streamed from the local API while processing. Completed results are saved under:

```text
workspace/projects/<project_id>/previews/<preview_id>/
  source.<ext>
  frames/
  result.mp4
```

The result video can be downloaded from the page or deleted as a whole. Deleting a preview removes
only this preview directory and its database row; the original upload, dataset images, annotations,
and training artifacts are not changed. Video uploads are limited to 500 MB.
