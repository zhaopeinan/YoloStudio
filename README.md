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
