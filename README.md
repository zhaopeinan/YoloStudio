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
