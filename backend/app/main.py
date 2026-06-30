from fastapi import FastAPI


app = FastAPI(title="YOLO Trainer API")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": "YOLO Trainer",
        "workspace_root": "",
        "database_path": "",
        "devices": {"selected": "cpu", "available": ["cpu"]},
    }
