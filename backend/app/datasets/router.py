from fastapi import APIRouter, HTTPException

from app.datasets.scanner import scan_dataset_zip
from app.datasets.schemas import DatasetScanRequest, DatasetScanSummary


router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/scan", response_model=DatasetScanSummary)
def scan_dataset(request: DatasetScanRequest) -> DatasetScanSummary:
    try:
        return scan_dataset_zip(request.source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset archive was not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
