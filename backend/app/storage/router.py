from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.session import get_db
from app.storage.schemas import StorageItemDetail, StorageItemList
from app.storage.service import (
    StoragePathError,
    get_storage_item_detail,
    list_storage_items,
)
from app.storage.visibility import StorageEntityNotFoundError


router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/items", response_model=StorageItemList)
def list_items(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageItemList:
    try:
        return list_storage_items(db, settings)
    except StoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/items/{entity_type}/{entity_id}", response_model=StorageItemDetail)
def get_item(
    entity_type: str,
    entity_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageItemDetail:
    try:
        return get_storage_item_detail(db, settings, entity_type, entity_id)
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
