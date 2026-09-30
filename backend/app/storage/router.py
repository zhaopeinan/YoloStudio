from fastapi import APIRouter, Depends, HTTPException

from app.auth.deps import get_current_user, require_admin
from sqlalchemy.orm import Session

from app.core.settings import Settings, get_settings
from app.db.session import get_db
from app.storage.schemas import (
    PurgeSummary,
    StorageItemDetail,
    StorageItemList,
    StorageMutationResponse,
    TrashItemList,
    TrashItemRead,
    TrashPurgeRequest,
)
from app.storage.service import (
    StorageConfirmationError,
    StorageConflictError,
    StorageMoveError,
    StoragePathError,
    cascade_purge_trash_item,
    get_storage_item_detail,
    list_trash_items,
    list_storage_items,
    move_storage_item_to_trash,
    purge_expired_trash,
    purge_trash_item,
    reconcile_trash,
    restore_trash_item,
)
from app.storage.visibility import StorageEntityNotFoundError


router = APIRouter(
    prefix="/api/storage",
    tags=["storage"],
    dependencies=[Depends(get_current_user), Depends(require_admin)],
)


def _raise_storage_error(exc: Exception) -> None:
    if isinstance(exc, StorageConflictError):
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "blockers": exc.blockers},
        ) from exc
    if isinstance(exc, StorageEntityNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (StoragePathError, StorageConfirmationError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, StorageMoveError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


@router.get("/items", response_model=StorageItemList)
def list_items(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageItemList:
    try:
        return list_storage_items(db, settings)
    except StoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/items/{entity_type}/{entity_id}/trash",
    response_model=TrashItemRead,
)
def trash_item(
    entity_type: str,
    entity_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TrashItemRead:
    try:
        return move_storage_item_to_trash(db, settings, entity_type, entity_id)
    except StorageConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "blockers": exc.blockers},
        ) from exc
    except StorageEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StorageMoveError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


@router.get("/trash", response_model=TrashItemList)
def list_trash(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TrashItemList:
    try:
        reconcile_trash(db, settings)
        purge_expired_trash(db, settings)
        return list_trash_items(db)
    except Exception as exc:
        _raise_storage_error(exc)


@router.post("/trash/purge-expired", response_model=PurgeSummary)
def purge_expired(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> PurgeSummary:
    try:
        reconcile_trash(db, settings)
        return purge_expired_trash(db, settings)
    except Exception as exc:
        _raise_storage_error(exc)


@router.post("/trash/{trash_id}/restore", response_model=StorageMutationResponse)
def restore_item(
    trash_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageMutationResponse:
    try:
        return restore_trash_item(db, settings, trash_id)
    except Exception as exc:
        _raise_storage_error(exc)


@router.delete("/trash/{trash_id}", response_model=StorageMutationResponse)
def purge_item(
    trash_id: int,
    request: TrashPurgeRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageMutationResponse:
    try:
        return purge_trash_item(db, settings, trash_id, request.confirm_name)
    except Exception as exc:
        _raise_storage_error(exc)


@router.post(
    "/trash/{trash_id}/cascade-purge",
    response_model=StorageMutationResponse,
)
def cascade_purge_item(
    trash_id: int,
    request: TrashPurgeRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageMutationResponse:
    try:
        return cascade_purge_trash_item(db, settings, trash_id, request.confirm_name)
    except Exception as exc:
        _raise_storage_error(exc)
