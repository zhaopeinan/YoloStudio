# Dataset Storage Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a recoverable storage management UI and backend for imported datasets, frozen annotated versions, and their related ended training runs.

**Architecture:** Add a `trash_items` table without altering existing tables, and use trash records to hide preserved source rows from normal workflows. A focused storage service owns canonical path validation, dependency checks, directory movement, restoration, purge ordering, expiry, and crash reconciliation. The dataset page receives a separate management component while `App.tsx` only coordinates loaded-workspace state.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, SQLite, React 18, TypeScript, Lucide React, Vitest, Testing Library, pytest

---

## File Structure

### Backend

- Modify `backend/app/db/models.py`: add the `TrashItem` model.
- Modify `backend/app/main.py`: register storage routes and run safe startup maintenance.
- Create `backend/app/storage/__init__.py`: package marker.
- Create `backend/app/storage/schemas.py`: typed active item, detail, trash, restore, and purge responses.
- Create `backend/app/storage/visibility.py`: shared active/trashed query predicates and guards.
- Create `backend/app/storage/service.py`: size calculation, dependency checks, path validation, move/restore/purge state machine, expiry, and reconciliation.
- Create `backend/app/storage/router.py`: `/api/storage` endpoints.
- Modify `backend/app/datasets/router.py`: exclude trashed datasets and guard dataset/image access.
- Modify `backend/app/quality/router.py`: reject trashed datasets.
- Modify `backend/app/versions/router.py`: exclude and guard trashed versions and parent datasets.
- Modify `backend/app/training/router.py`: exclude trashed runs and prevent training from trashed versions.
- Modify `backend/app/prediction/router.py`: reject trashed runs and datasets.
- Modify `backend/app/exports/router.py`: reject trashed runs.
- Create `backend/tests/test_storage_api.py`: lifecycle, dependency, purge, expiry, reconciliation, and path tests.
- Modify `backend/tests/test_settings_and_db.py`: assert table creation.
- Modify affected existing API tests only where trash visibility must be mocked or asserted.

### Frontend

- Modify `frontend/src/api.ts`: storage types and request functions; parse structured API errors.
- Create `frontend/src/StorageManagementView.tsx`: active/trash tabs, tables, drawer, and dialogs.
- Create `frontend/src/StorageManagementView.test.tsx`: focused component behavior.
- Modify `frontend/src/App.tsx`: import/management tabs and loaded-dataset cleanup callback.
- Modify `frontend/src/App.test.tsx`: dataset-page integration and workspace clearing.
- Modify `frontend/src/styles.css`: dense tables, drawer, dialogs, responsive compact rows.

### Task 1: Add the Trash Persistence Foundation

**Files:**
- Modify: `backend/app/db/models.py`
- Modify: `backend/tests/test_settings_and_db.py`

- [ ] **Step 1: Write the failing database table test**

Extend `test_init_db_creates_foundation_tables`:

```python
tables = set(inspect(engine).get_table_names())
assert {
    "projects",
    "datasets",
    "images",
    "export_artifacts",
    "trash_items",
}.issubset(tables)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_settings_and_db.py::test_init_db_creates_foundation_tables -v
```

Expected: FAIL because `trash_items` does not exist.

- [ ] **Step 3: Add the `TrashItem` model**

Add to `backend/app/db/models.py`:

```python
class TrashItem(TimestampMixin, Base):
    __tablename__ = "trash_items"
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_trash_items_entity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(Integer, index=True)
    version_id: Mapped[int | None] = mapped_column(Integer, index=True)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    trash_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    purge_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
```

Do not add `deleted_at` columns to existing tables.

- [ ] **Step 4: Run foundation tests**

```bash
cd backend
.venv/bin/pytest tests/test_settings_and_db.py -v
```

Expected: all settings and database tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/db/models.py backend/tests/test_settings_and_db.py
git commit -m "feat: add storage trash records"
```

### Task 2: Build Active Storage Catalog and Details

**Files:**
- Create: `backend/app/storage/__init__.py`
- Create: `backend/app/storage/schemas.py`
- Create: `backend/app/storage/visibility.py`
- Create: `backend/app/storage/service.py`
- Create: `backend/app/storage/router.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_storage_api.py`

- [ ] **Step 1: Write failing catalog tests**

Create fixtures in `backend/tests/test_storage_api.py` using `isolated_client`,
`create_import_zip`, annotations, a frozen version, and ended/active run rows.
Add assertions equivalent to:

```python
response = client.get("/api/storage/items")
assert response.status_code == 200
items = response.json()["items"]
dataset = next(item for item in items if item["entity_type"] == "dataset")
version = next(item for item in items if item["entity_type"] == "dataset_version")
assert dataset["image_count"] == 2
assert dataset["annotation_count"] == 2
assert dataset["size_bytes"] > 0
assert dataset["protected"] is True
assert dataset["blockers"][0]["entity_type"] == "dataset_version"
assert version["split_counts"] == {"train": 1, "val": 1, "test": 0}
assert version["protected"] is True

detail = client.get(f"/api/storage/items/dataset_version/{version['entity_id']}")
assert detail.status_code == 200
assert {run["status"] for run in detail.json()["related_runs"]} == {
    "completed",
    "running",
}
```

- [ ] **Step 2: Run tests and verify 404 failure**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -v
```

Expected: FAIL because `/api/storage/items` is not registered.

- [ ] **Step 3: Define schemas**

Create Pydantic models with these stable fields:

```python
class StorageBlocker(BaseModel):
    entity_type: str
    entity_id: int
    display_name: str
    status: str | None = None


class StorageItemRead(BaseModel):
    entity_type: str
    entity_id: int
    display_name: str
    project_id: int
    project_name: str
    dataset_id: int | None
    version_id: int | None
    artifact_path: str
    size_bytes: int
    image_count: int | None
    annotation_count: int | None
    split_counts: dict[str, int] | None
    created_at: datetime
    protected: bool
    blockers: list[StorageBlocker]


class StorageItemList(BaseModel):
    items: list[StorageItemRead]
    total_size_bytes: int


class RelatedRunRead(BaseModel):
    id: int
    status: str
    model: str
    size_bytes: int
    prediction_job_count: int
    export_count: int
    created_at: datetime


class StorageItemDetail(StorageItemRead):
    class_names: list[str]
    related_runs: list[RelatedRunRead]
```

- [ ] **Step 4: Implement visibility helpers**

In `visibility.py`, define:

```python
TRASHED_STATUSES = {"pending_move", "active", "pending_restore", "error"}


def active_entity_predicate(entity_type: str, entity_id_column):
    return ~exists().where(
        TrashItem.entity_type == entity_type,
        TrashItem.entity_id == entity_id_column,
        TrashItem.status.in_(TRASHED_STATUSES),
    )


def is_entity_trashed(db: Session, entity_type: str, entity_id: int) -> bool:
    return bool(
        db.scalar(
            select(TrashItem.id).where(
                TrashItem.entity_type == entity_type,
                TrashItem.entity_id == entity_id,
                TrashItem.status.in_(TRASHED_STATUSES),
            )
        )
    )
```

Also provide `require_active_entity` that raises a storage-domain not-found
exception used by routers.

- [ ] **Step 5: Implement catalog reads**

In `service.py`, add the shared constants and size helper:

```python
ACTIVE_RUN_STATUSES = {"queued", "preparing", "running"}
TRASH_RETENTION_DAYS = 30


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


```

Implement `list_storage_items(db, settings) -> StorageItemList` by selecting
active datasets and versions, joining project names, counting annotations and
runs with grouped subqueries, building blockers, calculating canonical directory
sizes, and sorting by creation time descending. Implement
`get_storage_item_detail(db, settings, entity_type, entity_id) ->
StorageItemDetail` from the same row builders and add selected class names and
non-trashed related runs. Dataset blockers are active versions. Version blockers
are active training runs. Related runs include both active and ended runs that
are not in trash. Resolve dataset paths from IDs, and validate version/run
artifact paths below the workspace before calculating size.

- [ ] **Step 6: Register read endpoints**

Create `router.py` with:

```python
router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/items", response_model=StorageItemList)
def list_items(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageItemList:
    return list_storage_items(db, settings)


@router.get("/items/{entity_type}/{entity_id}", response_model=StorageItemDetail)
def get_item(
    entity_type: str,
    entity_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StorageItemDetail:
    return get_storage_item_detail(db, settings, entity_type, entity_id)
```

Include the router in `app/main.py`.

- [ ] **Step 7: Run catalog tests**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -v
```

Expected: catalog and detail tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/storage backend/app/main.py backend/tests/test_storage_api.py
git commit -m "feat: add storage catalog API"
```

### Task 3: Move Objects to Trash with Dependency Protection

**Files:**
- Modify: `backend/app/storage/schemas.py`
- Modify: `backend/app/storage/service.py`
- Modify: `backend/app/storage/router.py`
- Modify: `backend/tests/test_storage_api.py`

- [ ] **Step 1: Write failing dependency and move tests**

Cover these cases:

```python
assert client.post(f"/api/storage/items/dataset/{dataset_id}/trash").status_code == 409
assert client.post(f"/api/storage/items/dataset_version/{version_id}/trash").status_code == 409
assert client.post(f"/api/storage/items/training_run/{running_run_id}/trash").status_code == 409

response = client.post(f"/api/storage/items/training_run/{completed_run_id}/trash")
assert response.status_code == 200
trash = response.json()
assert trash["entity_type"] == "training_run"
assert Path(trash["trash_path"]).exists()
assert not original_run_path.exists()
```

Then trash the unblocked version and dataset in child-to-parent order. Assert
each original directory moves below `workspace/.trash` and duplicate trash
requests return 409.

- [ ] **Step 2: Run target tests and verify failures**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -k "trash or protected" -v
```

Expected: FAIL because mutation endpoints do not exist.

- [ ] **Step 3: Add mutation response schemas**

```python
class TrashItemRead(BaseModel):
    id: int
    entity_type: str
    entity_id: int
    display_name: str
    project_id: int
    dataset_id: int | None
    version_id: int | None
    original_path: str
    trash_path: str
    size_bytes: int
    summary: dict
    status: str
    error_message: str | None
    deleted_at: datetime
    purge_after: datetime


class StorageConflictDetail(BaseModel):
    message: str
    blockers: list[StorageBlocker]
```

- [ ] **Step 4: Implement canonical path and move helpers**

In `service.py`, add:

```python
def require_workspace_path(path: Path, workspace_root: Path) -> Path:
    resolved = path.resolve()
    root = workspace_root.resolve()
    if not resolved.is_relative_to(root):
        raise StorageValidationError("数据路径不在当前工作空间内。")
    return resolved


```

Implement `expected_entity_path(settings, entity_type, entity) -> Path` with an
explicit branch for dataset, version, and training run. Compare the resolved
result with the exact ID-derived directory for that entity; stored artifact
paths may confirm the location but may not redirect it. Implement
`move_storage_item_to_trash(db, settings, entity_type, entity_id) -> TrashItem`
using the sequence below.

The move sequence is:

1. Validate entity type, object, status, blockers, canonical path, and duplicate
   trash record.
2. Calculate size and snapshot summary.
3. Insert and commit `pending_move` to obtain the trash ID.
4. Rename the source directory to
   `.trash/<trash-id>-<entity-type>-<entity-id>`.
5. Mark the record `active` and commit.
6. If the rename fails while the source remains, delete the pending record and
   return an error. If the rename succeeded but the final commit fails, leave
   `pending_move` for reconciliation.

- [ ] **Step 5: Add the trash endpoint and structured 409 errors**

```python
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
```

- [ ] **Step 6: Run move tests**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -k "trash or protected" -v
```

Expected: all trash and dependency tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/storage backend/tests/test_storage_api.py
git commit -m "feat: move storage items to trash"
```

### Task 4: Hide Trashed Objects from Normal Workflows

**Files:**
- Modify: `backend/app/datasets/router.py`
- Modify: `backend/app/quality/router.py`
- Modify: `backend/app/versions/router.py`
- Modify: `backend/app/training/router.py`
- Modify: `backend/app/prediction/router.py`
- Modify: `backend/app/exports/router.py`
- Modify: `backend/tests/test_storage_api.py`
- Modify affected existing backend tests where necessary.

- [ ] **Step 1: Write failing visibility tests**

After trashing each entity, assert:

```python
projects = client.get("/api/projects").json()["items"]
assert all(dataset["id"] != dataset_id for project in projects for dataset in project["datasets"])
assert client.get(f"/api/datasets/{dataset_id}/images").status_code == 404
assert client.get(f"/api/datasets/{dataset_id}/quality").status_code == 404

assert client.get(f"/api/datasets/{dataset_id}/versions").json()["items"] == []
assert client.post(
    "/api/training/runs",
    json={"version_id": version_id, "epochs": 1, "image_size": 320, "batch_size": 1},
).status_code == 404

runs = client.get(f"/api/projects/{project_id}/training/runs").json()["items"]
assert all(run["id"] != run_id for run in runs)
assert client.get(f"/api/training/runs/{run_id}").status_code == 404
assert client.get(f"/api/training/runs/{run_id}/exports").status_code == 404
```

Also assert image-file and prediction endpoints reject images/runs whose parent
is trashed.

- [ ] **Step 2: Run visibility tests and verify failures**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -k "hidden or visibility" -v
```

Expected: FAIL because existing routers query source tables directly.

- [ ] **Step 3: Apply shared visibility predicates and guards**

Use `active_entity_predicate` in list queries and `require_active_entity` in
single-object routes. Do not duplicate ad hoc `TrashItem` queries across every
router. Required entity names are:

```python
ENTITY_DATASET = "dataset"
ENTITY_VERSION = "dataset_version"
ENTITY_RUN = "training_run"
```

When serving an image or annotation, join its dataset and verify the dataset is
active. When serving prediction/export resources, verify the parent run is
active. Version creation and training creation must verify active parents.

- [ ] **Step 4: Run storage and existing API suites**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py tests/test_dataset_import_api.py \
  tests/test_quality_version_api.py tests/test_training_api.py \
  tests/test_prediction_api.py tests/test_exports_api.py -v
```

Expected: all selected suites pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/datasets/router.py backend/app/quality/router.py \
  backend/app/versions/router.py backend/app/training/router.py \
  backend/app/prediction/router.py backend/app/exports/router.py \
  backend/tests
git commit -m "feat: hide trashed data from workflows"
```

### Task 5: Restore, Permanent Purge, Expiry, and Reconciliation

**Files:**
- Modify: `backend/app/storage/schemas.py`
- Modify: `backend/app/storage/service.py`
- Modify: `backend/app/storage/router.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_storage_api.py`

- [ ] **Step 1: Write failing restore tests**

Test successful dataset restore and parent ordering:

```python
response = client.post(f"/api/storage/trash/{dataset_trash_id}/restore")
assert response.status_code == 200
assert original_dataset_path.exists()
assert not trash_path.exists()

blocked = client.post(f"/api/storage/trash/{run_trash_id}/restore")
assert blocked.status_code == 409
assert blocked.json()["detail"]["message"].startswith("请先恢复标注版本")
```

Test an occupied original path returns 409 and leaves the trash item active.

- [ ] **Step 2: Write failing purge and expiry tests**

Cover:

```python
wrong = client.request(
    "DELETE",
    f"/api/storage/trash/{trash_id}",
    json={"confirm_name": "wrong"},
)
assert wrong.status_code == 400

purged = client.request(
    "DELETE",
    f"/api/storage/trash/{trash_id}",
    json={"confirm_name": display_name},
)
assert purged.status_code == 200
```

Create run, version, and dataset trash records with expired `purge_after` and
assert explicit cleanup purges them child-first. A parent with a remaining child
must remain with an error instead of violating foreign keys.

- [ ] **Step 3: Write failing reconciliation tests**

Create pending records and filesystem states for all four reconciliation cases:

- Pending move with source only removes the trash record.
- Pending move with trash only becomes active.
- Pending restore with trash only becomes active.
- Pending restore with source only deletes the trash record.
- Both or neither paths marks `error` and preserves the record.

- [ ] **Step 4: Implement restore**

Implement `restore_trash_item(db, settings, trash_id) -> StorageMutationResponse`.
Load an active trash row, validate its active parent, canonical source/trash
paths, source absence, and trash presence. Commit `pending_restore`, rename the
directory, then delete the trash record. If final database deletion fails, leave
`pending_restore` for startup reconciliation.

- [ ] **Step 5: Implement permanent purge**

```python
class TrashPurgeRequest(BaseModel):
    confirm_name: str
```

Implement `purge_trash_item(db, settings, trash_id, confirm_name) ->
StorageMutationResponse`. Reject a confirmation mismatch before touching disk.
Rename the trash path to `.trash/.purging/<trash-id>`, delete child database rows
in the approved order, and commit. Restore the staged directory when the
transaction fails. After commit, remove the staged path. Training-run purge
deletes predictions, jobs, exports, metrics, then the run. Version purge requires
zero run rows. Dataset purge requires zero version rows and deletes annotations,
images, then the dataset. Never delete project classes or projects.

- [ ] **Step 6: Implement expiry and reconciliation**

Implement `purge_expired_trash(db, settings, now=None) -> PurgeSummary` and
`reconcile_trash(db, settings) -> ReconcileSummary`. Expiry sorts types by
`training_run`, `dataset_version`, then `dataset` and continues after individual
failures. Reconciliation applies every state/path combination from Step 3.
Startup removes abandoned `.purging` directories only when no trash or source
entity record remains.

- [ ] **Step 7: Add endpoints and startup maintenance**

Add:

```python
@router.get("/trash", response_model=TrashItemList)
def list_trash(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TrashItemList:
    reconcile_trash(db, settings)
    purge_expired_trash(db, settings)
    return list_trash_items(db)


@router.post("/trash/{trash_id}/restore", response_model=StorageMutationResponse)
def restore_item(trash_id: int, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    return restore_trash_item(db, settings, trash_id)


@router.delete("/trash/{trash_id}", response_model=StorageMutationResponse)
def purge_item(
    trash_id: int,
    request: TrashPurgeRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    return purge_trash_item(db, settings, trash_id, request.confirm_name)


@router.post("/trash/purge-expired", response_model=PurgeSummary)
def purge_expired(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    return purge_expired_trash(db, settings)
```

In the application lifespan, open a database session after `init_db()`, run
reconciliation and expiry, record individual failures, and always allow startup
to continue.

- [ ] **Step 8: Run lifecycle tests**

```bash
cd backend
.venv/bin/pytest tests/test_storage_api.py -v
```

Expected: all catalog, trash, restore, purge, expiry, and reconciliation tests pass.

- [ ] **Step 9: Commit**

```bash
git add backend/app/storage backend/app/main.py backend/tests/test_storage_api.py
git commit -m "feat: restore and purge trashed data"
```

### Task 6: Add Frontend Storage API and Focused Management Component

**Files:**
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/StorageManagementView.tsx`
- Create: `frontend/src/StorageManagementView.test.tsx`

- [ ] **Step 1: Write failing component tests**

Mock storage API functions and test:

```tsx
render(<StorageManagementView loadedDatasetId={1} onDatasetTrashed={onDatasetTrashed} />);

expect(await screen.findByRole("table", { name: "现有数据" })).toBeInTheDocument();
expect(screen.getByText("camouflage-set")).toBeInTheDocument();
expect(screen.getByText("2 张图像 / 2 个标注")).toBeInTheDocument();
expect(screen.getByText("受保护")).toBeInTheDocument();

await user.click(screen.getByRole("button", { name: "查看 camouflage-set" }));
expect(await screen.findByRole("dialog", { name: "数据详情" })).toBeInTheDocument();

await user.click(screen.getByRole("tab", { name: "回收站" }));
expect(await screen.findByText("29 天后清理")).toBeInTheDocument();
```

Also test disabled protected trash action, ended-run trash from the drawer,
restore, permanent-name confirmation, and failed API messages retaining dialogs.

- [ ] **Step 2: Run component tests and verify module failure**

```bash
cd frontend
npm test -- --run src/StorageManagementView.test.tsx
```

Expected: FAIL because the component and API functions do not exist.

- [ ] **Step 3: Add typed API functions**

Define types matching backend schemas and add:

```typescript
export function listStorageItems(): Promise<StorageItemListResponse>;
export function getStorageItem(
  entityType: StorageEntityType,
  entityId: number,
): Promise<StorageItemDetail>;
export function trashStorageItem(
  entityType: StorageEntityType,
  entityId: number,
): Promise<TrashItem>;
export function listTrashItems(): Promise<TrashItemListResponse>;
export function restoreTrashItem(trashId: number): Promise<StorageMutationResponse>;
export function purgeTrashItem(
  trashId: number,
  confirmName: string,
): Promise<StorageMutationResponse>;
```

Improve `requestJson` so FastAPI `{detail: {message, blockers}}` and
`{detail: "message"}` responses throw the user-facing message rather than raw
JSON text. Preserve existing callers.

- [ ] **Step 4: Build `StorageManagementView`**

Use Lucide `Database`, `HardDrive`, `Eye`, `Trash2`, `RotateCcw`, `Info`,
`MoreHorizontal`, and `X`. The component owns:

```typescript
type Props = {
  loadedDatasetId: number | null;
  onDatasetTrashed: (datasetId: number) => void;
  onStorageChanged: () => void | Promise<void>;
};
```

Render active/trash segmented tabs, dense tables, a right-side drawer, a normal
trash confirmation dialog, and an exact-name permanent purge dialog. Use title
attributes for full paths and icon tooltips. Do not render instructional feature
copy or nested cards.

- [ ] **Step 5: Run component tests**

```bash
cd frontend
npm test -- --run src/StorageManagementView.test.tsx
```

Expected: all focused component tests pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.ts frontend/src/StorageManagementView.tsx \
  frontend/src/StorageManagementView.test.tsx
git commit -m "feat: add storage management interface"
```

### Task 7: Integrate Management into the Dataset Page

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write failing App integration tests**

Add mocked storage API functions to the existing hoisted mock and assert:

```tsx
expect(await screen.findByRole("tab", { name: "导入数据" })).toBeInTheDocument();
await user.click(screen.getByRole("tab", { name: "数据管理" }));
expect(await screen.findByRole("table", { name: "现有数据" })).toBeInTheDocument();
```

Load dataset 1, trash it through the management view, then assert the loaded
summary disappears, project data refreshes, and the workflow reports that no
dataset is loaded.

- [ ] **Step 2: Run App tests and verify failure**

```bash
cd frontend
npm test -- --run src/App.test.tsx
```

Expected: FAIL because the dataset page has no tabs or management integration.

- [ ] **Step 3: Add dataset-page tab state**

In `App.tsx`, add:

```typescript
const [datasetPageTab, setDatasetPageTab] = useState<"import" | "manage">("import");
```

Render a segmented tablist above dataset content. Keep the current import panel
unchanged under `import`. Render `StorageManagementView` under `manage`.

- [ ] **Step 4: Add refresh and loaded-dataset callbacks**

On storage change, refresh projects and the current dataset workspace lists. If
the trashed dataset matches `importedDataset.dataset_id`, clear imported dataset,
images, annotations, quality, versions, runs, predictions, exports, and workflow
readiness using one focused helper rather than repeated setters.

- [ ] **Step 5: Add responsive styling**

Add stable classes for:

- Dataset primary tabs and active state.
- Active/trash segmented controls.
- Dense management table with fixed action column.
- Path ellipsis and tabular numeric size/date columns.
- Fixed right drawer with header/body/footer.
- Modal backdrop and confirmation layout.
- Mobile compact rows and overflow action menu below the existing breakpoint.

Keep cards at 8px radius or less, avoid nested cards, and ensure no controls or
text overlap at desktop and mobile widths.

- [ ] **Step 6: Run frontend tests and build**

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: all Vitest tests pass and TypeScript/Vite build succeeds.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "feat: integrate dataset storage management"
```

### Task 8: Full Verification, Browser Review, and Service Restart

**Files:**
- Verify all modified backend and frontend files.

- [ ] **Step 1: Run complete backend tests**

```bash
cd backend
.venv/bin/pytest
```

Expected: all backend tests pass.

- [ ] **Step 2: Run complete frontend tests and build**

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: all frontend tests pass and production build succeeds.

- [ ] **Step 3: Check patch cleanliness**

```bash
cd ~/DevProjects/YOLO_Trainer
git diff --check
git status --short
```

Expected: no whitespace errors. Preserve unrelated untracked files.

- [ ] **Step 4: Restart persistent services**

```bash
launchctl remove com.codex.yolo-trainer.backend || true
launchctl submit -l com.codex.yolo-trainer.backend -- /bin/zsh -lc \
  'cd ~/DevProjects/YOLO_Trainer/backend && exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >>/tmp/yolo-trainer-backend.log 2>&1'

launchctl remove com.codex.yolo-trainer.frontend || true
launchctl submit -l com.codex.yolo-trainer.frontend -- /bin/zsh -lc \
  'cd ~/DevProjects/YOLO_Trainer/frontend && exec npm run dev -- --host 127.0.0.1 --port 5173 >>/tmp/yolo-trainer-frontend.log 2>&1'
```

Verify `/api/health` and `http://127.0.0.1:5173/` return successfully.

- [ ] **Step 5: Verify desktop flow in the browser**

At `1440x900`, verify:

1. `项目与数据 → 数据管理` opens without changing workflow hash.
2. Active rows show correct dataset/version counts, size, and protection state.
3. Detail drawer paths truncate without overflow and related runs are usable.
4. Ended run can move to trash; active run cannot.
5. Child-to-parent trash, restore ordering, and permanent confirmation behave as
   designed.
6. Current workspace clears only when its loaded dataset is trashed.

- [ ] **Step 6: Verify mobile flow in the browser**

At `390x844`, verify compact rows, action menus, drawer width, modal fit, path
wrapping, and no horizontal overlap. Capture screenshots for both viewports.

- [ ] **Step 7: Final safety check**

Use disposable test entities for destructive browser verification. Do not trash
or purge the user's existing datasets, versions, training runs, or
`image_dataset.zip`.
