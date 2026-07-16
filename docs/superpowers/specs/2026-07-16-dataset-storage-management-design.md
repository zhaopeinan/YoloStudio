# Dataset Storage Management Design

## Goal

Add a local, single-user storage management module for imported datasets and
frozen annotated dataset versions. Users can inspect disk usage and dependency
status, move unused data into a recoverable trash area, restore it, or delete it
permanently. The module also exposes related completed training runs so users
can remove blockers before deleting a version.

## Product Scope

### Managed objects

- Imported datasets and their source image directories.
- Frozen annotated dataset versions and their exported YOLO images, labels,
  manifest, and `data.yaml`.
- Completed, failed, or cancelled training runs when they are managed as a
  dependency of a dataset version. Prediction jobs, predictions, metrics,
  exports, logs, and model artifacts move and purge with their parent run.

### Non-goals

- No batch deletion.
- No general-purpose file browser.
- No deletion of projects or project-level class libraries.
- No deletion of active queued, preparing, or running tasks.
- No separate top-level workflow step.

## User Experience

The existing `项目与数据` page gains two primary tabs:

- `导入数据`: preserves the current scan, import, and saved dataset loader.
- `数据管理`: provides storage inspection and deletion workflows.

The management tab contains two views:

- `现有数据`: a unified table of imported datasets and frozen versions.
- `回收站`: all recoverable dataset, version, and training-run trash entries.

### Existing data table

Each row displays:

- Name and object type (`原始数据集` or `标注版本`).
- Project name.
- Image and annotation counts, or train/validation/test split counts.
- Actual disk usage.
- Creation time.
- Dependency state (`可清理` or `受保护`).
- Detail and trash actions.

Clicking a row opens a right-side detail drawer. Dataset details show full path,
image count, annotation count, versions, and classes. Version details show its
path, selected classes, split counts, and related training runs. Related runs
show status, model, dates, prediction/export counts, and disk usage.

The trash button uses the Lucide trash icon with a tooltip. Protected rows keep
the button disabled and expose the blocking dependency count. Completed,
failed, and cancelled related runs can be moved to trash from the version
drawer. Active runs cannot be removed.

### Trash view

Each row displays object type, name, original path, size, deletion time, and
days remaining before automatic purge. Actions are restore and permanent
delete. Permanent deletion requires entering the exact object name in a modal.

Restoration follows parent dependencies:

- A training run requires its dataset version to be active.
- A dataset version requires its imported dataset to be active.
- If the parent is still trashed, restoration is blocked with the parent name.

If the currently loaded dataset is moved to trash, the frontend clears the
loaded workspace and returns to the data management table.

On narrow screens, the table becomes compact rows and places actions in a menu.
The desktop layout remains a dense operational table rather than a card grid.

## Dependency Rules

Dependency protection considers only objects that are not in trash:

- A dataset is protected while it has any active frozen version.
- A version is protected while it has any active training run.
- A training run is protected while its status is `queued`, `preparing`, or
  `running`.

Users clear dependencies from child to parent:

1. Move ended training runs to trash.
2. Move the now-unreferenced version to trash.
3. Move the now-unreferenced dataset to trash.

Trash entries retain their underlying database rows so foreign keys and
historical summaries remain valid while recovery is possible. Permanent purge
uses the same child-to-parent order. A parent cannot be permanently deleted
while any child database row still exists, including a child that remains in
trash.

## Data Model

Add a `TrashItem` SQLAlchemy model and `trash_items` table. No existing table
needs a schema alteration, so `Base.metadata.create_all()` can add the table to
the current SQLite database.

Fields:

- `id`: primary key.
- `entity_type`: `dataset`, `dataset_version`, or `training_run`.
- `entity_id`: identifier in the source table.
- `project_id`: owning project.
- `dataset_id`: nullable dataset identifier for parent checks.
- `version_id`: nullable version identifier for parent checks.
- `display_name`: immutable confirmation and display name.
- `original_path`: absolute canonical artifact directory.
- `trash_path`: absolute canonical path below `workspace/.trash`.
- `size_bytes`: captured before the move.
- `summary`: JSON snapshot used by the trash list.
- `status`: `pending_move`, `active`, `pending_restore`, or `error`.
- `error_message`: nullable reconciliation or purge error.
- `deleted_at`: deletion timestamp.
- `purge_after`: deletion timestamp plus 30 days.
- standard creation and update timestamps.

Only one non-restored trash item may exist for an entity type and entity ID.
Normal dataset, version, training, prediction, and export queries exclude rows
whose entity has a trash item in a non-restored state.

## Filesystem Layout

Trash content is stored under:

```text
workspace/.trash/<trash-id>-<entity-type>-<entity-id>/
```

Canonical source directories are:

- Dataset: `workspace/projects/<project-id>/datasets/<dataset-id>`.
- Version: its existing `artifact_path` below
  `workspace/projects/<project-id>/versions/<version-id>`.
- Training run: its existing `artifact_path` below
  `workspace/projects/<project-id>/runs/<run-id>`.

Every operation resolves and verifies paths against `workspace_root`. Absolute
paths outside the workspace, traversal, symlink escape, unexpected entity
directories, and trash paths outside `.trash` are rejected.

Directory movement uses a same-volume rename when possible. The backend stores
an operation status because SQLite and filesystem changes cannot share a single
atomic transaction.

### Move reconciliation

On startup and before listing trash, reconcile incomplete operations:

- `pending_move`, original exists, trash absent: cancel the trash record.
- `pending_move`, original absent, trash exists: mark the item active.
- `pending_restore`, original absent, trash exists: return the item to active.
- `pending_restore`, original exists, trash absent: finish restoration by
  deleting the trash record.
- Both paths present or both paths missing: mark the item `error` and require
  manual review; never delete either location automatically.

## Backend Services

Create a focused `app/storage` package:

- `schemas.py`: list, detail, trash, restore, and purge response models.
- `service.py`: dependency checks, directory sizing, canonical path resolution,
  move/restore state machine, reconciliation, and database purge routines.
- `router.py`: `/api/storage` endpoints and HTTP error translation.

### API

- `GET /api/storage/items`: list active datasets and versions with counts,
  sizes, and protection reasons.
- `GET /api/storage/items/{entity_type}/{entity_id}`: return drawer details and
  related runs.
- `POST /api/storage/items/{entity_type}/{entity_id}/trash`: validate and move a
  dataset, version, or ended run to trash.
- `GET /api/storage/trash`: reconcile, purge expired items, then list remaining
  entries.
- `POST /api/storage/trash/{trash_id}/restore`: validate parent and path, then
  restore.
- `DELETE /api/storage/trash/{trash_id}`: require `confirm_name`, enforce child
  ordering, purge files and database rows.
- `POST /api/storage/trash/purge-expired`: run explicit expiry cleanup and
  return purged and failed counts.

The application lifespan initializes the table, reconciles pending operations,
and attempts expired cleanup. Failure to clean one entry is recorded on that
entry and does not prevent the application from starting.

## Permanent Purge Order

### Training run

Delete database rows in this order:

1. Predictions for the run.
2. Prediction jobs.
3. Export artifacts.
4. Run metrics.
5. Training run.
6. Trash item.

### Dataset version

Require no training-run rows, active or trashed. Then delete the version and
trash item.

### Dataset

Require no version rows, active or trashed. Predictions must already be absent
because their runs and versions were purged first. Delete annotations for its
images, images, dataset, and trash item. Project classes and the project remain.

Before database deletion, atomically rename the trash directory into
`workspace/.trash/.purging/<trash-id>`. If the database transaction fails, move
that directory back and retain the trash record. After a successful database
commit, remove the staged purge directory. Startup cleanup removes abandoned
`.purging` directories whose trash and source records no longer exist. This
keeps permanent deletion recoverable until the database commit succeeds.

## Frontend Integration

Add typed storage API functions to `frontend/src/api.ts` and focused components:

- `DatasetPageTabs`: switches import and management without changing the URL
  workflow hash.
- `StorageManagementView`: owns active/trash view selection and refresh.
- `StorageItemsTable`: dense active object table.
- `StorageDetailDrawer`: details and related-run management.
- `TrashItemsTable`: restore and purge actions.
- `StorageConfirmDialog`: trash confirmation and name-confirmed permanent purge.

`App.tsx` remains responsible for clearing the loaded dataset after it is
trashed and refreshing projects, versions, runs, and workflow readiness after
storage operations. The storage components own only their local view state.

All visible labels and errors are precise Chinese. Icon-only actions use Lucide
icons and tooltips. Long paths use tail truncation with the full path in a
title/tooltip, matching the existing workspace path treatment.

## Error Handling

Expected conflicts return HTTP 409 with structured detail including a Chinese
message and blocker summaries. Missing entities return 404. Invalid paths or
confirmation names return 400. Filesystem failures return 500 while preserving
the recoverable trash record and error state.

The frontend keeps dialogs open when an operation fails and displays the
backend message. It refreshes lists only after a confirmed successful response.

## Testing

Backend coverage includes:

- Active list counts, sizes, and protection reasons.
- Dataset protection by versions.
- Version protection by active runs.
- Active-run deletion rejection.
- Dataset, version, and ended-run moves.
- Normal query exclusion after trashing.
- Restore success and parent-restore ordering.
- Restore path conflict.
- Exact-name permanent deletion.
- Database purge order for runs, versions, and datasets.
- Thirty-day expiry and child-first cleanup.
- Pending move/restore reconciliation.
- Workspace path, traversal, alias, and symlink escape rejection.
- Filesystem failure compensation and error recording.

Frontend coverage includes:

- Import/management and active/trash tab switching.
- Active table rendering and size formatting.
- Protected action state and blocker display.
- Detail drawer and related-run deletion.
- Trash confirmation, restore, and exact-name purge confirmation.
- Loaded workspace clearing after its dataset is trashed.
- Responsive action menu behavior.

Run the complete backend and frontend test suites, then verify the desktop and
mobile management flows in the browser before completion.
