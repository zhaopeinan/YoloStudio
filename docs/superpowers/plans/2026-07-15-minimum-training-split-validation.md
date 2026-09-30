# Minimum Training Split Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject dataset versions that cannot provide both training and validation images, and fail legacy invalid versions before Ultralytics starts.

**Architecture:** Keep the minimum-sample rule in the dataset version exporter because selected-class filtering determines the eligible image count there. Add a focused training preflight helper in the runner that validates the frozen manifest and exported files before importing or invoking Ultralytics. Existing API error propagation and run logging remain unchanged.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, pytest, Ultralytics YOLO

---

## File Structure

- Modify `backend/app/versions/exporter.py`: enforce two eligible annotated images before persisting a version.
- Modify `backend/app/training/runner.py`: validate `train` and `val` manifest entries and artifacts before loading Ultralytics.
- Modify `backend/tests/test_quality_version_api.py`: add one-image rejection coverage and update class-subset and unreadable-dimension fixtures to remain valid two-image exports.
- Modify `backend/tests/test_training_api.py`: make the shared training version fixture valid and test legacy invalid-version preflight behavior.

### Task 1: Enforce the Minimum at Version Creation

**Files:**
- Modify: `backend/tests/test_quality_version_api.py`
- Modify: `backend/app/versions/exporter.py`

- [ ] **Step 1: Write the failing one-image API test**

Add a test that imports the existing two-image sample, annotates only one image, and attempts to create a version:

```python
def test_create_dataset_version_requires_train_and_validation_images(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)

    with isolated_client(tmp_path) as client:
        dataset = _import_dataset(client, zip_path)
        class_payload = _create_class(client, dataset["project_id"], "drone")
        image = client.get(
            f"/api/datasets/{dataset['dataset_id']}/images"
        ).json()["items"][0]
        annotation_response = client.put(
            f"/api/images/{image['id']}/annotations",
            json={
                "annotations": [
                    {
                        "class_id": class_payload["id"],
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.25,
                        "height": 0.25,
                    }
                ]
            },
        )
        assert annotation_response.status_code == 200

        response = client.post(
            f"/api/datasets/{dataset['dataset_id']}/versions",
            json={"name": "too-small"},
        )

        assert response.status_code == 400
        assert "at least 2 annotated images are required" in response.text
        assert "current selection has 1" in response.text
        assert client.get(
            f"/api/datasets/{dataset['dataset_id']}/versions"
        ).json()["items"] == []
        versions_root = (
            tmp_path
            / "workspace"
            / "projects"
            / str(dataset["project_id"])
            / "versions"
        )
        assert not versions_root.exists()
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_quality_version_api.py::test_create_dataset_version_requires_train_and_validation_images -v
```

Expected: FAIL because the current exporter creates a `train=1, val=0` version.

- [ ] **Step 3: Add the exporter validation**

In `backend/app/versions/exporter.py`, define the requirement near `SPLITS` and validate immediately after querying `annotated_images`, before creating `DatasetVersion`:

```python
MIN_ANNOTATED_IMAGES = 2


eligible_image_count = len(annotated_images)
if eligible_image_count < MIN_ANNOTATED_IMAGES:
    raise VersionExportError(
        "Cannot create dataset version: at least "
        f"{MIN_ANNOTATED_IMAGES} annotated images are required for training and "
        f"validation; current selection has {eligible_image_count}."
    )
```

Retain the existing selected-class query so the count reflects only images usable by the requested class subset.

- [ ] **Step 4: Update existing valid-export fixtures**

In `test_create_dataset_version_can_freeze_selected_class_subset`, include a drone annotation on the second image as well as the decoy annotation, then update expectations to:

```python
assert version["split_counts"] == {"train": 1, "val": 1, "test": 0}
assert len(manifest["images"]) == 2
assert {item["image_id"] for item in manifest["images"]} == {
    first_image_id,
    second_image_id,
}
assert len(label_files) == 2
```

Change `_create_unreadable_dimension_zip` to write two unreadable JPEG images and annotate both images in `test_dataset_quality_reports_missing_image_dimensions_without_blocking_export`. Keep the warning assertions focused on unreadable dimensions and assert the two-image version still exports successfully.

- [ ] **Step 5: Run version and quality tests**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_quality_version_api.py -v
```

Expected: all tests pass, including `train=1, val=1` for every successful two-image version.

- [ ] **Step 6: Commit the version validation**

```bash
git add backend/app/versions/exporter.py backend/tests/test_quality_version_api.py
git commit -m "fix: require training and validation images"
```

### Task 2: Fail Invalid Frozen Versions Before Ultralytics

**Files:**
- Modify: `backend/tests/test_training_api.py`
- Modify: `backend/app/training/runner.py`

- [ ] **Step 1: Make the shared training fixture create a valid version**

Update `_create_version` in `backend/tests/test_training_api.py` to annotate both imported images:

```python
images = client.get(
    f"/api/datasets/{dataset['dataset_id']}/images"
).json()["items"]
for image in images:
    annotation_response = client.put(
        f"/api/images/{image['id']}/annotations",
        json={
            "annotations": [
                {
                    "class_id": class_payload["id"],
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.4,
                    "height": 0.4,
                }
            ]
        },
    )
    assert annotation_response.status_code == 200
```

Assert the resulting version has `{"train": 1, "val": 1, "test": 0}`.

- [ ] **Step 2: Write the failing legacy-version preflight test**

Create a valid version, remove its validation image artifact, and replace its persisted manifest with an empty validation split through a SQLAlchemy session. Install a fake `ultralytics.YOLO` whose constructor records calls, then start training and assert:

```python
assert run["status"] == "failed"
assert run["error_message"] == (
    "Dataset version requires at least one training image and one validation image. "
    "Annotate at least 2 images and create a new version."
)
assert yolo_constructor_calls == []
assert "ultralytics training started" not in logs
```

This proves validation happens before model loading or training.

- [ ] **Step 3: Run the preflight test and verify it fails**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_training_api.py::test_training_rejects_version_without_validation_split_before_ultralytics -v
```

Expected: FAIL because the current runner reaches `YOLO(...)` before checking split contents.

- [ ] **Step 4: Implement the focused preflight helper**

Add the following constants and helper to `backend/app/training/runner.py`:

```python
REQUIRED_TRAINING_SPLITS = ("train", "val")
INVALID_TRAINING_SPLIT_MESSAGE = (
    "Dataset version requires at least one training image and one validation image. "
    "Annotate at least 2 images and create a new version."
)


def validate_training_dataset(version: DatasetVersion, version_root: Path) -> None:
    manifest_images = version.split_manifest.get("images", [])
    for split in REQUIRED_TRAINING_SPLITS:
        split_entries = [
            item for item in manifest_images if item.get("split") == split
        ]
        if not split_entries:
            raise RuntimeError(INVALID_TRAINING_SPLIT_MESSAGE)
        for item in split_entries:
            image_path = version_root / str(item.get("export_image", ""))
            label_path = version_root / str(item.get("export_label", ""))
            if not image_path.is_file() or not label_path.is_file():
                raise RuntimeError(INVALID_TRAINING_SPLIT_MESSAGE)
```

In `execute_training_run`, perform the existing `data.yaml` and version-root checks, then call `validate_training_dataset(version, version_root)` inside a small `try/except RuntimeError` block that records `fail_training_run` and returns. Place this entire preflight before `from ultralytics import YOLO` and before `mark_training_run_running`.

- [ ] **Step 5: Run all training tests**

Run:

```bash
cd backend
.venv/bin/pytest tests/test_training_api.py -v
```

Expected: all training tests pass; fake successful trainers receive a data YAML with non-empty train and validation artifacts.

- [ ] **Step 6: Commit the training preflight**

```bash
git add backend/app/training/runner.py backend/tests/test_training_api.py
git commit -m "fix: validate dataset splits before training"
```

### Task 3: Full Verification and Service Restart

**Files:**
- Verify: `backend/app/versions/exporter.py`
- Verify: `backend/app/training/runner.py`
- Verify: `backend/tests/test_quality_version_api.py`
- Verify: `backend/tests/test_training_api.py`

- [ ] **Step 1: Run the complete backend suite**

Run:

```bash
cd backend
.venv/bin/pytest
```

Expected: all backend tests pass.

- [ ] **Step 2: Check formatting and patch cleanliness**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Restart the persistent backend service**

Run:

```bash
launchctl remove com.codex.yolo-trainer.backend || true
launchctl submit -l com.codex.yolo-trainer.backend -- /bin/zsh -lc \
  'cd ~/DevProjects/YOLO_Trainer/backend && exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >>/tmp/yolo-trainer-backend.log 2>&1'
```

Expected: `curl http://127.0.0.1:8000/api/health` returns `status: ok` and selects `mps` or `cpu` according to local availability.

- [ ] **Step 4: Verify the current invalid version fails quickly**

Create a new training task against version 4 through the API or UI. Expected: the run fails during preparation with the actionable minimum-image message and never logs `ultralytics training started`.

- [ ] **Step 5: Document the recovery workflow**

Report that the user must annotate one additional image, create a new dataset version, and submit a new training task. Do not mutate version 4 or retry the existing failed run.
