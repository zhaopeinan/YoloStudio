# 模型效果预览 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a standalone “效果预览” workflow where users inspect image predictions from the built-in or trained model and stream annotated video inference while saving downloadable results.

**Architecture:** Add a focused backend preview package with a database-backed video job, a shared model resolver, a read-only image prediction endpoint, and a filesystem-backed video frame stream. Add a focused PreviewView frontend component and connect it to the existing workflow without mixing preview state into training or formal evaluation state.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, Ultralytics YOLO, OpenCV, ffmpeg, React, TypeScript, Vitest, Testing Library.

---

## File Map

### Backend

- Create: backend/app/preview/__init__.py
- Create: backend/app/preview/schemas.py
- Create: backend/app/preview/service.py
- Create: backend/app/preview/router.py
- Modify: backend/app/db/models.py
- Modify: backend/app/main.py
- Create: backend/tests/test_preview_api.py

### Frontend

- Modify: frontend/src/workflow.ts
- Create: frontend/src/PreviewView.tsx
- Modify: frontend/src/api.ts
- Modify: frontend/src/App.tsx
- Modify: frontend/src/styles.css
- Create: frontend/src/PreviewView.test.tsx
- Modify: frontend/src/App.test.tsx
- Modify: frontend/src/workflow.test.ts

### Documentation

- Modify: README.md

## Task 1: Add the Preview Job Domain Model and Schemas

**Files:**
- Modify: backend/app/db/models.py
- Create: backend/app/preview/__init__.py
- Create: backend/app/preview/schemas.py
- Test: backend/tests/test_preview_api.py

- [ ] Step 1: Write failing model-list tests.

Add a fresh SQLite test that requests GET /api/preview/models and expects base:yolov8n.pt. Add a completed training fixture and assert run:{id} appears; add a failed training fixture and assert it does not appear.

~~~python
def test_preview_model_options(client_with_completed_run):
    response = client_with_completed_run.get("/api/preview/models")
    assert response.status_code == 200
    assert response.json()["items"][0]["model_ref"] == "base:yolov8n.pt"
    assert response.json()["items"][1]["model_ref"].startswith("run:")
~~~

- [ ] Step 2: Run the test and verify failure.

~~~bash
cd backend
pytest -q tests/test_preview_api.py::test_preview_model_options
~~~

Expected: FAIL because the preview package, router, schema, and table do not exist.

- [ ] Step 3: Add PreviewJob and response schemas.

Add PreviewJob(TimestampMixin, Base) with fields:

~~~python
id, project_id, dataset_id, run_id, model_ref, kind,
source_filename, source_path, result_path, frame_directory,
status, confidence_threshold, frame_step, fps,
total_frames, processed_frames, error_message,
started_at, ended_at
~~~

Use nullable foreign keys for dataset_id and run_id, counters defaulting to zero, and service-validated statuses queued, preparing, running, completed, failed, and deleted.

Define PreviewModelOption, PreviewBox, ImagePreviewRequest, ImagePreviewResponse, VideoPreviewCreate, PreviewJobRead, and PreviewJobList in backend/app/preview/schemas.py. Prediction coordinates are normalized YOLO x_center, y_center, width, and height.

- [ ] Step 4: Run the focused test and verify it passes.

~~~bash
cd backend
pytest -q tests/test_preview_api.py::test_preview_model_options
~~~

Expected: PASS.

- [ ] Step 5: Commit.

~~~bash
git add backend/app/db/models.py backend/app/preview backend/tests/test_preview_api.py
git commit -m "feat: add preview job domain"
~~~

## Task 2: Implement Model Resolution and Read-Only Image Preview

**Files:**
- Create: backend/app/preview/service.py
- Create: backend/app/preview/router.py
- Modify: backend/app/main.py
- Modify: backend/tests/test_preview_api.py

- [ ] Step 1: Write failing image preview tests.

Use the existing imported-dataset fixture and monkeypatch the Ultralytics loader to return one deterministic box. Cover:

~~~python
def test_preview_image_is_read_only(client_with_dataset):
    image = client_with_dataset.first_dataset_image()
    response = client_with_dataset.client.post(
        f"/api/preview/images/{image['id']}",
        json={"model_ref": "base:yolov8n.pt", "confidence_threshold": 0.05},
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert client_with_dataset.annotations(image["id"]) == []
~~~

Also test completed run:{id}, failed or missing runs, unknown model refs, missing images, invalid thresholds, and inclusion of real annotations without mutation.

- [ ] Step 2: Run tests to verify failure.

~~~bash
cd backend
pytest -q tests/test_preview_api.py -k "preview_image or model_ref"
~~~

Expected: FAIL because model resolution and image prediction are absent.

- [ ] Step 3: Implement the controlled model resolver.

Add these service functions:

~~~python
def resolve_model_path(db: Session, settings: Settings, model_ref: str) -> tuple[Path, int | None]:
    ...

def load_preview_model(model_path: Path):
    ...

def predict_image(model_path: Path, image_path: Path, confidence_threshold: float) -> list[PreviewBox]:
    ...
~~~

Resolve base:yolov8n.pt only to backend/yolov8n.pt. Resolve run:{id} through an active completed TrainingRun and require ultralytics/weights/best.pt. Reject arbitrary paths and path traversal. Cache loaded models by resolved path for the local process.

Use Ultralytics normalized xywhn output and the project class library for class names/colors. Keep this service read-only.

- [ ] Step 4: Implement the preview endpoints.

~~~text
GET  /api/preview/models
POST /api/preview/images/{image_id}
~~~

Validate the model, image path, project class mapping, and threshold before inference. Return image dimensions, image URL, real annotations, predictions, model ref, and actual threshold.

- [ ] Step 5: Register and verify.

~~~bash
cd backend
pytest -q tests/test_preview_api.py -k "preview_image or model_ref"
ruff check app/preview app/main.py tests/test_preview_api.py
~~~

Expected: focused tests pass and Ruff reports no errors.

- [ ] Step 6: Commit.

~~~bash
git add backend/app/preview backend/app/main.py backend/tests/test_preview_api.py
git commit -m "feat: add read-only image preview"
~~~

## Task 3: Implement Video Upload, Live Frame Stream, Result Saving, and Deletion

**Files:**
- Modify: backend/app/preview/schemas.py
- Modify: backend/app/preview/service.py
- Modify: backend/app/preview/router.py
- Modify: backend/tests/test_preview_api.py

- [ ] Step 1: Write failing video lifecycle tests.

Generate a three-frame MP4 fixture and monkeypatch inference to draw a deterministic box. Cover creation, completion, stream, result, deletion, invalid extension, missing model, forced inference failure, and cleanup.

~~~python
def test_video_preview_lifecycle(client_with_completed_run, video_bytes):
    response = client_with_completed_run.client.post(
        "/api/preview/video-jobs",
        data={
            "project_id": "1",
            "model_ref": "base:yolov8n.pt",
            "confidence_threshold": "0.05",
            "frame_step": "1",
        },
        files={"video": ("sample.mp4", video_bytes, "video/mp4")},
    )
    assert response.status_code == 200
    preview_id = response.json()["id"]
    status = wait_for_preview_status(client_with_completed_run.client, preview_id, "completed")
    assert status["processed_frames"] == 3
    assert client_with_completed_run.client.get(
        f"/api/preview/video-jobs/{preview_id}/result"
    ).status_code == 200
    assert client_with_completed_run.client.delete(
        f"/api/preview/video-jobs/{preview_id}"
    ).status_code == 204
~~~

- [ ] Step 2: Run the video tests and verify failure.

~~~bash
cd backend
pytest -q tests/test_preview_api.py -k "video"
~~~

Expected: FAIL because video jobs and processing do not exist.

- [ ] Step 3: Implement upload validation and job creation.

Accept mp4, mov, avi, mkv, and webm; enforce a documented maximum size and one active video job. Save:

~~~text
workspace/projects/{project_id}/previews/{preview_id}/source.<extension>
workspace/projects/{project_id}/previews/{preview_id}/frames/
~~~

Persist the job before scheduling background processing and return queued immediately.

- [ ] Step 4: Implement the OpenCV and ffmpeg worker.

Implement process_video_preview(preview_id, bind, settings):

1. Resolve and load the model.
2. Open the source with OpenCV and read fps, dimensions, and frame count.
3. Set status to preparing, then running.
4. Apply frame_step, infer, draw boxes and labels, and write sequential JPEGs.
5. Write a silent temporary MP4 using cv2.VideoWriter.
6. Use ffmpeg to mux original audio when available and publish result.mp4 atomically.
7. Update processed and total counters after frames.
8. Mark completed, or mark failed and remove incomplete result files on error.

Use the same normalized prediction conversion as image preview. Do not write to Prediction or annotations.

- [ ] Step 5: Implement status, stream, result, and delete.

~~~text
GET    /api/preview/video-jobs/{id}
GET    /api/preview/video-jobs/{id}/stream
GET    /api/preview/video-jobs/{id}/result
DELETE /api/preview/video-jobs/{id}
~~~

The stream generator reads frame files in order, waits briefly for the next frame while running, stops at completion, and exits on client disconnect. The result endpoint serves only completed output. Delete removes the preview directory and database record without touching source data.

- [ ] Step 6: Run focused backend verification.

~~~bash
cd backend
pytest -q tests/test_preview_api.py
ruff check app/preview app/main.py tests/test_preview_api.py
~~~

Expected: all preview tests pass and Ruff is clean.

- [ ] Step 7: Commit.

~~~bash
git add backend/app/preview backend/tests/test_preview_api.py
git commit -m "feat: add streaming video preview jobs"
~~~

## Task 4: Add Workflow Navigation and API Types

**Files:**
- Modify: frontend/src/workflow.ts
- Modify: frontend/src/api.ts
- Modify: frontend/src/App.tsx
- Modify: frontend/src/workflow.test.ts
- Modify: frontend/src/App.test.tsx

- [ ] Step 1: Write failing workflow tests.

~~~typescript
expect(workflowStepOrder).toEqual([
  "dataset",
  "classes",
  "annotation",
  "quality",
  "training",
  "preview",
  "evaluation",
]);
~~~

Assert preview is locked without a training run and available after a run exists, while evaluation keeps its existing availability rule.

- [ ] Step 2: Run tests to verify failure.

~~~bash
cd frontend
npm test -- --run src/workflow.test.ts
~~~

Expected: FAIL because preview is not a workflow step.

- [ ] Step 3: Add TypeScript preview contracts and API helpers.

Add types for PreviewModelOption, PreviewBox, ImagePreviewResponse, PreviewJob, and VideoPreviewStatus. Add:

~~~typescript
listPreviewModels(): Promise<PreviewModelOption[]>;
previewImage(imageId: number, body: ImagePreviewRequest): Promise<ImagePreviewResponse>;
createVideoPreview(body: FormData): Promise<PreviewJob>;
getVideoPreviewStatus(previewId: number): Promise<PreviewJob>;
getVideoPreviewStreamUrl(previewId: number): string;
getVideoPreviewResultUrl(previewId: number): string;
deleteVideoPreview(previewId: number): Promise<void>;
~~~

Use FormData for video upload and do not set the multipart content type manually.

- [ ] Step 4: Add the step and App wiring.

Add preview to the workflow union, order, labels, descriptions, hash parsing, and progress calculation. Render PreviewView with active project ID, datasets, completed runs, and class library.

- [ ] Step 5: Run focused frontend verification.

~~~bash
cd frontend
npm test -- --run src/workflow.test.ts src/App.test.tsx
npm run build
~~~

Expected: new workflow tests pass and the production build succeeds.

- [ ] Step 6: Commit.

~~~bash
git add frontend/src/workflow.ts frontend/src/api.ts frontend/src/App.tsx frontend/src/workflow.test.ts frontend/src/App.test.tsx
git commit -m "feat: add effect preview workflow step"
~~~

## Task 5: Build the Image Preview View

**Files:**
- Create: frontend/src/PreviewView.tsx
- Create: frontend/src/PreviewView.test.tsx
- Modify: frontend/src/App.tsx
- Modify: frontend/src/styles.css

- [ ] Step 1: Write failing component tests.

Cover model/dataset selection, initial image load, threshold reload, previous/next, prediction/ground-truth toggles, empty predictions, request errors, and navigation boundaries.

~~~typescript
it("previews the first image after model and dataset selection", async () => {
  render(<PreviewView {...fixtureProps} />);
  await user.selectOptions(screen.getByLabelText("预览模型"), "run:1");
  await user.selectOptions(screen.getByLabelText("预览数据集"), "4:3");
  expect(await screen.findByRole("img", { name: "frame001.jpg" })).toBeInTheDocument();
  expect(apiMock.previewImage).toHaveBeenCalledWith(101, {
    model_ref: "run:1",
    confidence_threshold: 0.05,
  });
});
~~~

- [ ] Step 2: Run tests to verify failure.

~~~bash
cd frontend
npm test -- --run src/PreviewView.test.tsx
~~~

Expected: FAIL because PreviewView does not exist.

- [ ] Step 3: Implement the image preview state.

Use:

~~~typescript
type PreviewViewProps = {
  projectId: number | null;
  datasets: ProjectDatasetSummary[];
  runs: TrainingRun[];
  classes: ProjectClass[];
};
~~~

Load the first dataset image after selection, call previewImage, and cache results by image ID/model/threshold. Clear the cache when model or dataset changes.

- [ ] Step 4: Implement the read-only canvas.

Render image, prediction boxes, and real annotation boxes in one normalized coordinate system. Reuse existing zoom helpers where practical. Add accessible labels for model, dataset, threshold, previous, next, prediction visibility, and ground-truth visibility.

- [ ] Step 5: Add responsive styles and verify.

Use a two-column desktop layout and stacked mobile layout. Run:

~~~bash
cd frontend
npm test -- --run src/PreviewView.test.tsx
npm run build
~~~

Expected: all image preview tests pass and the build succeeds.

- [ ] Step 6: Commit.

~~~bash
git add frontend/src/PreviewView.tsx frontend/src/PreviewView.test.tsx frontend/src/App.tsx frontend/src/styles.css
git commit -m "feat: add image effect preview"
~~~

## Task 6: Build the Live Video Preview View

**Files:**
- Modify: frontend/src/PreviewView.tsx
- Modify: frontend/src/PreviewView.test.tsx
- Modify: frontend/src/styles.css

- [ ] Step 1: Add failing video interaction tests.

Cover upload, live progress, stream URL, completion, result playback, download, deletion, invalid file type, upload failure, and processing failure.

~~~typescript
it("uploads a video and exposes result actions", async () => {
  render(<PreviewView {...fixtureProps} />);
  await user.upload(
    screen.getByLabelText("上传视频"),
    new File(["video"], "sample.mp4", { type: "video/mp4" }),
  );
  expect(apiMock.createVideoPreview).toHaveBeenCalled();
  expect(await screen.findByText("推理中")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "下载结果视频" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "删除本次结果" }));
  expect(apiMock.deleteVideoPreview).toHaveBeenCalledWith(7);
});
~~~

- [ ] Step 2: Run tests to verify failure.

~~~bash
cd frontend
npm test -- --run src/PreviewView.test.tsx
~~~

Expected: FAIL because video state and controls are not implemented.

- [ ] Step 3: Implement upload and polling.

Use FormData. After creation, set the stream URL and poll status every 500 ms until completed or failed. Stop polling on unmount, deletion, and terminal state.

- [ ] Step 4: Implement stream and result actions.

Use an image element for the multipart stream while running. When complete, show a video element for result.mp4 and provide:

- 下载结果视频: result URL with download attribute.
- 删除本次结果: confirm once, call delete API, and clear stream/result state.

- [ ] Step 5: Run frontend verification.

~~~bash
cd frontend
npm test -- --run src/PreviewView.test.tsx src/App.test.tsx
npm run build
~~~

Expected: preview and App tests pass and the production build succeeds.

- [ ] Step 6: Commit.

~~~bash
git add frontend/src/PreviewView.tsx frontend/src/PreviewView.test.tsx frontend/src/styles.css
git commit -m "feat: add live video effect preview"
~~~

## Task 7: Documentation and Full Regression Verification

**Files:**
- Modify: README.md
- Modify: backend/tests/test_preview_api.py
- Modify: frontend/src/App.test.tsx

- [ ] Step 1: Document the local workflow.

Document creating a frozen version, completing a training run, opening “效果预览”, selecting a base/trained model, previewing images, uploading a supported video, downloading/deleting results, result paths, and the fact that preview does not change annotations.

- [ ] Step 2: Run the complete automated suite.

~~~bash
cd backend
pytest -q
ruff check app tests

cd ../frontend
npm test -- --run
npm run build

cd ..
git diff --check
~~~

Expected: all tests pass, Ruff is clean, the production build succeeds, and git diff --check reports no whitespace errors.

- [ ] Step 3: Run the local manual smoke test.

With both services running:

~~~bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/preview/models
~~~

In the browser, open #preview, choose the current project, a dataset, and yolov8n.pt, preview two images, change threshold, choose a completed run best.pt, upload a short MP4, verify live frames, completion, result playback, download, deletion, and original-data preservation.

- [ ] Step 4: Commit documentation and final integration changes.

~~~bash
git add README.md backend/tests/test_preview_api.py frontend/src/App.test.tsx
git commit -m "docs: document effect preview workflow"
~~~

## Self-Review Checklist

- [ ] The plan covers approved image preview and live video preview scope.
- [ ] Base and trained model references use controlled identifiers, not arbitrary paths.
- [ ] Preview predictions remain separate from annotations and formal prediction jobs.
- [ ] Video results are saved, downloadable, and deletable without touching source files.
- [ ] Stream, status polling, result download, and deletion use consistent preview IDs.
- [ ] Tests cover backend lifecycle, frontend interactions, failure states, and existing regression suites.
