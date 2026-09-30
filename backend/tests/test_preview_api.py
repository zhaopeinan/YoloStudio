from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dataset_import_api import create_import_zip, isolated_client


class _TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _FakeModel:
    names = {0: "target"}

    def __call__(self, _source, **_kwargs):
        boxes = SimpleNamespace(
            xywhn=_TensorLike([[0.5, 0.5, 0.4, 0.3]]),
            cls=_TensorLike([0]),
            conf=_TensorLike([0.91]),
        )
        return [SimpleNamespace(boxes=boxes)]


def _import_dataset_with_annotation(client, zip_path: Path) -> tuple[int, int]:
    imported = client.post(
        "/api/datasets/import",
        json={
            "source_path": str(zip_path),
            "project_name": "Preview Test Project",
            "dataset_name": "preview-set",
        },
    ).json()
    class_response = client.post(
        f"/api/projects/{imported['project_id']}/classes",
        json={"name": "target", "color": "#22a06b"},
    )
    assert class_response.status_code == 200
    image = client.get(f"/api/datasets/{imported['dataset_id']}/images").json()["items"][0]
    annotation_response = client.put(
        f"/api/images/{image['id']}/annotations",
        json={
            "annotations": [
                {
                    "class_id": class_response.json()["id"],
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                }
            ]
        },
    )
    assert annotation_response.status_code == 200
    return imported["project_id"], image["id"]


def test_preview_models_include_builtin_model(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        response = client.get("/api/preview/models")

    assert response.status_code == 200
    assert response.json()["items"][0]["model_ref"] == "base:yolov8n.pt"


def test_image_preview_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    monkeypatch.setattr("app.preview.service.load_preview_model", lambda _path: _FakeModel())

    with isolated_client(tmp_path) as client:
        project_id, image_id = _import_dataset_with_annotation(client, zip_path)
        before = client.get(f"/api/images/{image_id}/annotations").json()
        response = client.post(
            f"/api/preview/images/{image_id}",
            json={"model_ref": "base:yolov8n.pt", "confidence_threshold": 0.05},
        )
        after = client.get(f"/api/images/{image_id}/annotations").json()

    assert project_id == 1
    assert response.status_code == 200
    assert response.json()["predictions"][0]["confidence"] == 0.91
    assert len(response.json()["annotations"]) == 1
    assert after == before


def test_video_preview_rejects_unsupported_extension(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    with isolated_client(tmp_path) as client:
        imported = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Preview Video Project",
                "dataset_name": "preview-set",
            },
        ).json()
        response = client.post(
            "/api/preview/video-jobs",
            data={"project_id": str(imported["project_id"]), "model_ref": "base:yolov8n.pt"},
            files={"video": ("sample.txt", b"not a video", "text/plain")},
        )

    assert response.status_code == 400
    assert "支持" in response.json()["detail"]


def test_preview_video_library_upload_list_retest_and_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"fake-weights")

    def _fake_process(preview_id, bind, settings):
        from sqlalchemy.orm import Session
        from app.db.models import PreviewJob

        with Session(bind) as db:
            job = db.get(PreviewJob, preview_id)
            assert job is not None
            job.status = "completed"
            job.result_path = str(tmp_path / f"result-{preview_id}.mp4")
            Path(job.result_path).write_bytes(b"fake-result")
            db.commit()

    monkeypatch.setattr(
        "app.preview.service.resolve_model_path",
        lambda db, settings, model_ref: (weights, None),
    )
    monkeypatch.setattr("app.preview.router.process_video_preview", _fake_process)

    with isolated_client(tmp_path) as client:
        imported = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Preview Library Project",
                "dataset_name": "preview-set",
            },
        ).json()
        project_id = imported["project_id"]

        upload = client.post(
            "/api/preview/videos",
            data={"project_id": str(project_id)},
            files={"video": ("clip.mp4", b"fake-mp4-bytes", "video/mp4")},
        )
        assert upload.status_code == 200, upload.text
        video = upload.json()
        assert video["original_filename"] == "clip.mp4"
        assert video["size_bytes"] == len(b"fake-mp4-bytes")

        listed = client.get(f"/api/preview/videos?project_id={project_id}")
        assert listed.status_code == 200
        assert len(listed.json()["items"]) == 1

        job_one = client.post(
            "/api/preview/video-jobs",
            data={
                "project_id": str(project_id),
                "model_ref": "base:yolov8n.pt",
                "video_id": str(video["id"]),
            },
        )
        assert job_one.status_code == 200, job_one.text
        assert job_one.json()["video_id"] == video["id"]
        completed_one = client.get(f"/api/preview/video-jobs/{job_one.json()['id']}")
        assert completed_one.status_code == 200
        assert completed_one.json()["status"] == "completed"

        delete_job = client.delete(f"/api/preview/video-jobs/{job_one.json()['id']}")
        assert delete_job.status_code == 204

        still_listed = client.get(f"/api/preview/videos?project_id={project_id}")
        assert len(still_listed.json()["items"]) == 1

        job_two = client.post(
            "/api/preview/video-jobs",
            data={
                "project_id": str(project_id),
                "model_ref": "base:yolov8n.pt",
                "video_id": str(video["id"]),
            },
        )
        assert job_two.status_code == 200, job_two.text
        assert job_two.json()["video_id"] == video["id"]
        completed_two = client.get(f"/api/preview/video-jobs/{job_two.json()['id']}")
        assert completed_two.json()["status"] == "completed"
        delete_video = client.delete(f"/api/preview/videos/{video['id']}")
        assert delete_video.status_code == 204
        empty = client.get(f"/api/preview/videos?project_id={project_id}")
        assert empty.json()["items"] == []


def test_preview_video_delete_blocked_while_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zip_path = tmp_path / "sample.zip"
    create_import_zip(zip_path)
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"fake-weights")

    def _noop_process(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "app.preview.service.resolve_model_path",
        lambda db, settings, model_ref: (weights, None),
    )
    monkeypatch.setattr("app.preview.router.process_video_preview", _noop_process)

    with isolated_client(tmp_path) as client:
        imported = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Preview Active Project",
                "dataset_name": "preview-set",
            },
        ).json()
        project_id = imported["project_id"]
        upload = client.post(
            "/api/preview/videos",
            data={"project_id": str(project_id)},
            files={"video": ("clip.mp4", b"fake-mp4-bytes", "video/mp4")},
        ).json()
        job = client.post(
            "/api/preview/video-jobs",
            data={
                "project_id": str(project_id),
                "model_ref": "base:yolov8n.pt",
                "video_id": str(upload["id"]),
            },
        )
        assert job.status_code == 200
        assert job.json()["status"] == "queued"

        blocked = client.delete(f"/api/preview/videos/{upload['id']}")
        assert blocked.status_code == 409
