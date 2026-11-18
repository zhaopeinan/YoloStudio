from __future__ import annotations

from pathlib import Path

from app.core.settings import Settings


def remote_project_root(settings: Settings, project_id: int) -> Path:
    return settings.remote_root / "projects" / str(project_id)


def remote_version_root(settings: Settings, project_id: int, version_id: int) -> Path:
    return remote_project_root(settings, project_id) / "versions" / str(version_id)


def remote_weight_path(
    settings: Settings,
    project_id: int,
    fingerprint: str,
    filename: str = "weights.pt",
) -> Path:
    return remote_project_root(settings, project_id) / "weights" / fingerprint / filename


def remote_run_root(settings: Settings, project_id: int, run_id: int) -> Path:
    return remote_project_root(settings, project_id) / "runs" / str(run_id)


def remote_scripts_root(settings: Settings) -> Path:
    return settings.remote_root / "scripts"


def remote_manifest_path(settings: Settings, project_id: int, version_id: int) -> Path:
    return remote_version_root(settings, project_id, version_id) / "remote_manifest.json"
