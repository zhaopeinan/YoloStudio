from pathlib import Path

from sqlalchemy import inspect

from app.core.settings import Settings
from app.db.session import create_engine_for_settings, init_db


def test_settings_resolve_workspace_and_database_paths(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path / "workspace")

    assert settings.workspace_root == tmp_path / "workspace"
    assert settings.database_path == tmp_path / "workspace" / "app.db"
    assert settings.database_url.startswith("sqlite:///")


def test_init_db_creates_foundation_tables(tmp_path: Path):
    settings = Settings(workspace_root=tmp_path / "workspace")
    engine = create_engine_for_settings(settings)

    init_db(engine)

    tables = set(inspect(engine).get_table_names())
    assert {
        "projects",
        "datasets",
        "images",
        "export_artifacts",
        "trash_items",
        "users",
    }.issubset(tables)
    assert settings.workspace_root.exists()
