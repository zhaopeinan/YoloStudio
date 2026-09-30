from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import Settings, get_settings
from app.db.models import Base


def create_engine_for_settings(settings: Settings) -> Engine:
    settings.ensure_workspace()
    return create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        future=True,
    )


engine = create_engine_for_settings(get_settings())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(db_engine: Engine = engine) -> None:
    Base.metadata.create_all(bind=db_engine)
    _ensure_preview_job_video_id(db_engine)
    _ensure_prediction_job_version_id(db_engine)


def _ensure_sqlite_column(
    db_engine: Engine,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    """SQLite create_all does not add columns to existing tables."""
    with db_engine.begin() as connection:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        if not rows:
            return
        column_names = {row[1] for row in rows}
        if column_name not in column_names:
            connection.exec_driver_sql(ddl)


def _ensure_preview_job_video_id(db_engine: Engine) -> None:
    _ensure_sqlite_column(
        db_engine,
        "preview_jobs",
        "video_id",
        "ALTER TABLE preview_jobs ADD COLUMN video_id INTEGER REFERENCES preview_videos(id)",
    )


def _ensure_prediction_job_version_id(db_engine: Engine) -> None:
    _ensure_sqlite_column(
        db_engine,
        "prediction_jobs",
        "version_id",
        "ALTER TABLE prediction_jobs ADD COLUMN version_id INTEGER REFERENCES dataset_versions(id)",
    )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
