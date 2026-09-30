from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.auth.deps import get_current_user
from app.auth.schemas import UserCreate
from app.auth.service import create_user, ensure_default_admin
from app.core.settings import Settings, get_settings
from app.db.models import Base
from app.db.session import create_engine_for_settings, get_db
from app.main import app


def _client_for(tmp_path: Path) -> tuple[TestClient, Settings, object]:
    settings = Settings(
        workspace_root=tmp_path / "workspace",
        jwt_secret="test-secret",
        default_admin_username="admin",
        default_admin_password="admin123",
    )
    engine = create_engine_for_settings(settings)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    with Session(engine) as db:
        ensure_default_admin(db, settings)
    return TestClient(app), settings, engine


def test_login_and_me(tmp_path: Path):
    client, settings, engine = _client_for(tmp_path)
    try:
        bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert bad.status_code == 401

        login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        assert login.status_code == 200
        token = login.json()["access_token"]
        assert login.json()["user"]["role"] == "admin"

        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["username"] == "admin"
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_settings, None)
        engine.dispose()


def test_annotator_forbidden_from_admin_routes(tmp_path: Path):
    client, settings, engine = _client_for(tmp_path)
    try:
        with Session(engine) as db:
            create_user(
                db,
                UserCreate(username="labeler", password="labeler1", role="annotator"),
            )
        login = client.post(
            "/api/auth/login",
            json={"username": "labeler", "password": "labeler1"},
        )
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        users = client.get("/api/users", headers=headers)
        assert users.status_code == 403

        scan = client.post(
            "/api/datasets/scan",
            headers=headers,
            json={"source_path": str(tmp_path / "missing.zip")},
        )
        assert scan.status_code == 403
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_settings, None)
        engine.dispose()


def test_admin_can_manage_users(tmp_path: Path):
    client, settings, engine = _client_for(tmp_path)
    try:
        login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        created = client.post(
            "/api/users",
            headers=headers,
            json={"username": "ann1", "password": "annpass1", "role": "annotator"},
        )
        assert created.status_code == 201
        user_id = created.json()["id"]

        listed = client.get("/api/users", headers=headers)
        assert listed.status_code == 200
        assert any(item["username"] == "ann1" for item in listed.json()["items"])

        updated = client.patch(
            f"/api/users/{user_id}",
            headers=headers,
            json={"is_active": False},
        )
        assert updated.status_code == 200
        assert updated.json()["is_active"] is False

        deleted = client.delete(f"/api/users/{user_id}", headers=headers)
        assert deleted.status_code == 204
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_settings, None)
        engine.dispose()


def test_unauthenticated_api_rejected(tmp_path: Path):
    client, settings, engine = _client_for(tmp_path)
    try:
        response = client.get("/api/projects")
        assert response.status_code == 401
        health = client.get("/api/health")
        assert health.status_code == 200
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_settings, None)
        engine.dispose()
