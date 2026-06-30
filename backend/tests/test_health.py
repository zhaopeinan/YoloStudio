from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_local_app_state():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "YOLO Trainer"
    assert "workspace_root" in payload
    assert "database_path" in payload
    assert payload["devices"]["selected"] in {"cuda", "mps", "cpu"}
