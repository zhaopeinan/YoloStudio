from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_reports_local_app_state():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "YoloStudio"
    assert payload["workspace_root"]
    assert payload["database_path"].endswith("app.db")
    assert payload["devices"]["selected"] in {"cuda", "mps", "cpu"}
    assert "cpu" in payload["devices"]["available"]
