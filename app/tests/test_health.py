from fastapi.testclient import TestClient
from app.main import create_app


def test_health():
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"