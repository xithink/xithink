from pathlib import Path

from fastapi.testclient import TestClient
from xithink.web.app import app


def test_web_health_and_index():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["version"] == "0.7.0"
    page = client.get("/")
    assert page.status_code == 200
    assert "XThink" in page.text
    assert "micButton" in page.text
    assert "规则演化" in page.text
    assert "0.7" in page.text
    assert "规则家谱/成长" in page.text
    assert "运行年龄模拟" in page.text


def test_static_assets_exist():
    base = Path(__file__).parents[1] / "xithink" / "web" / "static"
    assert (base / "app.js").exists()
    assert (base / "styles.css").exists()


def test_websocket_chat_flow(tmp_path, monkeypatch):
    import xithink.web.app as webapp
    monkeypatch.setattr(webapp, "DEFAULT_DB", tmp_path / "ws.db")
    client = TestClient(webapp.app)
    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_json({
            "text": "自由意味着选择。选择导致风险。什么是自由？",
            "cycles": 2,
            "environment": "测试环境",
        })
        seen = []
        for _ in range(8):
            msg = ws.receive_json()
            seen.append(msg["type"])
            if msg["type"] == "semantic":
                assert msg["data"]["environment"] == "测试环境"
            if msg["type"] == "cognition":
                assert msg["data"]["worldview"]["environment"] == "测试环境"
                assert "rule_evolution" in msg["data"]
                assert "rule_genealogy" in msg["data"]
                assert "active_hypothesis_testing" in msg["data"]
                assert "cognitive_profile" in msg["data"]
                assert "multi_perspective" in msg["data"]
            if msg["type"] == "reply":
                assert msg["text"]
                break
        assert "semantic" in seen
        assert "symbolic_answer" in seen
        assert "cognition" in seen
        assert "thought" in seen
        assert "reply" in seen


def test_simulation_api_runs_age_checkpoints():
    client = TestClient(app)
    response = client.post("/api/simulate", json={
        "profile": "balanced",
        "total_experiences": 40,
        "max_age": 40,
        "seed": 123,
        "skepticism": 0.7,
        "exploration": 0.8,
        "active_test_aggressiveness": 0.75,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_experiences"] == 40
    assert [x["age"] for x in data["checkpoints"]] == [0, 10, 20, 30, 40]
    assert data["profile"]["skepticism"] == 0.7
    assert data["profile"]["exploration"] == 0.8
    assert data["profile"]["active_test_aggressiveness"] == 0.75
