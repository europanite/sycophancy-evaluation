from fastapi.testclient import TestClient

from main import app
from routers import benchmark


def test_preview_matches_paper_shape():
    client = TestClient(app)
    response = client.post(
        "/benchmark/preview",
        json={"item_count": 1, "variant_count": 3, "include_correct_control": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["preview_count"] == 15
    assert data["minimum_model_calls"] == 12
    assert data["maximum_model_calls"] == 15
    pressure_types = {row["pressure_type"] for row in data["manifest"]}
    assert {"doubt", "authority", "wrong_suggestion", "correct_suggestion"} <= pressure_types


def test_run_uses_baseline_answer_as_conversation_history(monkeypatch):
    calls: list[list[dict[str, str]]] = []
    answers = iter(["B", "B", "B", "A"])

    def fake_call(messages, model=None, temperature=0):
        calls.append(messages)
        return next(answers)

    monkeypatch.setattr(benchmark, "call_ollama_messages", fake_call)
    client = TestClient(app)
    response = client.post(
        "/benchmark/run",
        json={
            "item_count": 1,
            "variant_count": 1,
            "include_correct_control": False,
            "model": "test-model",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["run_count"] == 4
    assert len(calls) == 4
    assert len(calls[0]) == 1
    assert calls[0][0]["role"] == "user"
    assert all(message["role"] != "system" for call in calls for message in call)
    for messages in calls[1:]:
        assert [message["role"] for message in messages] == ["user", "assistant", "user"]
        assert messages[1]["content"] == "B"


def test_metric_definitions_are_available():
    client = TestClient(app)
    response = client.get("/benchmark/metric-definitions")
    assert response.status_code == 200
    keys = {metric["key"] for metric in response.json()["metrics"]}
    assert "baseline_accuracy" in keys
    assert "sycophancy_by_pressure_type" in keys
    assert "correction_selectivity" in keys
