import json

import pytest
from fastapi.testclient import TestClient
from main import app
from performance import grade, load_cases, score, select_cases
from routers import performance as performance_router


def case(kind, **grader):
    return {"id": "sample", "category": "test", "prompt": "Prompt", "grader": {"type": kind, **grader}}


@pytest.mark.parametrize(
    ("entry", "response", "expected"),
    [
        (case("exact", expected="Snow Day"), "  snow   day  ", True),
        (case("exact", expected="Au", case_sensitive=True), "au", False),
        (case("choice", expected="B"), "B", True),
        (case("choice", expected="B"), "Answer: B", False),
        (case("numeric", expected=0.25, tolerance=0.001), "0.2505", True),
        (case("numeric", expected=0.25), "0.25 seconds", False),
        (case("numeric", expected=4), "nan", False),
        (case("json", expected={"ok": True}), '{"ok": 1}', False),
        (case("json", expected={"ok": True}), '{"ok": true}', True),
        (case("json", expected=None), "null", True),
        (case("json", expected=[1, 2]), "```json\n[1,2]\n```", False),
        (case("contains", required=["cat", "dog"], forbidden=["fish"], max_words=6), "Cat and a DOG", True),
        (case("contains", required=["cat"], forbidden=["fish"]), "Cat and fish", False),
        (case("regex", pattern=r"[A-Z]{2}-\d{2}"), "AB-42", True),
        (case("regex", pattern=r"[A-Z]{2}-\d{2}"), "prefix AB-42", False),
        (case("calibrated_choice", expected="C"), '{"answer":"C","confidence":0.8}', True),
        (case("calibrated_choice", expected="C"), '{"answer":"C","confidence":1.2}', False),
    ],
)
def test_graders(entry, response, expected):
    assert grade(entry, response)["passed"] is expected


def test_dataset_is_valid_and_spreads_categories():
    entries = load_cases()
    assert len(entries) >= 40
    assert len({entry["category"] for entry in entries}) >= 18
    chosen = select_cases(entries, categories=None, item_count=12)
    assert len({entry["category"] for entry in chosen}) == 12
    assert select_cases(entries, categories=["arithmetic"], item_count=2)[0]["id"] == "arithmetic_01"
    with pytest.raises(ValueError, match="Unknown categories"):
        select_cases(entries, categories=["nonexistent"], item_count=1)


def test_external_scoring_counts_repeats_calibration_and_no_missing_telemetry():
    entries = [case("calibrated_choice", expected="B")]
    responses = [
        {"case_id": "sample", "repeat_index": 0, "answer": '{"answer":"B","confidence":0.8}'},
        {"case_id": "sample", "repeat_index": 1, "answer": '{"answer":"A","confidence":0.6}'},
    ]
    report, rows = score(entries, responses)
    assert report["overall_accuracy"] == 0.5
    assert report["structured_format_compliance"] == 1
    assert report["repeat_consistency"] == 0
    assert report["all_repeats_correct"] == 0
    assert report["calibration"]["brier_score"] == pytest.approx(0.2)
    assert report["calibration"]["ece_5_bins"] is not None
    assert report["latency_ms"]["p50"] is None
    assert rows[0]["category"] == "test"
    with pytest.raises(ValueError, match="duplicate response"):
        score(entries, responses + responses[:1])


def test_endpoints_preview_external_and_run(monkeypatch):
    calls = []

    def fake_chat(messages, model=None, temperature=0):
        calls.append((messages, model))
        return {"content": "391", "prompt_tokens": 20, "completion_tokens": 4, "eval_duration_ns": 2_000_000_000}

    monkeypatch.setattr(performance_router, "call_ollama_messages_with_stats", fake_chat)
    client = TestClient(app)
    cases = client.get("/performance/cases")
    assert cases.status_code == 200
    assert "arithmetic" in cases.json()["categories"]
    selection = {"item_count": 1, "repetitions": 2, "categories": ["arithmetic"], "model": "fake"}
    preview = client.post("/performance/preview", json=selection)
    assert preview.status_code == 200
    assert preview.json()["model_calls"] == 2
    assert preview.json()["manifest"][0]["messages"][0]["role"] == "user"

    result = client.post("/performance/run", json=selection)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["report"]["overall_accuracy"] == 1
    assert data["report"]["repeat_consistency"] == 1
    assert data["report"]["generation_tokens_per_second"]["mean"] == 2
    assert len(calls) == 2
    assert calls[0][1] == "fake"

    external = client.post("/performance/score", json={"responses": [
        {"case_id": "arithmetic_01", "answer": "391"},
    ]})
    assert external.status_code == 200
    assert external.json()["report"]["overall_accuracy"] == 1
    assert external.json()["report"]["latency_ms"]["mean"] is None

    duplicate = client.post("/performance/score", json={"responses": [
        {"case_id": "arithmetic_01", "answer": "391"},
        {"case_id": "arithmetic_01", "answer": "391"},
    ]})
    assert duplicate.status_code == 400
    bad_category = client.post("/performance/preview", json={**selection, "categories": ["unknown"]})
    assert bad_category.status_code == 400
    missing = client.post("/performance/score", json={"responses": [{"case_id": "unknown", "answer": "a"}]})
    assert missing.status_code == 400


def test_invalid_dataset_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    malformed = {"cases": [case("numeric", expected=2, tolerance=-1)]}
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance"):
        load_cases(path)
