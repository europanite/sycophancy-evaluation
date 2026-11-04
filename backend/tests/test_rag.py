import json
import math

import pytest
from fastapi.testclient import TestClient
from main import app
from rag_evaluation import retrieval_metrics, score_rag
from routers import rag


def test_exact_retrieval_metrics_and_rank_weighted_precision():
    report = retrieval_metrics(["noise", "d2", "d1"], ["d1", "d2", "d3"], [1, 2, 3, 5])
    assert report["by_k"]["1"]["recall"] == 0
    assert report["by_k"]["2"]["recall"] == pytest.approx(1 / 3)
    assert report["by_k"]["2"]["precision"] == 0.5
    assert report["by_k"]["3"]["hit_rate"] == 1
    assert report["by_k"]["3"]["mrr"] == 0.5
    assert report["by_k"]["5"]["recall"] == pytest.approx(2 / 3)
    assert report["by_k"]["5"]["precision"] == pytest.approx(2 / 5)
    assert report["by_k"]["3"]["average_precision"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    actual = 1 / math.log2(3) + 1 / math.log2(4)
    assert report["by_k"]["3"]["ndcg"] == pytest.approx(actual / ideal)
    assert report["context_precision_ap"] == pytest.approx((1 / 2 + 2 / 3) / 3)


def test_missing_gold_is_undefined_not_zero():
    assert retrieval_metrics(["a"], None, [1])["by_k"]["1"] is None
    assert retrieval_metrics(["a"], [], [1])["by_k"]["1"] is None


def test_annotated_examples_distinguish_faithfulness_correctness_and_noise():
    sample = rag.examples()
    report = score_rag([{**item, "judgment_source": "external_annotation"} for item in sample["items"]], sample["ks"])
    assert report["item_count"] == 2
    assert report["retrieval_at_k"]["1"]["recall"]["mean"] == 0
    assert report["retrieval_at_k"]["3"]["recall"]["mean"] == 1
    good, misleading = report["rows"]
    assert good["generation"]["faithfulness"] == 1
    assert good["generation"]["contextual_recall"] == 1
    assert misleading["generation"]["faithfulness"] == 1
    assert misleading["generation"]["answer_correctness"] == 0
    assert misleading["generation"]["noise_sensitivity_irrelevant"] == 1
    assert misleading["generation"]["noise_sensitivity_relevant"] == 0
    assert misleading["generation"]["paired_noise_flip"] == 1
    assert report["generation"]["answer_correctness"]["samples"] == 2


def test_rag_score_endpoints_accept_external_and_reject_invalid_ids():
    client = TestClient(app)
    sample = client.get("/rag/examples").json()
    response = client.post("/rag/score", json=sample)
    assert response.status_code == 200, response.text
    assert response.json()["generation"]["contextual_recall"]["mean"] == 1
    assert response.json()["rows"][0]["judgment_source"] == "external_annotation"

    sample["items"][0]["judgment"]["response_claims"][0]["supported_context_ids"] = ["unknown"]
    response = client.post("/rag/score", json=sample)
    assert response.status_code == 400
    assert "not retrieved" in response.json()["detail"]

    sample["items"][0]["retrieved_contexts"].append(sample["items"][0]["retrieved_contexts"][0])
    assert client.post("/rag/score", json=sample).status_code == 422


def test_rag_judge_api_provenance_and_strict_validation(monkeypatch):
    requests = []
    sample = rag.examples()
    sample["items"] = sample["items"][:1]

    def fake_judge(messages, model=None, **kwargs):
        requests.append((messages, model, kwargs))
        return {"content": json.dumps(sample["items"][0]["judgment"])}

    monkeypatch.setattr(rag, "call_ollama_messages_with_stats", fake_judge)
    client = TestClient(app)
    response = client.post("/rag/judge", json={**sample, "judge_model": "independent-judge"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["judge_model"] == "independent-judge"
    assert body["rows"][0]["judgment_source"] == "ollama_judge:independent-judge"
    assert body["rows"][0]["judge_raw_response"]
    assert requests[0][0][0]["role"] == "system"
    assert requests[0][1] == "independent-judge"
    assert requests[0][2]["max_tokens"] == 1024
    assert requests[0][2]["response_format"] == "json"

    monkeypatch.setattr(rag, "call_ollama_messages_with_stats", lambda *a, **kw: {"content": "not JSON"})
    response = client.post("/rag/judge", json=sample)
    assert response.status_code == 502

    bad_judgment = sample["items"][0]["judgment"].copy()
    bad_judgment["response_claims"] = [
        {"claim": "invented", "supported_context_ids": ["not-a-real-id"]}
    ]
    monkeypatch.setattr(
        rag, "call_ollama_messages_with_stats",
        lambda *a, **kw: {"content": json.dumps(bad_judgment)},
    )
    assert client.post("/rag/judge", json=sample).status_code == 502


def test_no_judgment_means_no_generated_scores():
    case = {
        "item_id": "a", "question": "Q", "retrieved_contexts": [],
        "response": "R", "reference_answer": "Gold", "relevant_context_ids": ["missing"],
    }
    report = score_rag([case], [1, 5])
    assert report["retrieval_at_k"]["1"]["recall"]["mean"] == 0
    assert report["generation"]["faithfulness"] == {"mean": None, "samples": 0}
    assert report["generation"]["contextual_recall"] == {"mean": None, "samples": 0}
