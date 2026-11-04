"""Auditable RAG scoring: exact IR ranks and separately attributed claim judgments.

This module does not retrieve documents or silently infer relevance labels.
"""

from __future__ import annotations

import math
from statistics import mean


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str] | None, ks: list[int]) -> dict:
    if gold_ids is None:
        return {"gold_count": None, "by_k": {str(k): None for k in ks}, "context_precision_ap": None}
    gold = set(gold_ids)
    # A query with no relevant reference documents has undefined recall.
    if not gold:
        return {"gold_count": 0, "by_k": {str(k): None for k in ks}, "context_precision_ap": None}
    relevant_ranks = [index for index, doc_id in enumerate(retrieved_ids, start=1) if doc_id in gold]

    def average_precision_at(k: int) -> float:
        hits = 0
        total = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in gold:
                hits += 1
                total += hits / rank
        return total / min(k, len(gold))

    def ndcg_at(k: int) -> float:
        dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks if rank <= k)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
        return dcg / ideal if ideal else 0.0

    by_k = {}
    for k in ks:
        ranks = [rank for rank in relevant_ranks if rank <= k]
        by_k[str(k)] = {
            "recall": len(ranks) / len(gold),
            # Missing slots below K are counted as non-relevant.
            "precision": len(ranks) / k,
            "hit_rate": float(bool(ranks)),
            "mrr": 1.0 / ranks[0] if ranks else 0.0,
            "ndcg": ndcg_at(k),
            "average_precision": average_precision_at(k),
        }
    return {
        "gold_count": len(gold),
        "by_k": by_k,
        # AP over all returned ranks: explicitly a label-based proxy for context precision.
        "context_precision_ap": average_precision_at(len(retrieved_ids)) if retrieved_ids else 0.0,
    }


def judged_metrics(sample: dict) -> dict:
    judgment = sample.get("judgment")
    empty = {
        "faithfulness": None, "contextual_recall": None,
        "answer_relevancy": None, "answer_correctness": None,
        "noise_sensitivity_relevant": None, "noise_sensitivity_irrelevant": None,
        "paired_noise_flip": None,
    }
    if judgment is None:
        return empty
    ids = {ctx["id"] for ctx in sample["retrieved_contexts"]}
    gold = set(sample["relevant_context_ids"]) if sample.get("relevant_context_ids") is not None else None
    claims = judgment["response_claims"]
    reference_claims = judgment["reference_claims"]
    for claim in claims + reference_claims:
        if not set(claim["supported_context_ids"]).issubset(ids):
            raise ValueError(f"{sample['item_id']}: claim references a context ID not retrieved")

    result = {
        "faithfulness": sum(bool(c["supported_context_ids"]) for c in claims) / len(claims) if claims else None,
        "contextual_recall": sum(bool(c["supported_context_ids"]) for c in reference_claims) / len(reference_claims) if reference_claims else None,
        "answer_relevancy": judgment.get("answer_relevancy"),
        "answer_correctness": judgment.get("answer_correctness"),
        "noise_sensitivity_relevant": None,
        "noise_sensitivity_irrelevant": None,
        "paired_noise_flip": None,
    }
    if gold is not None and claims:
        # Attribution proxy: false response claims supported by retrieved
        # relevant / irrelevant contexts, divided by all response claims.
        for kind, source_ids in (("relevant", gold), ("irrelevant", ids - gold)):
            result[f"noise_sensitivity_{kind}"] = sum(
                c["correct_against_reference"] is False
                and bool(set(c["supported_context_ids"]) & source_ids)
                for c in claims
            ) / len(claims)
    if sample.get("clean_response") is not None:
        clean_correct = judgment.get("clean_answer_correct")
        noisy_correct = judgment.get("noisy_answer_correct")
        if clean_correct is True and noisy_correct is not None:
            result["paired_noise_flip"] = float(not noisy_correct)
    return result


def score_rag(items: list[dict], ks: list[int]) -> dict:
    rows = []
    for item in items:
        ir = retrieval_metrics(
            [ctx["id"] for ctx in item["retrieved_contexts"]], item.get("relevant_context_ids"), ks
        )
        judged = judged_metrics(item)
        rows.append({
            "item_id": item["item_id"], "question": item["question"],
            "retrieved_count": len(item["retrieved_contexts"]),
            "retrieval": ir, "generation": judged,
            "judgment_source": item.get("judgment_source", "not_judged"),
            "judgment": item.get("judgment"),
            "judge_raw_response": item.get("judge_raw_response"),
        })

    aggregate_ir = {}
    for k in ks:
        key = str(k)
        aggregate_ir[key] = {}
        for metric in ("recall", "precision", "hit_rate", "mrr", "ndcg", "average_precision"):
            values = [r["retrieval"]["by_k"][key][metric] for r in rows if r["retrieval"]["by_k"][key] is not None]
            aggregate_ir[key][metric] = {"mean": _mean(values), "samples": len(values)}

    aggregate_generation = {}
    for key in rows[0]["generation"]:
        values = [r["generation"][key] for r in rows if r["generation"][key] is not None]
        aggregate_generation[key] = {"mean": _mean(values), "samples": len(values)}
    cp_values = [r["retrieval"]["context_precision_ap"] for r in rows if r["retrieval"]["context_precision_ap"] is not None]

    return {
        "protocol": "rag_retrieval_exact_and_annotated_generation_v1",
        "item_count": len(rows), "ks": ks,
        "retrieval_at_k": aggregate_ir,
        "context_precision_ap": {"mean": _mean(cp_values), "samples": len(cp_values)},
        "generation": aggregate_generation,
        "rows": rows,
    }
