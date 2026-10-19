"""Additional, independent experimental benchmark for counterfactual evidence."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evaluation import parse_answer
from evidence import (
    CONDITIONS,
    baseline_prompt,
    followup_messages,
    load_evidence_cases,
    preview_cases,
    score_evidence_rows,
)
from llm import call_ollama_messages, get_default_model

router = APIRouter(prefix="/evidence", tags=["evidence-counterfactual"])


class EvidenceRequest(BaseModel):
    item_count: int = Field(default=4, ge=1, le=50)
    model: str | None = None


def _select(request: EvidenceRequest) -> list[dict]:
    cases = load_evidence_cases()
    if request.item_count > len(cases):
        raise HTTPException(400, f"Requested {request.item_count}; only {len(cases)} evidence cases available")
    return cases[: request.item_count]


@router.get("/items")
def items():
    cases = load_evidence_cases()
    return {"count": len(cases), "items": cases}


@router.post("/preview")
def preview(request: EvidenceRequest):
    cases = _select(request)
    manifest = preview_cases(cases)
    return {"total_model_calls": len(manifest), "manifest": manifest}


@router.post("/run")
def run(request: EvidenceRequest):
    cases = _select(request)
    model = request.model or get_default_model()
    rows = []
    for case in cases:
        prompt = baseline_prompt(case)
        first_messages = [{"role": "user", "content": prompt}]
        try:
            first_raw = call_ollama_messages(first_messages, model=model)
        except Exception as exc:
            raise HTTPException(502, f"Evidence model call failed: {case['item_id']}/baseline: {exc}") from exc
        rows.append({
            "item_id": case["item_id"],
            "condition": "baseline",
            "correct_answer": case["correct_answer"],
            "wrong_suggestion": case["wrong_suggestion"],
            "messages": first_messages,
            "raw_answer": first_raw,
            "answer": parse_answer(first_raw),
        })
        for condition in CONDITIONS:
            messages = followup_messages(case, condition, first_raw)
            try:
                raw = call_ollama_messages(messages, model=model)
            except Exception as exc:
                raise HTTPException(502, f"Evidence model call failed: {case['item_id']}/{condition}: {exc}") from exc
            rows.append({
                "item_id": case["item_id"],
                "condition": condition,
                "correct_answer": case["correct_answer"],
                "wrong_suggestion": case["wrong_suggestion"],
                "messages": messages,
                "raw_answer": raw,
                "answer": parse_answer(raw),
            })
    return {"model": model, "report": score_evidence_rows(rows), "runs": rows}
