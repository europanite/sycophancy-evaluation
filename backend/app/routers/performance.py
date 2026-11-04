"""Independent multi-capability diagnostic suite; SycoBench endpoints unchanged."""

from __future__ import annotations

import math
from time import perf_counter

from fastapi import APIRouter, HTTPException
from llm import call_ollama_messages_with_stats, get_default_model
from performance import load_cases, score, select_cases
from pydantic import BaseModel, Field

router = APIRouter(prefix="/performance", tags=["performance"])


class Selection(BaseModel):
    item_count: int = Field(12, ge=1, le=100)
    repetitions: int = Field(1, ge=1, le=5)
    categories: list[str] | None = None


class RunSelection(Selection):
    model: str | None = None


class Response(BaseModel):
    case_id: str
    repeat_index: int = Field(0, ge=0, le=4)
    answer: str = Field(max_length=20000)
    elapsed_ms: float | None = Field(None, ge=0)
    prompt_tokens: int | None = Field(None, ge=0)
    completion_tokens: int | None = Field(None, ge=0)
    generation_tokens_per_second: float | None = Field(None, ge=0)


class ExternalScore(BaseModel):
    responses: list[Response] = Field(min_length=1, max_length=500)


@router.get("/cases")
def cases():
    entries = load_cases()
    return {"count": len(entries), "categories": sorted({c["category"] for c in entries}), "cases": entries}


def _selection(request: Selection):
    try:
        return select_cases(load_cases(), categories=request.categories, item_count=request.item_count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/preview")
def preview(request: Selection):
    selected = _selection(request)
    return {"model_calls": len(selected) * request.repetitions, "manifest": [
        {"case_id": c["id"], "category": c["category"], "repeat_index": repeat,
         "messages": [{"role": "user", "content": c["prompt"]}], "grader_type": c["grader"]["type"]}
        for c in selected for repeat in range(request.repetitions)
    ]}


@router.post("/score")
def external_score(request: ExternalScore):
    try:
        report, rows = score(load_cases(), [r.model_dump() for r in request.responses])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"report": report, "runs": rows, "provenance": "external_responses"}


@router.post("/run")
def run(request: RunSelection):
    selected = _selection(request)
    model = request.model or get_default_model()
    responses = []
    for case in selected:
        for repeat in range(request.repetitions):
            started = perf_counter()
            try:
                raw = call_ollama_messages_with_stats([{"role": "user", "content": case["prompt"]}], model=model)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Model call failed for {case['id']} repeat {repeat}: {exc}") from exc
            elapsed_ms = (perf_counter() - started) * 1000
            eval_count = raw.get("completion_tokens")
            eval_duration = raw.get("eval_duration_ns")
            tps = None
            if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and math.isfinite(eval_duration) and eval_duration > 0:
                tps = eval_count * 1e9 / eval_duration
            responses.append({
                "case_id": case["id"], "repeat_index": repeat, "answer": raw["content"],
                "elapsed_ms": elapsed_ms, "prompt_tokens": raw.get("prompt_tokens"),
                "completion_tokens": eval_count, "generation_tokens_per_second": tps,
            })
    report, rows = score(selected, responses)
    return {"model": model, "run_count": len(rows), "report": report, "runs": rows}
