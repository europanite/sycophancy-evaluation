from __future__ import annotations

from fastapi import APIRouter, HTTPException

from evaluation import (
    PRESSURE_TYPES,
    build_baseline_prompt,
    build_preview_manifest,
    classify_rows,
    load_prompt_config,
    parse_answer,
    pressure_text,
    save_prompt_config,
    score_classified,
    variant_count,
)
from llm import call_ollama_messages, get_default_model
from metrics_catalog import METRIC_DEFINITIONS
from schemas import PreviewRequest, RunRequest, ScoreRequest, TestPromptConfig

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


def _select_items(config: dict, item_count: int) -> list[dict]:
    items = config["items"]
    if item_count > len(items):
        raise HTTPException(
            status_code=400,
            detail=f"Requested {item_count} items but only {len(items)} test prompts exist.",
        )
    return items[:item_count]


def _variant_ids(config: dict, requested: int) -> list[int]:
    available = variant_count(config)
    if requested > available:
        raise HTTPException(status_code=400, detail=f"Requested {requested} variants; only {available} exist.")
    return list(range(requested))


@router.get("/metric-definitions")
def metric_definitions():
    return {"metrics": METRIC_DEFINITIONS}


@router.get("/items")
def items():
    config = load_prompt_config()
    rows = config["items"]
    return {
        "count": len(rows),
        "items": [
            {"item_id": row["item_id"], "domain": row.get("domain", "unknown"), "question": row["question"]}
            for row in rows
        ],
    }


@router.get("/test-prompts")
def get_test_prompts():
    return load_prompt_config()


@router.put("/test-prompts")
def put_test_prompts(request: TestPromptConfig):
    try:
        saved = save_prompt_config(request.model_dump())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True, "count": len(saved["items"]), "config": saved}


@router.post("/preview")
def preview(request: PreviewRequest):
    config = load_prompt_config()
    selected = _select_items(config, request.item_count)
    variants = _variant_ids(config, request.variant_count)
    manifest = build_preview_manifest(
        selected,
        include_correct_control=request.include_correct_control,
        variant_ids=variants,
        config=config,
    )
    unconditional = sum(not row.get("conditional") for row in manifest)
    conditional = sum(bool(row.get("conditional")) for row in manifest)
    return {
        "protocol": config["protocol"],
        "preview_count": len(manifest),
        "minimum_model_calls": unconditional,
        "maximum_model_calls": unconditional + conditional,
        "manifest": manifest,
    }


@router.post("/score")
def score(request: ScoreRequest):
    # External scoring remains available for already-materialized paper-style runs.
    by_id = {r.run_id: r.answer for r in request.responses}
    rows = []
    for row in request.manifest:
        if row.get("conditional") and row["run_id"] not in by_id:
            continue
        if row["run_id"] not in by_id:
            raise HTTPException(status_code=400, detail=f"Missing response for {row['run_id']}")
        rows.append({**row, "raw_answer": by_id[row["run_id"]], "answer": parse_answer(by_id[row["run_id"]])})
    try:
        classified = classify_rows(rows)
        report = score_classified(classified)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"report": report, "runs": classified}


@router.post("/run")
def run(request: RunRequest):
    config = load_prompt_config()
    selected = _select_items(config, request.item_count)
    variants = _variant_ids(config, request.variant_count)
    model = request.model or get_default_model()

    rows: list[dict] = []
    for item in selected:
        baseline_prompt = build_baseline_prompt(item, config)
        for variant_id in variants:
            common = {
                "item_id": item["item_id"],
                "domain": item.get("domain", "unknown"),
                "variant_id": variant_id,
                "correct_answer": item["correct_answer"].upper(),
                "baseline_prompt": baseline_prompt,
            }
            baseline_messages = [{"role": "user", "content": baseline_prompt}]
            baseline_id = f"{item['item_id']}__v{variant_id}__baseline"
            try:
                baseline_raw = call_ollama_messages(baseline_messages, model=model)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Model call failed for {baseline_id}: {exc}") from exc
            baseline_answer = parse_answer(baseline_raw)
            rows.append(
                {
                    **common,
                    "run_id": baseline_id,
                    "condition": "baseline",
                    "pressure_type": None,
                    "user_suggestion": None,
                    "messages": baseline_messages,
                    "raw_answer": baseline_raw,
                    "answer": baseline_answer,
                }
            )

            for ptype in PRESSURE_TYPES:
                suggestion = item["wrong_suggestion"].upper() if ptype == "wrong_suggestion" else None
                messages = [
                    {"role": "user", "content": baseline_prompt},
                    {"role": "assistant", "content": baseline_raw},
                    {"role": "user", "content": pressure_text(config, ptype, variant_id, item)},
                ]
                run_id = f"{item['item_id']}__v{variant_id}__{ptype}"
                try:
                    raw = call_ollama_messages(messages, model=model)
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"Model call failed for {run_id}: {exc}") from exc
                rows.append(
                    {
                        **common,
                        "run_id": run_id,
                        "condition": "wrong_pressure",
                        "pressure_type": ptype,
                        "user_suggestion": suggestion,
                        "messages": messages,
                        "raw_answer": raw,
                        "answer": parse_answer(raw),
                    }
                )

            if request.include_correct_control and baseline_answer is not None and baseline_answer != item["correct_answer"].upper():
                messages = [
                    {"role": "user", "content": baseline_prompt},
                    {"role": "assistant", "content": baseline_raw},
                    {"role": "user", "content": pressure_text(config, "correct_suggestion", variant_id, item)},
                ]
                run_id = f"{item['item_id']}__v{variant_id}__correct_suggestion"
                try:
                    raw = call_ollama_messages(messages, model=model)
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"Model call failed for {run_id}: {exc}") from exc
                rows.append(
                    {
                        **common,
                        "run_id": run_id,
                        "condition": "correct_pressure",
                        "pressure_type": "correct_suggestion",
                        "user_suggestion": item["correct_answer"].upper(),
                        "messages": messages,
                        "raw_answer": raw,
                        "answer": parse_answer(raw),
                    }
                )

    classified = classify_rows(rows)
    report = score_classified(classified)
    return {
        "model": model,
        "protocol": config["protocol"],
        "run_count": len(rows),
        "report": report,
        "runs": classified,
    }
