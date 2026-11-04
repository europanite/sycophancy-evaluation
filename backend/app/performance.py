"""Deterministic, auditable checks for a small multi-capability LLM suite.

These are local diagnostic tasks, not established leaderboard benchmarks. In
particular, passing a keyword check is not equivalent to semantic quality.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "performance_cases.json"
GRADERS = {"exact", "choice", "numeric", "json", "contains", "regex", "calibrated_choice"}
STRUCTURED_GRADERS = {"choice", "numeric", "json", "calibrated_choice"}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _normalized(value: str, *, case_sensitive: bool = False) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    return normalized if case_sensitive else normalized.casefold()


def load_cases(path: Path | None = None) -> list[dict]:  # noqa: PLR0912
    with (path or DATA_FILE).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("performance_cases.json must contain a nonempty cases array")
    seen: set[str] = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k].strip() for k in ("id", "category", "prompt")):
            raise ValueError("each case needs nonempty id, category and prompt strings")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"]) or case["id"] in seen:
            raise ValueError(f"invalid or duplicate case id: {case['id']}")
        seen.add(case["id"])
        grader = case.get("grader")
        if not isinstance(grader, dict) or grader.get("type") not in GRADERS:
            raise ValueError(f"{case['id']}: unsupported grader")
        kind = grader["type"]
        if kind in {"exact", "choice", "numeric", "json", "calibrated_choice"} and "expected" not in grader:
            raise ValueError(f"{case['id']}: expected is required")
        if kind == "choice" and grader["expected"] not in ("A", "B", "C", "D"):
            raise ValueError(f"{case['id']}: choice expected must be A-D")
        if kind == "calibrated_choice" and grader["expected"] not in ("A", "B", "C", "D"):
            raise ValueError(f"{case['id']}: calibrated choice expected must be A-D")
        if kind == "numeric":
            if not isinstance(grader["expected"], (int, float)) or isinstance(grader["expected"], bool) or not math.isfinite(grader["expected"]):
                raise ValueError(f"{case['id']}: numeric expected must be a number")
            tolerance = grader.get("tolerance", 0)
            if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"{case['id']}: numeric tolerance must be nonnegative")
        if kind == "contains":
            if not isinstance(grader.get("required"), list) or not grader["required"]:
                raise ValueError(f"{case['id']}: contains requires a nonempty required list")
            if not isinstance(grader.get("forbidden", []), list):
                raise ValueError(f"{case['id']}: forbidden must be a list")
            if any(not isinstance(x, str) or not x for x in grader["required"] + grader.get("forbidden", [])):
                raise ValueError(f"{case['id']}: contains phrases must be strings")
            for bound in ("min_words", "max_words"):
                if bound in grader and (type(grader[bound]) is not int or grader[bound] < 0):
                    raise ValueError(f"{case['id']}: {bound} must be a nonnegative integer")
            if grader.get("min_words", 0) > grader.get("max_words", float("inf")):
                raise ValueError(f"{case['id']}: min_words cannot exceed max_words")
        if kind == "regex":
            if not isinstance(grader.get("pattern"), str):
                raise ValueError(f"{case['id']}: regex pattern is required")
            re.compile(grader["pattern"])
    return data["cases"]


def select_cases(cases: list[dict], *, categories: list[str] | None, item_count: int) -> list[dict]:
    available = {case["category"] for case in cases}
    if categories:
        unknown = set(categories) - available
        if unknown:
            raise ValueError(f"Unknown categories: {', '.join(sorted(unknown))}")
    selected = [case for case in cases if not categories or case["category"] in categories]
    if item_count > len(selected):
        raise ValueError(f"Requested {item_count} cases, but only {len(selected)} available")
    # Round-robin by category: small runs don't accidentally benchmark one domain.
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in selected:
        groups[case["category"]].append(case)
    spread: list[dict] = []
    while any(groups.values()) and len(spread) < item_count:
        for name in sorted(groups):
            if groups[name] and len(spread) < item_count:
                spread.append(groups[name].pop(0))
    return spread


def _strict_json(raw: str):
    # No markdown fences, comments, or text outside JSON: this measures format compliance.
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def grade(case: dict, raw: str) -> dict:
    spec = case["grader"]
    kind = spec["type"]
    result = {"passed": False, "format_valid": None, "reason": "mismatch", "confidence": None, "parsed_answer": None}
    if kind == "exact":
        result["passed"] = _normalized(raw, case_sensitive=spec.get("case_sensitive", False)) == _normalized(str(spec["expected"]), case_sensitive=spec.get("case_sensitive", False))
    elif kind == "choice":
        match = re.fullmatch(r"\s*([ABCD])\s*", raw)
        result["format_valid"] = match is not None
        result["parsed_answer"] = match.group(1) if match else None
        result["passed"] = bool(match and match.group(1) == spec["expected"])
    elif kind == "numeric":
        number = raw.strip()
        result["format_valid"] = bool(NUMBER.fullmatch(number))
        if result["format_valid"]:
            parsed = float(number)
            result["format_valid"] = math.isfinite(parsed)
            if result["format_valid"]:
                result["parsed_answer"] = number
                result["passed"] = abs(parsed - spec["expected"]) <= spec.get("tolerance", 0)
    elif kind in {"json", "calibrated_choice"}:
        parsed_ok = True
        try:
            parsed = _strict_json(raw)
        except (ValueError, TypeError):
            parsed_ok = False
            parsed = None
        if kind == "json":
            result["format_valid"] = parsed_ok
            # JSON canonicalization distinguishes booleans from numbers, including nested values.
            result["passed"] = parsed_ok and json.dumps(parsed, sort_keys=True) == json.dumps(spec["expected"], sort_keys=True)
        else:
            valid = (
                parsed_ok and isinstance(parsed, dict) and set(parsed) == {"answer", "confidence"}
                and type(parsed["answer"]) is str and parsed["answer"] in ("A", "B", "C", "D")
                and type(parsed["confidence"]) in (int, float)
                and math.isfinite(parsed["confidence"]) and 0 <= parsed["confidence"] <= 1
            )
            result["format_valid"] = valid
            if valid:
                result["confidence"] = float(parsed["confidence"])
                result["parsed_answer"] = parsed["answer"]
                result["passed"] = parsed["answer"] == spec["expected"]
    elif kind == "contains":
        normalized = _normalized(raw, case_sensitive=spec.get("case_sensitive", False))
        required = all(_normalized(s, case_sensitive=spec.get("case_sensitive", False)) in normalized for s in spec["required"])
        forbidden = any(_normalized(s, case_sensitive=spec.get("case_sensitive", False)) in normalized for s in spec.get("forbidden", []))
        words = len(raw.split())
        result["passed"] = required and not forbidden and spec.get("min_words", 0) <= words <= spec.get("max_words", float("inf"))
    elif kind == "regex":
        result["passed"] = re.fullmatch(spec["pattern"], raw.strip(), flags=re.DOTALL) is not None
    result["reason"] = "pass" if result["passed"] else ("invalid_format" if result["format_valid"] is False else "mismatch")
    return result


def score(cases: list[dict], responses: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {case["id"]: case for case in cases}
    seen: set[tuple[str, int]] = set()
    rows: list[dict] = []
    for response in responses:
        case_id, repeat_index = response["case_id"], response["repeat_index"]
        key = (case_id, repeat_index)
        if case_id not in by_id or key in seen:
            raise ValueError(f"unknown or duplicate response: {case_id} repeat {repeat_index}")
        seen.add(key)
        case = by_id[case_id]
        outcome = grade(case, response["answer"])
        rows.append({
            "case_id": case_id, "category": case["category"], "repeat_index": repeat_index,
            "grader_type": case["grader"]["type"], "prompt": case["prompt"],
            "messages": [{"role": "user", "content": case["prompt"]}],
            "raw_answer": response["answer"], **outcome,
            "elapsed_ms": response.get("elapsed_ms"),
            "prompt_tokens": response.get("prompt_tokens"),
            "completion_tokens": response.get("completion_tokens"),
            "generation_tokens_per_second": response.get("generation_tokens_per_second"),
        })
    def accuracy(items):
        return sum(row["passed"] for row in items) / len(items) if items else None

    per_category = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        per_category[category] = {"runs": len(subset), "accuracy": accuracy(subset), "cases": len({r["case_id"] for r in subset})}

    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    repeated = [group for group in by_case.values() if len(group) > 1]
    consistent = sum(len({r["raw_answer"] for r in group}) == 1 for group in repeated)
    pass_consistent = sum(all(r["passed"] for r in group) for group in repeated)

    formatted = [r for r in rows if r["grader_type"] in STRUCTURED_GRADERS]
    calibrated = [r for r in rows if r["confidence"] is not None]
    # ECE: five fixed-width bins; descriptive only for tiny samples.
    bins = [[] for _ in range(5)]
    for row in calibrated:
        bins[min(int(row["confidence"] * 5), 4)].append(row)
    ece = sum(len(b) / len(calibrated) * abs(statistics.mean(r["confidence"] for r in b) - accuracy(b)) for b in bins if b) if calibrated else None
    brier = statistics.mean((r["confidence"] - float(r["passed"])) ** 2 for r in calibrated) if calibrated else None

    def telemetry(key):
        values = [r[key] for r in rows if r[key] is not None]
        return {"observed": len(values), "mean": statistics.mean(values) if values else None,
                "p50": statistics.median(values) if values else None,
                "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None}

    return ({
        "protocol": "local_deterministic_performance_v1", "runs": len(rows), "unique_cases": len(by_case),
        "overall_accuracy": accuracy(rows), "category_breakdown": per_category,
        "structured_format_compliance": sum(r["format_valid"] is True for r in formatted) / len(formatted) if formatted else None,
        "repeat_consistency": consistent / len(repeated) if repeated else None,
        "all_repeats_correct": pass_consistent / len(repeated) if repeated else None,
        "repeated_cases": len(repeated),
        "calibration": {"samples": len(calibrated), "brier_score": brier, "ece_5_bins": ece},
        "latency_ms": telemetry("elapsed_ms"),
        "generation_tokens_per_second": telemetry("generation_tokens_per_second"),
        "prompt_tokens": telemetry("prompt_tokens"),
        "completion_tokens": telemetry("completion_tokens"),
    }, rows)
