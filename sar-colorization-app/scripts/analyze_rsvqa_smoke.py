#!/usr/bin/env python3
"""Diagnose the completed RSVQA-LR smoke run without changing production behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path("artifacts/rsvqa_smoke")
DEFAULT_OUTPUT = DEFAULT_ROOT / "diagnostics"
PROJECTED_BENCHMARK_SIZE = 10_004
EXPECTED_TASKS = {
    "comp": "comparison_vqa",
    "count": "count_vqa",
    "presence": "presence_vqa",
    "rural_urban": "rural_urban_classification",
}
SUPPLIED_EXPECTATIONS = {
    "total_samples": 50,
    "completed_count": 50,
    "correct_count": 19,
    "overall_accuracy": 0.38,
    "null_answer_count": 15,
    "endpoint_error_count": 0,
    "average_latency_ms": 1060.34,
    "median_latency_ms": 949.5,
    "p95_latency_ms": 2233.75,
    "slowest_latency_ms": 4851.0,
}
FAILURE_CATEGORIES = (
    "routing failure",
    "unsupported target or vocabulary",
    "insufficient scene evidence",
    "SVE disagreement or weak evidence",
    "caption evidence mismatch",
    "grounding unavailable",
    "grounding returned zero regions",
    "grounding overcount",
    "grounding undercount",
    "comparison operand unavailable",
    "comparison logic failure",
    "answer normalization failure",
    "model prediction error",
    "unknown",
)


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_id", "question", "question_type", "normalized_ground_truth",
            "normalized_prediction", "correct", "outcome", "confidence", "task",
            "status", "result_status", "effective_latency_ms", "warnings_json",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"predictions.csv is missing columns: {', '.join(sorted(missing))}")
        rows: List[Dict[str, Any]] = []
        for row in reader:
            try:
                warnings = json.loads(row.get("warnings_json") or "[]")
            except json.JSONDecodeError:
                warnings = []
            try:
                reuse = json.loads(row.get("reuse_observations_json") or "{}")
            except json.JSONDecodeError:
                reuse = {}
            parsed: Dict[str, Any] = dict(row)
            parsed.update(
                {
                    "manifest_index": int(row.get("manifest_index") or len(rows) + 1),
                    "correct": _bool(row.get("correct")),
                    "completed": _bool(row.get("completed")),
                    "confidence": _float(row.get("confidence")),
                    "latency_ms": _float(row.get("effective_latency_ms")),
                    "normalized_prediction": row.get("normalized_prediction") or None,
                    "answer": row.get("answer") or None,
                    "warnings": warnings if isinstance(warnings, list) else [],
                    "reuse_observations": reuse if isinstance(reuse, dict) else {},
                }
            )
            rows.append(parsed)
    if not rows:
        raise ValueError("predictions.csv contains no samples.")
    question_ids = [str(row["question_id"]) for row in rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("predictions.csv contains duplicate question IDs.")
    return sorted(rows, key=lambda row: row["manifest_index"])


def is_null_prediction(row: Mapping[str, Any]) -> bool:
    return row.get("normalized_prediction") in (None, "")


def is_answered(row: Mapping[str, Any]) -> bool:
    return bool(row.get("completed", True)) and not is_null_prediction(row)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _percent(value: Optional[float]) -> Optional[float]:
    return round(value * 100.0, 2) if value is not None else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    selected = [float(value) for value in values if value is not None]
    return round(statistics.fmean(selected), 6) if selected else None


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    return {
        "samples": len(rows),
        "latency_samples": len(values),
        "average_latency_ms": round(statistics.fmean(values), 3) if values else None,
        "median_latency_ms": round(statistics.median(values), 3) if values else None,
        "p95_latency_ms": round(float(np.percentile(values, 95)), 3) if values else None,
        "minimum_latency_ms": round(min(values), 3) if values else None,
        "maximum_latency_ms": round(max(values), 3) if values else None,
    }


def majority_baseline(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(row.get("normalized_ground_truth")) for row in rows)
    if not counts:
        return {"label": None, "count": 0, "accuracy": None, "accuracy_percent": None}
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    accuracy = count / len(rows)
    return {"label": label, "count": count, "accuracy": accuracy, "accuracy_percent": _percent(accuracy)}


def compute_group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    answered = sum(is_answered(row) for row in rows)
    nulls = sum(is_null_prediction(row) for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    coverage = _rate(answered, total)
    overall_accuracy = _rate(correct, total)
    answered_accuracy = _rate(correct, answered)
    result = {
        "samples": total,
        "answered_count": answered,
        "null_count": nulls,
        "coverage": coverage,
        "coverage_percent": _percent(coverage),
        "null_rate": _rate(nulls, total),
        "null_rate_percent": _percent(_rate(nulls, total)),
        "correct_count": correct,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": _percent(overall_accuracy),
        "answered_only_accuracy": answered_accuracy,
        "answered_only_accuracy_percent": _percent(answered_accuracy),
        "average_confidence": _mean(row.get("confidence") for row in rows),
        **latency_summary(rows),
        "majority_baseline": majority_baseline(rows),
    }
    return result


def compute_core_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = compute_group_metrics(rows)
    by_type: Dict[str, Dict[str, Any]] = {}
    by_task: Dict[str, Dict[str, Any]] = {}
    for question_type in sorted({str(row.get("question_type")) for row in rows}):
        by_type[question_type] = compute_group_metrics([row for row in rows if row.get("question_type") == question_type])
    for task in sorted({str(row.get("task") or "missing") for row in rows}):
        by_task[task] = latency_summary([row for row in rows if str(row.get("task") or "missing") == task])
    outcome_groups = {
        "correct": [row for row in rows if row.get("correct")],
        "wrong": [row for row in rows if not row.get("correct") and not is_null_prediction(row)],
        "null": [row for row in rows if is_null_prediction(row)],
    }
    metrics.update(
        {
            "by_type": by_type,
            "confidence_by_outcome": {
                name: {"samples": len(group), "average_confidence": _mean(row.get("confidence") for row in group)}
                for name, group in outcome_groups.items()
            },
            "latency_by_outcome": {name: latency_summary(group) for name, group in outcome_groups.items()},
            "latency_by_task": by_task,
        }
    )
    return metrics


def classify_wording(question: str, question_type: str) -> str:
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    if question_type == "rural_urban":
        if "rural" in normalized and "urban" in normalized:
            return "rural-or-urban"
        if "urban" in normalized:
            return "direct urban"
        if "rural" in normalized:
            return "direct rural"
        return "other"
    if question_type == "presence":
        if normalized.startswith("is there"):
            return "is there"
        if normalized.startswith("are there"):
            return "are there"
        if re.search(r"does (?:the|this) image contain", normalized):
            return "does the image contain"
        if any(term in normalized for term in ("visible", "can you see", " present")):
            return "visible/can you see"
        return "other"
    if question_type == "count":
        if any(term in normalized for term in (" next to ", " near ", " adjacent to ")):
            return "relational count"
        if normalized.startswith("how many"):
            return "how many"
        if "number of" in normalized:
            return "number of"
        if "amount of" in normalized:
            return "amount of"
        return "other"
    if question_type == "comp":
        if re.search(r"\b(?:less|fewer)\b", normalized):
            return "less/fewer"
        if re.search(r"\b(?:more|greater)\b", normalized):
            return "more/greater"
        if re.search(r"\b(?:equal|same)\b", normalized):
            return "equal/same"
        return "other"
    return "other"


ENTITY_PATTERNS: Tuple[Tuple[str, str, re.Pattern], ...] = (
    ("commercial buildings", "commercial buildings", re.compile(r"\bcommercial buildings?\b")),
    ("residential buildings", "residential buildings", re.compile(r"\bresidential buildings?\b")),
    ("agricultural land", "agricultural land", re.compile(r"\bagricultural land\b")),
    ("roads", "roads", re.compile(r"\broads?\b")),
    ("buildings", "buildings", re.compile(r"\bbuildings?\b")),
    ("water", "water", re.compile(r"\bwater(?: areas?| bodies?)?\b")),
    ("grass", "grass", re.compile(r"\bgrass(?: areas?|lands?)?\b")),
    ("forest", "forest", re.compile(r"\bforests?\b")),
    ("heath", "heath", re.compile(r"\bheath\b")),
    ("other", "wetland", re.compile(r"\bwetlands?\b")),
    ("other", "scrub", re.compile(r"\bscrubs?\b")),
    ("other", "industrial", re.compile(r"\bindustrials?\b")),
)


def extract_targets(question: str, question_type: str) -> List[Dict[str, str]]:
    if question_type == "rural_urban":
        return [{"target": "rural/urban scene", "detail": "rural/urban scene"}]
    normalized = question.lower()
    matches: List[Tuple[int, int, str, str]] = []
    for target, detail, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(normalized):
            matches.append((match.start(), match.end(), target, detail))
    accepted: List[Tuple[int, int, str, str]] = []
    for candidate in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(candidate[0] < item[1] and item[0] < candidate[1] for item in accepted):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: item[0])
    return [{"target": item[2], "detail": item[3]} for item in accepted] or [{"target": "other", "detail": "unresolved"}]


def comparison_relation(question: str) -> Optional[str]:
    normalized = question.lower()
    if re.search(r"\b(?:less|fewer)\b", normalized):
        return "less/fewer"
    if re.search(r"\b(?:more|greater)\b", normalized):
        return "more/greater"
    if re.search(r"\b(?:equal|same)\b", normalized):
        return "equal/same"
    return None


def _warning_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(item) for item in row.get("warnings", [])).lower()


def _integer(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    return int(text) if re.fullmatch(r"0|[1-9]\d*", text) else None


def _normalizable_raw_answer(value: Any) -> bool:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return text in {"yes", "no", "rural", "urban"} or bool(re.fullmatch(r"0|[1-9]\d*", text))


def classify_failure(row: Mapping[str, Any]) -> Tuple[str, str]:
    """Assign one conservative primary cause using only persisted fields."""
    if row.get("correct"):
        raise ValueError("Failure taxonomy can only classify wrong or null rows.")
    question_type = str(row.get("question_type") or "")
    task = str(row.get("task") or "")
    warning_text = _warning_text(row)
    targets = extract_targets(str(row.get("question") or ""), question_type)
    target_names = {item["target"] for item in targets}
    target_details = {item["detail"] for item in targets}
    expected_task = EXPECTED_TASKS.get(question_type)

    if expected_task and task != expected_task:
        return "routing failure", f"Stored task was {task or 'missing'}; expected {expected_task} for manifest type {question_type}."
    if "did not match a supported deterministic routing rule" in warning_text:
        return "routing failure", "Stored warning states that no deterministic routing rule matched."
    if row.get("answer") not in (None, "") and is_null_prediction(row) and _normalizable_raw_answer(row.get("answer")):
        return "answer normalization failure", "Raw answer was in the allowed benchmark domain but normalized_prediction is null."
    if question_type == "comp" and is_null_prediction(row):
        return "comparison operand unavailable", "Comparison returned insufficient evidence; persisted artifacts do not contain both operand counts."
    vocabulary_gap = bool(target_details.intersection({"heath", "wetland", "scrub", "industrial"}))
    vocabulary_gap = vocabulary_gap or (question_type == "count" and bool(target_names.intersection({"forest", "grass", "other"})))
    if is_null_prediction(row) and vocabulary_gap:
        return "unsupported target or vocabulary", "Target wording has no persisted count/presence answer and no supported evidence path is recorded."
    no_regions = "no reliable localized region was found" in warning_text
    prediction_count = _integer(row.get("normalized_prediction"))
    truth_count = _integer(row.get("normalized_ground_truth"))
    if no_regions and (
        (question_type == "count" and prediction_count == 0 and truth_count not in (None, 0))
        or (question_type == "comp" and not is_null_prediction(row))
    ):
        return "grounding returned zero regions", "Persisted warnings report no reliable localized region for one or more required entities."
    if question_type == "count" and prediction_count is not None and truth_count is not None:
        if prediction_count > truth_count:
            return "grounding overcount", f"Predicted accepted-region count {prediction_count} exceeds label {truth_count}."
        if prediction_count < truth_count:
            return "grounding undercount", f"Predicted accepted-region count {prediction_count} is below label {truth_count}."
    if is_null_prediction(row) and question_type in {"count", "comp"} and "rs_grounder" not in str(row.get("model_used") or ""):
        return "grounding unavailable", "No rs_grounder execution is recorded for a count-dependent question."
    if "sve" in warning_text and any(term in warning_text for term in ("disagree", "weak", "limited")):
        return "SVE disagreement or weak evidence", "Persisted warning explicitly reports weak or disagreeing SVE evidence."
    if "scene-level evidence" in warning_text and any(term in warning_text for term in ("disagree", "weak", "limited")):
        return "SVE disagreement or weak evidence", "Persisted warning explicitly reports limited scene-level evidence."
    if "caption" in warning_text and any(term in warning_text for term in ("mismatch", "disagree", "inconsistent")):
        return "caption evidence mismatch", "Persisted warning explicitly reports caption inconsistency."
    if is_null_prediction(row) and "insufficient" in str(row.get("status") or "").lower() + " " + warning_text:
        return "insufficient scene evidence", "Endpoint status or warning explicitly reports insufficient evidence."
    if question_type in EXPECTED_TASKS and task == expected_task and not is_null_prediction(row):
        return "model prediction error", "Routing and answer domain were valid, but the persisted normalized token differs from the label."
    return "unknown", "Stored fields do not support a more specific primary cause."


def add_diagnostic_fields(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        targets = extract_targets(str(row["question"]), str(row["question_type"]))
        row["wording_pattern"] = classify_wording(str(row["question"]), str(row["question_type"]))
        row["targets"] = targets
        row["target_1"] = targets[0]["target"] if targets else None
        row["target_2"] = targets[1]["target"] if len(targets) > 1 else None
        row["target_details"] = [item["detail"] for item in targets]
        row["relation"] = comparison_relation(str(row["question"])) if row["question_type"] == "comp" else None
        if not row["correct"]:
            category, evidence = classify_failure(row)
            row["failure_category"] = category
            row["failure_evidence"] = evidence
        else:
            row["failure_category"] = None
            row["failure_evidence"] = None


def metrics_by_wording(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["question_type"]), str(row["wording_pattern"]))].append(row)
    output = []
    for (question_type, pattern), group in sorted(grouped.items()):
        output.append({"question_type": question_type, "wording_pattern": pattern, **compute_group_metrics(group)})
    return output


def metrics_by_target(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    details: Dict[str, set] = defaultdict(set)
    for row in rows:
        seen = set()
        for target in row.get("targets", []):
            name = str(target["target"])
            details[name].add(str(target["detail"]))
            if name not in seen:
                grouped[name].append(row)
                seen.add(name)
    output = []
    for target, group in sorted(grouped.items()):
        output.append({"target": target, "target_details": sorted(details[target]), **compute_group_metrics(group)})
    return output


def count_answer_bucket(value: Any) -> str:
    number = _integer(value)
    if number is None:
        return "invalid"
    if number == 0:
        return "ground truth 0"
    if number == 1:
        return "ground truth 1"
    return "ground truth 2+"


def count_bucket_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    count_rows = [row for row in rows if row.get("question_type") == "count"]
    output = {}
    for bucket in ("ground truth 0", "ground truth 1", "ground truth 2+"):
        group = [row for row in count_rows if count_answer_bucket(row.get("normalized_ground_truth")) == bucket]
        output[bucket] = compute_group_metrics(group)
    answered_count = [row for row in count_rows if is_answered(row)]
    zero_predictions = sum(row.get("normalized_prediction") == "0" for row in answered_count)
    output["zero_prediction_dominance"] = {
        "answered_count_samples": len(answered_count),
        "zero_predictions": zero_predictions,
        "zero_prediction_rate": _rate(zero_predictions, len(answered_count)),
        "zero_prediction_rate_percent": _percent(_rate(zero_predictions, len(answered_count))),
        "accuracy_exceeds_majority_baseline": compute_group_metrics(count_rows)["overall_accuracy"]
        > compute_group_metrics(count_rows)["majority_baseline"]["accuracy"] if count_rows else False,
    }
    return output


def answer_distributions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for question_type in sorted({str(row["question_type"]) for row in rows}):
        group = [row for row in rows if row["question_type"] == question_type]
        ground_truth = Counter(str(row["normalized_ground_truth"]) for row in group)
        predictions = Counter(str(row["normalized_prediction"]) for row in group if not is_null_prediction(row))
        null_ground_truth = Counter(str(row["normalized_ground_truth"]) for row in group if is_null_prediction(row))
        most_frequent = sorted(predictions.items(), key=lambda item: (-item[1], item[0]))[0] if predictions else (None, 0)
        result[question_type] = {
            "sample_count": len(group),
            "ground_truth_distribution": dict(sorted(ground_truth.items())),
            "predicted_distribution": dict(sorted(predictions.items())),
            "null_count": sum(null_ground_truth.values()),
            "null_ground_truth_distribution": dict(sorted(null_ground_truth.items())),
            "most_frequent_prediction": {"answer": most_frequent[0], "count": most_frequent[1]},
            "majority_class_baseline": majority_baseline(group),
            "observed_accuracy": compute_group_metrics(group)["overall_accuracy"],
            "exceeds_majority_baseline": compute_group_metrics(group)["overall_accuracy"] > majority_baseline(group)["accuracy"],
        }
    return result


def verify_inputs(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    errors: Sequence[Any],
    confusion_overall: Path,
    confusion_by_type: Path,
) -> Dict[str, Any]:
    metrics = compute_core_metrics(rows)
    slowest = max((row.get("latency_ms") or 0 for row in rows), default=0)
    actual = {
        "total_samples": len(rows),
        "completed_count": sum(bool(row.get("completed")) for row in rows),
        "correct_count": sum(bool(row.get("correct")) for row in rows),
        "overall_accuracy": metrics["overall_accuracy"],
        "null_answer_count": metrics["null_count"],
        "endpoint_error_count": len(errors),
        "average_latency_ms": metrics["average_latency_ms"],
        "median_latency_ms": metrics["median_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "slowest_latency_ms": slowest,
    }
    checks = []
    for name, expected in SUPPLIED_EXPECTATIONS.items():
        value = actual[name]
        passed = math.isclose(float(value), float(expected), rel_tol=0, abs_tol=1e-6) if isinstance(expected, float) else value == expected
        checks.append({"name": name, "expected": expected, "actual": value, "passed": passed, "source": "predictions.csv"})
    stored_summary = results.get("summary", {}) if isinstance(results, dict) else {}
    for name in ("total_samples", "completed_count", "correct_count", "overall_accuracy", "null_answer_count", "endpoint_error_count"):
        stored_value = stored_summary.get(name)
        value = actual[name]
        passed = math.isclose(float(value), float(stored_value), abs_tol=1e-9) if stored_value is not None else False
        checks.append({"name": f"results.json:{name}", "expected": value, "actual": stored_value, "passed": passed, "source": "results.json"})

    def confusion_total(path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(int(row.get("total") or 0) for row in csv.DictReader(handle))

    for label, path in (("confusion_matrix_overall", confusion_overall), ("confusion_matrix_by_type", confusion_by_type)):
        value = confusion_total(path)
        checks.append({"name": f"{label}:total", "expected": len(rows), "actual": value, "passed": value == len(rows), "source": path.name})
    return {"all_passed": all(item["passed"] for item in checks), "checks": checks, "recomputed": actual}


def projection(metrics: Mapping[str, Any], size: int = PROJECTED_BENCHMARK_SIZE) -> Dict[str, Any]:
    accuracy = float(metrics["overall_accuracy"] or 0)
    null_rate = float(metrics["null_rate"] or 0)
    mean_latency = float(metrics["average_latency_ms"] or 0)
    p95_latency = float(metrics["p95_latency_ms"] or 0)
    mean_seconds = size * mean_latency / 1000.0
    p95_seconds = size * p95_latency / 1000.0
    return {
        "projection_only": True,
        "projected_question_count": size,
        "assumption": "Smoke-run accuracy, null rate, and sequential per-request latency remain unchanged.",
        "expected_correct_answers": size * accuracy,
        "expected_correct_answers_rounded": round(size * accuracy),
        "expected_null_answers": size * null_rate,
        "expected_null_answers_rounded": round(size * null_rate),
        "mean_latency_runtime": {"seconds": round(mean_seconds, 3), "hours": round(mean_seconds / 3600.0, 3)},
        "p95_latency_runtime": {"seconds": round(p95_seconds, 3), "hours": round(p95_seconds / 3600.0, 3)},
        "caution": "The p95-based duration is a rate scenario, not a measured end-to-end runtime percentile.",
    }


def repair_priorities(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    failures = [row for row in rows if not row.get("correct")]
    counts = Counter(str(row.get("failure_category")) for row in failures)
    dense_count_failures = sum(
        row.get("question_type") == "count"
        and count_answer_bucket(row.get("normalized_ground_truth")) == "ground truth 2+"
        and not row.get("correct")
        for row in rows
    )
    definitions = [
        {
            "issue": "Expand controlled target/evidence availability for missing operands and vocabulary",
            "section": "evidence-availability fixes",
            "affected_samples": counts["unsupported target or vocabulary"] + counts["comparison operand unavailable"] + counts["grounding unavailable"],
            "expected_recoverability": 0.75,
            "recommendation": "Add auditable support for currently unresolved target terms and require both comparison operands; do not use benchmark labels or metadata at inference time.",
        },
        {
            "issue": "Treat zero accepted grounding regions as uncertain evidence, not an automatically reliable zero count",
            "section": "evidence-availability fixes",
            "affected_samples": counts["grounding returned zero regions"],
            "expected_recoverability": 0.60,
            "recommendation": "Calibrate zero-region behavior and distinguish no detection from confirmed absence before deriving counts or comparisons.",
        },
        {
            "issue": "Cover benchmark presence constructions that fell through to generic or unsupported routing",
            "section": "routing/normalization fixes",
            "affected_samples": counts["routing failure"],
            "expected_recoverability": 0.90,
            "recommendation": "Add text-only routing tests for observed 'Is a … present' forms while retaining centralized answer normalization.",
        },
        {
            "issue": "Improve rural/urban and presence discrimination after routing succeeds",
            "section": "model-quality limitations",
            "affected_samples": counts["model prediction error"] + counts["SVE disagreement or weak evidence"] + counts["caption evidence mismatch"],
            "expected_recoverability": 0.40,
            "recommendation": "Evaluate calibrated or retrained scene/presence evidence on a held-out set; the smoke result does not justify claiming improvement.",
        },
        {
            "issue": "Resolve the mismatch between dense RSVQA count labels and accepted-region box counts",
            "section": "benchmark-specific risks",
            "affected_samples": dense_count_failures,
            "expected_recoverability": 0.15,
            "recommendation": "Determine whether a scientifically valid counting model can represent dense road/building/area counts before a full run; never hard-code answers.",
        },
        {
            "issue": "Investigate unclassified failures only after richer evidence is persisted",
            "section": "benchmark-specific risks",
            "affected_samples": counts["unknown"],
            "expected_recoverability": 0.25,
            "recommendation": "Persist safe operand/count evidence in future benchmark runs; do not infer causes from absent fields.",
        },
    ]
    for item in definitions:
        item["impact_score"] = round(item["affected_samples"] * item["expected_recoverability"], 3)
    ranked = sorted(definitions, key=lambda item: (-item["impact_score"], -item["affected_samples"], item["issue"]))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fieldnames:
                value = row.get(key)
                encoded[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            writer.writerow(encoded)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric_csv_row(prefix: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **prefix,
        "samples": metrics["samples"],
        "answered_count": metrics["answered_count"],
        "null_count": metrics["null_count"],
        "coverage": metrics["coverage"],
        "coverage_percent": metrics["coverage_percent"],
        "null_rate": metrics["null_rate"],
        "null_rate_percent": metrics["null_rate_percent"],
        "correct_count": metrics["correct_count"],
        "overall_accuracy": metrics["overall_accuracy"],
        "overall_accuracy_percent": metrics["overall_accuracy_percent"],
        "answered_only_accuracy": metrics["answered_only_accuracy"],
        "answered_only_accuracy_percent": metrics["answered_only_accuracy_percent"],
        "average_confidence": metrics["average_confidence"],
        "average_latency_ms": metrics["average_latency_ms"],
        "median_latency_ms": metrics["median_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "majority_label": metrics["majority_baseline"]["label"],
        "majority_baseline_accuracy": metrics["majority_baseline"]["accuracy"],
        "majority_baseline_accuracy_percent": metrics["majority_baseline"]["accuracy_percent"],
    }


METRIC_FIELDS = (
    "samples", "answered_count", "null_count", "coverage", "coverage_percent", "null_rate",
    "null_rate_percent", "correct_count", "overall_accuracy", "overall_accuracy_percent",
    "answered_only_accuracy", "answered_only_accuracy_percent", "average_confidence",
    "average_latency_ms", "median_latency_ms", "p95_latency_ms", "majority_label",
    "majority_baseline_accuracy", "majority_baseline_accuracy_percent",
)


def write_answer_distribution(path: Path, distributions: Mapping[str, Any]) -> None:
    rows = []
    for question_type, values in distributions.items():
        common = {
            "question_type": question_type,
            "sample_count": values["sample_count"],
            "most_frequent_prediction": values["most_frequent_prediction"]["answer"],
            "most_frequent_prediction_count": values["most_frequent_prediction"]["count"],
            "majority_ground_truth_label": values["majority_class_baseline"]["label"],
            "majority_baseline_accuracy": values["majority_class_baseline"]["accuracy"],
            "observed_accuracy": values["observed_accuracy"],
            "exceeds_majority_baseline": values["exceeds_majority_baseline"],
        }
        for distribution_name, mapping in (
            ("ground_truth", values["ground_truth_distribution"]),
            ("prediction", values["predicted_distribution"]),
            ("null_ground_truth", values["null_ground_truth_distribution"]),
        ):
            if not mapping and distribution_name == "null_ground_truth":
                rows.append({**common, "distribution": distribution_name, "answer": "<none>", "count": 0, "share_of_type": 0.0})
            for answer, count in mapping.items():
                rows.append({**common, "distribution": distribution_name, "answer": answer, "count": count, "share_of_type": count / values["sample_count"]})
    fields = (
        "question_type", "distribution", "answer", "count", "share_of_type", "sample_count",
        "most_frequent_prediction", "most_frequent_prediction_count", "majority_ground_truth_label",
        "majority_baseline_accuracy", "observed_accuracy", "exceeds_majority_baseline",
    )
    _write_csv(path, fields, rows)


def repair_markdown(priorities: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# RSVQA Smoke Repair Priorities",
        "",
        "Priority score is `affected samples × expected recoverability`. Recoverability is an explicit diagnostic estimate, not measured improvement.",
        "No recommendation uses ground-truth metadata at inference time or hard-codes benchmark answers.",
        "",
    ]
    for section in (
        "routing/normalization fixes", "evidence-availability fixes",
        "model-quality limitations", "benchmark-specific risks",
    ):
        lines.extend([f"## {section.title()}", ""])
        section_items = [item for item in priorities if item["section"] == section]
        if not section_items:
            lines.extend(["No evidence-backed issue in this category.", ""])
            continue
        for item in section_items:
            lines.extend(
                [
                    f"### {item['rank']}. {item['issue']}",
                    "",
                    f"- Affected samples: {item['affected_samples']}",
                    f"- Expected recoverability: {item['expected_recoverability']:.2f}",
                    f"- Impact score: {item['impact_score']:.2f}",
                    f"- Recommendation: {item['recommendation']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Recommended Decision",
            "",
            "**B. Fix routing/evidence first.** The run has 30% null coverage loss and identifiable routing, vocabulary, and operand-availability gaps. Resolve those before paying for a 10,004-question run or attributing residual errors to model quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def report_markdown(diagnostic: Mapping[str, Any]) -> str:
    metrics = diagnostic["metrics"]
    taxonomy = diagnostic["failure_taxonomy_counts"]
    distributions = diagnostic["answer_distributions"]
    priorities = diagnostic["repair_priorities"]
    lines = [
        "# RSVQA-LR Smoke Diagnostic Report",
        "",
        "This report analyzes the persisted 50-question smoke run only. It does not change production behavior and does not claim a full RSVQA benchmark result.",
        "",
        "## Aggregate Verification",
        "",
        f"All supplied aggregate checks passed: **{diagnostic['verification']['all_passed']}**.",
        "",
        f"- Samples: {metrics['samples']}",
        f"- Answered: {metrics['answered_count']}",
        f"- Null: {metrics['null_count']}",
        f"- Coverage: {_fmt_percent(metrics['coverage_percent'])}",
        f"- All-sample accuracy: {_fmt_percent(metrics['overall_accuracy_percent'])}",
        f"- Answered-only accuracy: {_fmt_percent(metrics['answered_only_accuracy_percent'])}",
        "",
        "## Metrics by Type",
        "",
        "| Type | Samples | Coverage | Null rate | Overall accuracy | Answered-only accuracy | Majority baseline |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for question_type, values in metrics["by_type"].items():
        baseline = values["majority_baseline"]
        lines.append(
            f"| {question_type} | {values['samples']} | {_fmt_percent(values['coverage_percent'])} | "
            f"{_fmt_percent(values['null_rate_percent'])} | {_fmt_percent(values['overall_accuracy_percent'])} | "
            f"{_fmt_percent(values['answered_only_accuracy_percent'])} | {baseline['label']} ({_fmt_percent(baseline['accuracy_percent'])}) |"
        )
    lines.extend(
        [
            "",
            "No question type exceeds its majority-class baseline in this smoke sample. Count accuracy equals its zero-label baseline; it does not exceed it.",
            "",
            "## Confidence and Latency",
            "",
            "| Outcome | Samples | Average confidence | Average latency (ms) | Median (ms) | P95 (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for outcome in ("correct", "wrong", "null"):
        confidence = metrics["confidence_by_outcome"][outcome]
        latency = metrics["latency_by_outcome"][outcome]
        lines.append(
            f"| {outcome} | {latency['samples']} | {confidence['average_confidence']} | "
            f"{latency['average_latency_ms']} | {latency['median_latency_ms']} | {latency['p95_latency_ms']} |"
        )
    lines.extend(["", "### Latency by task", "", "| Task | Samples | Average (ms) | Median (ms) | P95 (ms) |", "|---|---:|---:|---:|---:|"])
    for task, values in metrics["latency_by_task"].items():
        lines.append(f"| {task} | {values['samples']} | {values['average_latency_ms']} | {values['median_latency_ms']} | {values['p95_latency_ms']} |")
    lines.extend(["", "## Failure Taxonomy", "", "| Primary cause | Samples |", "|---|---:|"])
    for category, count in sorted(taxonomy.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "Causal labels are conservative. In particular, raw comparison operand counts, caption scores, and SVE consistency objects were not persisted, so absent evidence is classified as `unknown` rather than inferred.",
            "",
            "## Wording and Target Findings",
            "",
        ]
    )
    for item in diagnostic["metrics_by_wording"]:
        lines.append(
            f"- `{item['question_type']} / {item['wording_pattern']}`: n={item['samples']}, coverage={_fmt_percent(item['coverage_percent'])}, "
            f"accuracy={_fmt_percent(item['overall_accuracy_percent'])}, null={_fmt_percent(item['null_rate_percent'])}."
        )
    lines.extend(["", "Target-level metrics are in `metrics_by_target.csv`; comparison rows contribute to both entity targets.", ""])
    lines.extend(["## Answer Distributions", ""])
    for question_type, values in distributions.items():
        baseline = values["majority_class_baseline"]
        lines.append(
            f"- **{question_type}:** ground truth {values['ground_truth_distribution']}; predictions {values['predicted_distribution']}; "
            f"null={values['null_count']}; most frequent prediction={values['most_frequent_prediction']}; "
            f"majority baseline={baseline['label']} at {_fmt_percent(baseline['accuracy_percent'])}."
        )
    count_buckets = diagnostic["count_answer_buckets"]
    lines.extend(["", "### Count label buckets", ""])
    for bucket in ("ground truth 0", "ground truth 1", "ground truth 2+"):
        values = count_buckets[bucket]
        lines.append(
            f"- {bucket}: n={values['samples']}, coverage={_fmt_percent(values['coverage_percent'])}, "
            f"overall accuracy={_fmt_percent(values['overall_accuracy_percent'])}."
        )
    dominance = count_buckets["zero_prediction_dominance"]
    lines.append(
        f"- Zero prediction dominance: {dominance['zero_predictions']}/{dominance['answered_count_samples']} answered count questions "
        f"({_fmt_percent(dominance['zero_prediction_rate_percent'])}) predicted `0`."
    )
    lines.extend(["", "## Priority Repairs", ""])
    for item in priorities:
        lines.append(
            f"{item['rank']}. **{item['issue']}** — affected={item['affected_samples']}, recoverability={item['expected_recoverability']:.2f}, impact={item['impact_score']:.2f}."
        )
    projection_data = diagnostic["full_run_projection"]
    lines.extend(
        [
            "",
            "## Full 10,004-Question Projection",
            "",
            "**Projection, not a measured benchmark.** It assumes smoke-run rates and sequential latency remain unchanged.",
            "",
            f"- Expected correct: {projection_data['expected_correct_answers']:.2f} (~{projection_data['expected_correct_answers_rounded']})",
            f"- Expected null: {projection_data['expected_null_answers']:.2f} (~{projection_data['expected_null_answers_rounded']})",
            f"- Mean-latency runtime: {projection_data['mean_latency_runtime']['hours']:.3f} hours",
            f"- P95-rate runtime scenario: {projection_data['p95_latency_runtime']['hours']:.3f} hours",
            "",
            "## Decision",
            "",
            "**B. Fix routing/evidence first.** The current 30% null rate and recoverable routing/vocabulary/operand gaps make a full run premature. Reassess model training or replacement after those deterministic gaps are removed on a held-out smoke set.",
            "",
        ]
    )
    return "\n".join(lines)


def build_diagnostic(
    rows: List[Dict[str, Any]],
    results: Mapping[str, Any],
    errors: Sequence[Any],
    confusion_overall: Path,
    confusion_by_type: Path,
) -> Dict[str, Any]:
    add_diagnostic_fields(rows)
    metrics = compute_core_metrics(rows)
    failures = [row for row in rows if not row["correct"]]
    wording = metrics_by_wording(rows)
    targets = metrics_by_target(rows)
    distributions = answer_distributions(rows)
    priorities = repair_priorities(rows)
    return {
        "analysis": "RSVQA-LR SatQuery smoke diagnostic",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "production_behavior_changed": False,
        "verification": verify_inputs(rows, results, errors, confusion_overall, confusion_by_type),
        "metrics": metrics,
        "failure_taxonomy_counts": dict(Counter(row["failure_category"] for row in failures)),
        "failure_taxonomy_domain": list(FAILURE_CATEGORIES),
        "metrics_by_wording": wording,
        "metrics_by_target": targets,
        "answer_distributions": distributions,
        "count_answer_buckets": count_bucket_metrics(rows),
        "repair_priorities": priorities,
        "full_run_projection": projection(metrics),
        "recommended_decision": {
            "choice": "B",
            "label": "fix routing/evidence first",
            "reason": "Recoverable routing, vocabulary, and comparison-operand gaps contribute materially to 30% null coverage loss.",
        },
    }


def write_outputs(rows: Sequence[Mapping[str, Any]], diagnostic: Mapping[str, Any], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "diagnostic_summary.json",
        "report": output_dir / "diagnostic_report.md",
        "taxonomy": output_dir / "failure_taxonomy.csv",
        "null": output_dir / "null_answers.csv",
        "wrong": output_dir / "wrong_answers.csv",
        "correct": output_dir / "correct_answers.csv",
        "type": output_dir / "metrics_by_type.csv",
        "wording": output_dir / "metrics_by_wording.csv",
        "target": output_dir / "metrics_by_target.csv",
        "distribution": output_dir / "answer_distribution.csv",
        "repairs": output_dir / "repair_priorities.md",
    }
    _write_json(paths["summary"], diagnostic)
    paths["report"].write_text(report_markdown(diagnostic), encoding="utf-8")
    paths["repairs"].write_text(repair_markdown(diagnostic["repair_priorities"]), encoding="utf-8")
    answer_fields = (
        "manifest_index", "question_id", "image_id", "question_type", "question", "normalized_ground_truth",
        "normalized_prediction", "answer", "outcome", "confidence", "task", "status", "result_status",
        "latency_ms", "wording_pattern", "target_1", "target_2", "target_details", "relation",
        "failure_category", "failure_evidence",
    )
    failures = [row for row in rows if not row["correct"]]
    _write_csv(paths["taxonomy"], answer_fields, failures)
    _write_csv(paths["null"], answer_fields, [row for row in rows if is_null_prediction(row)])
    _write_csv(paths["wrong"], answer_fields, [row for row in rows if not row["correct"] and not is_null_prediction(row)])
    _write_csv(paths["correct"], answer_fields, [row for row in rows if row["correct"]])
    type_rows = [
        _metric_csv_row({"question_type": question_type}, values)
        for question_type, values in diagnostic["metrics"]["by_type"].items()
    ]
    _write_csv(paths["type"], ("question_type", *METRIC_FIELDS), type_rows)
    wording_rows = [
        _metric_csv_row({"question_type": item["question_type"], "wording_pattern": item["wording_pattern"]}, item)
        for item in diagnostic["metrics_by_wording"]
    ]
    _write_csv(paths["wording"], ("question_type", "wording_pattern", *METRIC_FIELDS), wording_rows)
    target_rows = [
        _metric_csv_row({"target": item["target"], "target_details": item["target_details"]}, item)
        for item in diagnostic["metrics_by_target"]
    ]
    _write_csv(paths["target"], ("target", "target_details", *METRIC_FIELDS), target_rows)
    write_answer_distribution(paths["distribution"], diagnostic["answer_distributions"])
    return list(paths.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_ROOT / "predictions.csv")
    parser.add_argument("--results", type=Path, default=DEFAULT_ROOT / "results.json")
    parser.add_argument("--errors", type=Path, default=DEFAULT_ROOT / "errors.json")
    parser.add_argument("--confusion-overall", type=Path, default=DEFAULT_ROOT / "confusion_matrix_overall.csv")
    parser.add_argument("--confusion-by-type", type=Path, default=DEFAULT_ROOT / "confusion_matrix_by_type.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_predictions(args.predictions)
        results = _load_json(args.results)
        errors = _load_json(args.errors)
        if not isinstance(errors, list):
            raise ValueError("errors.json must contain a JSON array.")
        diagnostic = build_diagnostic(rows, results, errors, args.confusion_overall, args.confusion_by_type)
        generated = write_outputs(rows, diagnostic, args.output_dir)
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as error:
        print(f"Diagnostic analysis failed: {error}", file=sys.stderr)
        return 2

    metrics = diagnostic["metrics"]
    taxonomy = sorted(diagnostic["failure_taxonomy_counts"].items(), key=lambda item: (-item[1], item[0]))
    priorities = diagnostic["repair_priorities"]
    print("RSVQA smoke diagnostic")
    print(f"All-sample accuracy: {metrics['overall_accuracy_percent']:.2f}%")
    print(f"Answered-only accuracy: {metrics['answered_only_accuracy_percent']:.2f}%")
    print(f"Coverage: {metrics['coverage_percent']:.2f}%")
    print("Null rate by type: " + ", ".join(f"{name}={values['null_rate_percent']:.2f}%" for name, values in metrics["by_type"].items()))
    print("Majority baseline by type: " + ", ".join(f"{name}={values['majority_baseline']['accuracy_percent']:.2f}%" for name, values in metrics["by_type"].items()))
    print("Top failure causes: " + "; ".join(f"{name} ({count})" for name, count in taxonomy[:3]))
    print("Top repair priorities: " + "; ".join(f"{item['rank']}. {item['issue']}" for item in priorities[:3]))
    print("Decision: B. fix routing/evidence first")
    print(f"Generated {len(generated)} files in {args.output_dir}")
    return 0 if diagnostic["verification"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
