#!/usr/bin/env python3
"""Compare two persisted RSVQA smoke runs without changing either run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


QUESTION_TYPES = ("comp", "count", "presence", "rural_urban")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No predictions found in {path}")
    ids = [row.get("question_id", "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Predictions contain empty or duplicate question IDs: {path}")
    return rows


def metrics(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    selected = list(rows)
    total = len(selected)
    answered = [row for row in selected if row.get("normalized_prediction", "").strip()]
    correct = [row for row in selected if _truthy(row.get("correct"))]
    endpoint_errors = [row for row in selected if _truthy(row.get("endpoint_error"))]
    latency = []
    for row in selected:
        try:
            value = float(row.get("effective_latency_ms", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            latency.append(value)
    ordered_latency = sorted(latency)
    if ordered_latency:
        position = (len(ordered_latency) - 1) * 0.95
        lower = math.floor(position)
        upper = math.ceil(position)
        p95 = ordered_latency[lower] + (ordered_latency[upper] - ordered_latency[lower]) * (position - lower)
    else:
        p95 = None
    return {
        "samples": total,
        "correct": len(correct),
        "answered": len(answered),
        "null_answers": total - len(answered),
        "endpoint_errors": len(endpoint_errors),
        "accuracy": len(correct) / total if total else 0.0,
        "coverage": len(answered) / total if total else 0.0,
        "average_latency_ms": statistics.fmean(latency) if latency else None,
        "median_latency_ms": statistics.median(latency) if latency else None,
        "p95_latency_ms": p95,
    }


def compare(v1_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v1 = {row["question_id"]: row for row in v1_rows}
    v2 = {row["question_id"]: row for row in v2_rows}
    if set(v1) != set(v2):
        missing_v2 = sorted(set(v1) - set(v2))
        missing_v1 = sorted(set(v2) - set(v1))
        raise ValueError(f"Question sets differ (missing_v2={missing_v2}, missing_v1={missing_v1})")

    changes: list[dict[str, Any]] = []
    for question_id in v1:
        before, after = v1[question_id], v2[question_id]
        before_correct, after_correct = _truthy(before.get("correct")), _truthy(after.get("correct"))
        before_answered = bool(before.get("normalized_prediction", "").strip())
        after_answered = bool(after.get("normalized_prediction", "").strip())
        if not before_correct and after_correct:
            change = "fixed"
        elif before_correct and not after_correct:
            change = "regression"
        elif before_correct:
            change = "unchanged_correct"
        elif not before_answered and after_answered:
            change = "newly_answered_wrong"
        elif before_answered and not after_answered:
            change = "newly_null"
        else:
            change = "unchanged_wrong_or_null"
        changes.append({
            "question_id": question_id,
            "image_id": before.get("image_id"),
            "question_type": before.get("question_type"),
            "question": before.get("question"),
            "ground_truth": before.get("normalized_ground_truth"),
            "v1_prediction": before.get("normalized_prediction") or None,
            "v2_prediction": after.get("normalized_prediction") or None,
            "v1_correct": before_correct,
            "v2_correct": after_correct,
            "v1_status": before.get("status"),
            "v2_status": after.get("status"),
            "change": change,
        })

    v1_metrics, v2_metrics = metrics(v1_rows), metrics(v2_rows)
    per_type: dict[str, Any] = {}
    majority: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        before = [row for row in v1_rows if row.get("question_type") == question_type]
        after = [row for row in v2_rows if row.get("question_type") == question_type]
        before_metrics, after_metrics = metrics(before), metrics(after)
        labels = Counter(row.get("normalized_ground_truth", "") for row in before)
        majority_label, majority_count = labels.most_common(1)[0]
        baseline = majority_count / len(before)
        per_type[question_type] = {
            "v1": before_metrics,
            "v2": after_metrics,
            "accuracy_change": after_metrics["accuracy"] - before_metrics["accuracy"],
            "coverage_change": after_metrics["coverage"] - before_metrics["coverage"],
        }
        majority[question_type] = {
            "label": majority_label,
            "accuracy": baseline,
            "v1_exceeds": before_metrics["accuracy"] > baseline,
            "v2_exceeds": after_metrics["accuracy"] > baseline,
        }

    count_distribution = {
        "v1": dict(sorted(Counter(row.get("normalized_prediction") or "null" for row in v1_rows if row.get("question_type") == "count").items())),
        "v2": dict(sorted(Counter(row.get("normalized_prediction") or "null" for row in v2_rows if row.get("question_type") == "count").items())),
    }
    change_counts = Counter(row["change"] for row in changes)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v1": v1_metrics,
        "v2": v2_metrics,
        "overall_accuracy_change": v2_metrics["accuracy"] - v1_metrics["accuracy"],
        "coverage_change": v2_metrics["coverage"] - v1_metrics["coverage"],
        "null_answer_change": v2_metrics["null_answers"] - v1_metrics["null_answers"],
        "fixed_questions": change_counts["fixed"],
        "regressions": change_counts["regression"],
        "per_type": per_type,
        "majority_baselines": majority,
        "count_prediction_distribution": count_distribution,
    }
    return summary, changes


def _write_comparison_confusions(
    output_dir: Path,
    v1_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
) -> tuple[Path, Path]:
    runs = {"Old heuristic": v1_rows, "RSVQA Specialist v1": v2_rows}
    labels = sorted({
        row.get(key) or "<null>"
        for rows in runs.values()
        for row in rows
        for key in ("normalized_ground_truth", "normalized_prediction")
    })
    overall_path = output_dir / "confusion_matrices_overall.csv"
    task_path = output_dir / "confusion_matrices_by_task.csv"
    with overall_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "ground_truth", "total", *labels])
        writer.writeheader()
        for model, rows in runs.items():
            counts: dict[str, Counter] = defaultdict(Counter)
            for row in rows:
                counts[row.get("normalized_ground_truth") or "<null>"][row.get("normalized_prediction") or "<null>"] += 1
            for actual in sorted(counts):
                writer.writerow({"model": model, "ground_truth": actual, "total": sum(counts[actual].values()), **{label: counts[actual][label] for label in labels}})
    with task_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "question_type", "ground_truth", "total", *labels])
        writer.writeheader()
        for model, rows in runs.items():
            counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
            for row in rows:
                key = (row.get("question_type") or "unknown", row.get("normalized_ground_truth") or "<null>")
                counts[key][row.get("normalized_prediction") or "<null>"] += 1
            for question_type, actual in sorted(counts):
                values = counts[(question_type, actual)]
                writer.writerow({"model": model, "question_type": question_type, "ground_truth": actual, "total": sum(values.values()), **{label: values[label] for label in labels}})
    return overall_path, task_path


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    changes: list[dict[str, Any]],
    v1_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v1_vs_v2.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(changes[0])
    with (output_dir / "per_question_changes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)
    _write_comparison_confusions(output_dir, v1_rows, v2_rows)
    lines = [
        "# RSVQA Smoke Benchmark: Old Heuristic vs RSVQA Specialist v1",
        "",
        "| Metric | Old heuristic | RSVQA Specialist v1 | Change |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {summary['v1']['accuracy']:.2%} | {summary['v2']['accuracy']:.2%} | {summary['overall_accuracy_change']:+.2%} |",
        f"| Coverage | {summary['v1']['coverage']:.2%} | {summary['v2']['coverage']:.2%} | {summary['coverage_change']:+.2%} |",
        f"| Null answers | {summary['v1']['null_answers']} | {summary['v2']['null_answers']} | {summary['null_answer_change']:+d} |",
        f"| Endpoint errors | {summary['v1']['endpoint_errors']} | {summary['v2']['endpoint_errors']} | {summary['v2']['endpoint_errors'] - summary['v1']['endpoint_errors']:+d} |",
        "",
        f"Fixed questions: **{summary['fixed_questions']}**. Regressions: **{summary['regressions']}**.",
        "",
        "## Per-type results",
        "",
        "| Type | Heuristic accuracy | Specialist accuracy | Heuristic coverage | Specialist coverage | Majority baseline | Specialist exceeds |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for question_type in QUESTION_TYPES:
        item = summary["per_type"][question_type]
        baseline = summary["majority_baselines"][question_type]
        lines.append(
            f"| {question_type} | {item['v1']['accuracy']:.2%} | {item['v2']['accuracy']:.2%} | "
            f"{item['v1']['coverage']:.2%} | {item['v2']['coverage']:.2%} | "
            f"{baseline['label']} ({baseline['accuracy']:.2%}) | {'yes' if baseline['v2_exceeds'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Latency",
        "",
        "| Metric | Old heuristic | RSVQA Specialist v1 |",
        "|---|---:|---:|",
        f"| Average | {summary['v1']['average_latency_ms']:.1f} ms | {summary['v2']['average_latency_ms']:.1f} ms |",
        f"| Median | {summary['v1']['median_latency_ms']:.1f} ms | {summary['v2']['median_latency_ms']:.1f} ms |",
        f"| P95 | {summary['v1']['p95_latency_ms']:.1f} ms | {summary['v2']['p95_latency_ms']:.1f} ms |",
        "",
        "## Count prediction distribution",
        "",
        f"- Old heuristic: `{summary['count_prediction_distribution']['v1']}`",
        f"- RSVQA Specialist v1: `{summary['count_prediction_distribution']['v2']}`",
        "",
        "The specialist's `201+` class is the exported overflow label. Counts are learned benchmark labels, not calibrated physical object totals.",
        "",
        "## Confusion matrices",
        "",
        "Combined old/new confusion matrices are in `confusion_matrices_overall.csv` and `confusion_matrices_by_task.csv`.",
    ])
    (output_dir / "v1_vs_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("artifacts/rsvqa_smoke/predictions.csv"))
    parser.add_argument("--v2", type=Path, default=Path("artifacts/rsvqa_smoke_v2/predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rsvqa_smoke_comparison"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v1_rows, v2_rows = load_predictions(args.v1), load_predictions(args.v2)
    summary, changes = compare(v1_rows, v2_rows)
    write_outputs(args.output_dir, summary, changes, v1_rows, v2_rows)
    print(
        f"v1 {summary['v1']['accuracy']:.2%} -> v2 {summary['v2']['accuracy']:.2%}; "
        f"coverage {summary['v1']['coverage']:.2%} -> {summary['v2']['coverage']:.2%}; "
        f"fixed={summary['fixed_questions']} regressions={summary['regressions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
