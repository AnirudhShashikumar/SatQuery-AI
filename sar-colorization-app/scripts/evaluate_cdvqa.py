#!/usr/bin/env python3
"""Evaluate SatQuery's production bi-temporal QA path on a labelled CDVQA split."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from evaluation_common import benchmark_record, environment_record, summarize_latencies, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import benchmark_record, environment_record, summarize_latencies, write_csv, write_json


def normalize_answer(value: Any) -> str:
    """Minimal exact-answer normalization; no synonym or ontology rewriting."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".?! ")


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("questions", "data", "annotations", "samples"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise ValueError("CDVQA annotation JSON must contain a list or questions/data/annotations/samples list")


def discover_cdvqa(root: Path, split: str) -> list[dict[str, Any]]:
    candidates = [
        root / f"{split}.json", root / "annotations" / f"{split}.json",
        root / "questions" / f"{split}.json", root / f"cdvqa_{split}.json",
    ]
    annotation = next((path for path in candidates if path.is_file()), None)
    if annotation is None:
        raise FileNotFoundError(f"CDVQA {split} annotations were not found; checked: {', '.join(str(path) for path in candidates)}")
    result = []
    for index, raw in enumerate(_records(json.loads(annotation.read_text(encoding="utf-8")))):
        question = raw.get("question") or raw.get("query")
        answer = raw.get("answer") if "answer" in raw else raw.get("label")
        before_name = raw.get("before_image") or raw.get("image1") or raw.get("image_before") or raw.get("A")
        after_name = raw.get("after_image") or raw.get("image2") or raw.get("image_after") or raw.get("B")
        if not all(value is not None for value in (question, answer, before_name, after_name)):
            raise ValueError(f"CDVQA record {index} is missing question, answer, before image, or after image")
        def resolve(name: Any, folders: tuple[str, ...]) -> Path:
            value = Path(str(name))
            choices = [root / value, *(root / folder / value for folder in folders)]
            path = next((choice for choice in choices if choice.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"CDVQA image is missing for record {index}: {name}")
            return path
        result.append({
            "sample_id": str(raw.get("question_id") or raw.get("id") or index),
            "question": str(question), "answer": str(answer),
            "category": str(raw.get("category") or raw.get("question_type") or "unknown"),
            "before": resolve(before_name, ("images", "A", split + "/A")),
            "after": resolve(after_name, ("images", "B", split + "/B")),
            "before_date": str(raw.get("before_date") or "2000-01-01"),
            "after_date": str(raw.get("after_date") or "2001-01-01"),
        })
    if not result:
        raise RuntimeError(f"CDVQA {split} split contains no labelled questions")
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CDVQA dataset is unavailable: {root}")
    records = discover_cdvqa(root, args.split)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    rows, failures, latencies = [], [], []
    for record in records:
        started = time.perf_counter()
        try:
            response = client.post(
                "/api/agent/query",
                data={
                    "query": record["question"], "input_mode": "bi_temporal",
                    "primary_modality": "optical", "secondary_modality": "optical",
                    "primary_date": record["before_date"], "secondary_date": record["after_date"],
                    "use_cache": "false",
                },
                files={
                    "primary_image": (record["before"].name, record["before"].read_bytes()),
                    "secondary_image": (record["after"].name, record["after"].read_bytes()),
                },
            )
            response.raise_for_status()
            payload = response.json()
            predicted = payload.get("answer")
            correct = normalize_answer(predicted) == normalize_answer(record["answer"])
            latency = (time.perf_counter() - started) * 1000.0
            latencies.append(latency)
            rows.append({
                "sample_id": record["sample_id"], "category": record["category"],
                "question": record["question"], "reference_answer": record["answer"],
                "prediction": predicted, "normalized_reference": normalize_answer(record["answer"]),
                "normalized_prediction": normalize_answer(predicted), "correct": correct,
                "status": payload.get("status"), "fallback_used": bool((payload.get("change_engine") or {}).get("fallback_used")),
                "confidence_level": (payload.get("confidence") or {}).get("level"),
                "latency_ms": latency,
            })
        except Exception as error:
            failures.append({"sample_id": record["sample_id"], "question": record["question"], "error": f"{type(error).__name__}: {error}"[:500]})
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "per_question_results.csv", rows)
    write_csv(output / "failures.csv", failures, ("sample_id", "question", "error"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    category_rows = [
        {"category": category, "samples": len(items), "exact_accuracy": sum(bool(item["correct"]) for item in items) / len(items), "failures": 0}
        for category, items in sorted(grouped.items())
    ]
    write_csv(output / "per_category_metrics.csv", category_rows)
    complete = len(rows) == len(records) and not failures
    exact = sum(bool(row["correct"]) for row in rows) / len(records) if complete else None
    summary = {
        "dataset": "CDVQA", "split": args.split, "expected_samples": len(records), "completed_samples": len(rows),
        "failures": len(failures), "exact_accuracy": exact,
        "normalization": "Unicode NFKC, lowercase, whitespace collapse, terminal punctuation trim only",
        "performance": summarize_latencies(latencies),
        "status": "verified_test" if complete and args.split == "test" else "verified_validation" if complete else "unavailable",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment_record())
    metric_rows = [{"name": "exact_accuracy", "value": exact, "unit": "ratio", "primary": True, "definition": summary["normalization"]}]
    record = benchmark_record(
        benchmark_id=f"cdvqa.production.{args.split}", specialist_id="bitemporal_change_analyzer",
        display_name="SatQuery CDVQA evaluation", task="change_vqa", model_name="SatQuery bi-temporal change stack",
        model_version="semantic-interpreter-1.0", checkpoint=None, checkpoint_sha256=None,
        dataset="CDVQA", split=args.split, sample_count=len(records) if complete else None,
        status=summary["status"], metrics=metric_rows, performance=summary["performance"],
        artifacts=["predictions.csv", "per_question_results.csv", "per_category_metrics.csv", "summary.json", "environment.json", "report.md"],
        limitations=["Official CDVQA release variants use different annotation layouts; the selected adapter and split must be reviewed.", "Only minimal exact-answer normalization is applied."],
    )
    write_json(output / "benchmark_record.json", record)
    (output / "report.md").write_text(
        "# CDVQA evaluation\n\n"
        f"Status: **{summary['status']}**  \nExpected/completed: **{len(records)}/{len(rows)}**  \n"
        f"Exact accuracy: **{exact if exact is not None else 'unavailable'}**\n\n"
        "No result is considered verified when a discovered sample fails. Dataset release identity and official split must be confirmed by the evaluator operator.\n",
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError("CDVQA evaluation was incomplete; artifacts record failures and no accuracy claim was emitted")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto", help="Recorded run preference; production engine configuration remains authoritative.")
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/cdvqa_test"))
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
