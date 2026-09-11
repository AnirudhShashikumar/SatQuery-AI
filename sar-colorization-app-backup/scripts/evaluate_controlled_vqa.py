#!/usr/bin/env python3
"""Controlled GeoVision VQA validation over a local manifest.

This is not an RSVQA benchmark. Each manifest item must contain ``image_path``
and ``question``. Optional checks are ``expected_normalized_answer`` or
``expected_range`` with ``field``, ``minimum``, and ``maximum``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import app  # noqa: E402


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.%]+", " ", value.lower())).strip()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("The JSON manifest must be an array of samples.")
    return payload


def nested_value(data: Dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        value = value[part]
    return value


def evaluate(samples: Iterable[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    client = TestClient(app)
    predictions: List[Dict[str, Any]] = []
    categories: Dict[str, Dict[str, int]] = defaultdict(lambda: {"samples": 0, "supported": 0, "correct": 0})
    exact_total = exact_correct = numeric_total = numeric_correct = 0
    runtime_total = 0.0

    for index, sample in enumerate(samples):
        image_path = Path(sample["image_path"]).expanduser().resolve()
        question = str(sample["question"])
        modality = str(sample.get("modality", "optical"))
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        response = client.post(
            "/api/agent/query",
            data={"query": question, "input_mode": "single", "primary_modality": modality},
            files={"primary_image": (image_path.name, image_path.read_bytes(), mime)},
        )
        body = response.json()
        details = body.get("vqa_details") or {}
        category = details.get("question_category", "unsupported")
        supported = bool(details.get("supported", False))
        runtime = float(body.get("execution", {}).get("duration_ms", 0))
        runtime_total += runtime
        categories[category]["samples"] += 1
        categories[category]["supported"] += int(supported)
        correct: Any = None

        expected = sample.get("expected_normalized_answer")
        if expected is not None:
            exact_total += 1
            correct = normalize_answer(str(body.get("answer") or "")) == normalize_answer(str(expected))
            exact_correct += int(correct)
        expected_range = sample.get("expected_range")
        if expected_range is not None:
            numeric_total += 1
            field = str(expected_range["field"])
            value = float(nested_value(body, field))
            correct = float(expected_range["minimum"]) <= value <= float(expected_range["maximum"])
            numeric_correct += int(correct)
        if correct:
            categories[category]["correct"] += 1
        predictions.append({
            "sample_index": index,
            "image_path": str(image_path),
            "question": question,
            "status_code": response.status_code,
            "status": body.get("status"),
            "supported": supported,
            "question_category": category,
            "answer": body.get("answer"),
            "statistics_used": details.get("statistics_used", {}),
            "runtime_ms": runtime,
            "correct": correct,
        })

    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    total = len(predictions)
    return {
        "validation_name": "Controlled GeoVision VQA validation",
        "total_samples": total,
        "supported_samples": sum(int(item["supported"]) for item in predictions),
        "unsupported_samples": sum(not item["supported"] for item in predictions),
        "normalized_exact_match_accuracy": exact_correct / exact_total if exact_total else None,
        "numeric_tolerance_accuracy": numeric_correct / numeric_total if numeric_total else None,
        "per_category_results": dict(categories),
        "average_runtime_ms": runtime_total / total if total else 0.0,
        "predictions_jsonl": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled GeoVision VQA validation (not RSVQA benchmark performance).")
    parser.add_argument("manifest", type=Path, help="JSON array or JSONL validation manifest")
    parser.add_argument("--output", type=Path, default=Path("controlled_vqa_predictions.jsonl"))
    args = parser.parse_args()
    try:
        summary = evaluate(load_manifest(args.manifest), args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
