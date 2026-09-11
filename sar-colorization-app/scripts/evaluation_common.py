"""Small dependency-light utilities for reproducible SatQuery evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Direct ``python scripts/<tool>.py`` execution places only ``scripts`` on the
# import path.  Make the repository root available so every shipped CLI can
# import the same production modules that the application uses.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        if names:
            writer.writeheader()
            writer.writerows(rows)


def environment_record() -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_version, mps, cuda = None, False, False
    return {
        "timestamp": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch_version,
        "mps_available": mps,
        "cuda_available": cuda,
        "processor": platform.processor() or None,
    }


def binary_confusion(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} versus {truth.shape}")
    return {
        "tp": int(np.count_nonzero(pred & truth)),
        "fp": int(np.count_nonzero(pred & ~truth)),
        "fn": int(np.count_nonzero(~pred & truth)),
        "tn": int(np.count_nonzero(~pred & ~truth)),
    }


def binary_metrics(confusion: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (confusion[key] for key in ("tp", "fp", "fn", "tn"))
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "overall_accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "specificity": ratio(tn, tn + fp),
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    items = list(values)
    return float(np.percentile(items, q)) if items else None


def summarize_latencies(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(values, 95),
    }


def benchmark_record(
    *,
    benchmark_id: str,
    specialist_id: str,
    display_name: str,
    task: str,
    model_name: str,
    model_version: str,
    checkpoint: str | None,
    checkpoint_sha256: str | None,
    dataset: str,
    split: str,
    sample_count: int | None,
    status: str,
    metrics: Sequence[dict[str, Any]],
    performance: dict[str, Any],
    artifacts: Sequence[str],
    limitations: Sequence[str],
) -> dict[str, Any]:
    environment = environment_record()
    normalized_metrics = []
    for metric in metrics:
        metric_id = str(metric.get("id") or metric.get("name") or "metric").lower().replace(" ", "_")
        description = str(metric.get("description") or metric.get("definition") or "Metric computed by the repository evaluator.")
        normalized_metrics.append({
            "id": metric_id,
            "label": str(metric.get("label") or metric_id.replace("_", " ").title()),
            "value": metric.get("value"),
            "unit": metric.get("unit", "score"),
            "higher_is_better": metric.get("higher_is_better", not any(term in metric_id for term in ("loss", "error", "mae", "latency"))),
            "description": description,
            "primary": bool(metric.get("primary", False)),
        })
    artifact_values = list(artifacts)
    def artifact(suffixes: tuple[str, ...]) -> str | None:
        return next((value for value in artifact_values if value.endswith(suffixes)), None)
    mean_latency = performance.get("mean_latency_ms", performance.get("mean_ms"))
    median_latency = performance.get("median_latency_ms", performance.get("median_ms"))
    p95_latency = performance.get("p95_latency_ms", performance.get("p95_ms"))
    throughput = performance.get("throughput_samples_per_second")
    if throughput is None and mean_latency:
        throughput = 1000.0 / float(mean_latency)
    return {
        "schema_version": "1.0.0",
        "benchmark_id": benchmark_id,
        "specialist_id": specialist_id,
        "display_name": display_name,
        "task": task,
        "model": {
            "name": model_name,
            "version": model_version,
            "architecture": model_name,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "training_dataset": dataset,
            "adaptation": None,
            "license": None,
            "source": "Repository production implementation",
        },
        "evaluation": {
            "status": status,
            "dataset": dataset,
            "split": split,
            "sample_count": sample_count,
            "timestamp": utc_now(),
            "protocol": "repository evaluator; dataset identity and completeness must be confirmed",
            "environment": {
                "device": "cuda" if environment["cuda_available"] else "mps" if environment["mps_available"] else "cpu",
                "hardware": environment["machine"],
                "os": environment["platform"],
                "python": environment["python"],
                "torch": environment["torch"],
                "input_size": None,
            },
        },
        "metrics": normalized_metrics,
        "performance": {
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "throughput_samples_per_second": throughput,
            "peak_memory_mb": performance.get("peak_memory_mb"),
            "model_load_ms": performance.get("model_load_ms", performance.get("cold_load_ms")),
            "warm_reuse": performance.get("warm_reuse"),
        },
        "artifacts": {
            "report": artifact((".md",)),
            "predictions": artifact(("predictions.csv", "per_question_results.csv")),
            "per_class_metrics": artifact(("per_class_metrics.csv", "per_category_metrics.csv", "per_image_metrics.csv")),
            "confusion_matrix": artifact(("confusion_matrix.csv",)),
            "visual_examples": [value for value in artifact_values if "qualitative" in value or "visual" in value],
        },
        "limitations": list(limitations),
        "notes": ["No missing value is represented as zero.", "Smoke or parity execution is not promoted to accuracy evidence."],
        "provenance": {
            "result_origin": "satquery_reproduced" if status.startswith("verified_") else "unmeasured" if status == "unavailable" else "satquery_reproduced",
            "source_artifacts": artifact_values,
            "generated_by": "repository evaluation script",
            "verified": status.startswith("verified_"),
        },
    }


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}")
    return resolved
