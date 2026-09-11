#!/usr/bin/env python3
"""Run a resumable RSVQA-LR smoke benchmark against SatQuery's image endpoint."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np
import rasterio
import requests
from PIL import Image


DEFAULT_MANIFEST = Path("datasets/RSVQA/rsvqa_lr_test_smoke50.jsonl")
DEFAULT_IMAGE_ROOT = Path("datasets/RSVQA/images/Images_LR")
DEFAULT_ENDPOINT = "http://127.0.0.1:8010/api/analysis/image"
DEFAULT_OUTPUT_DIR = Path("artifacts/rsvqa_smoke")
ALLOWED_QUESTION_TYPES = {"comp", "count", "presence", "rural_urban"}
PREDICTION_FIELDS = (
    "manifest_index",
    "question_id",
    "image_id",
    "source_image_path",
    "converted_image_path",
    "question",
    "question_type",
    "ground_truth",
    "normalized_ground_truth",
    "answer",
    "normalized_prediction",
    "correct",
    "outcome",
    "completed",
    "endpoint_error",
    "error_stage",
    "http_status",
    "confidence",
    "task",
    "status",
    "result_status",
    "processing_time_ms",
    "client_elapsed_ms",
    "effective_latency_ms",
    "model_used",
    "cached",
    "warnings_json",
    "reuse_observations_json",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    image_root: Path
    endpoint: str
    output_dir: Path
    timeout: float = 120.0
    resume: bool = False

    @property
    def converted_dir(self) -> Path:
        return self.output_dir / "converted"

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / "predictions.csv"


def normalize_answer(value: Any, question_type: Optional[str] = None) -> Optional[str]:
    """Normalize only values in the benchmark answer domain."""
    if value is None:
        return None
    if isinstance(value, bool):
        token = "yes" if value else "no"
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        token = str(int(value)) if int(value) >= 0 else ""
    elif isinstance(value, str):
        if str(question_type or "").strip().lower() == "count" and value.strip() == "201+":
            return "201+"
        token = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    else:
        return None

    expected_domain = {
        "comp": "yes_no",
        "presence": "yes_no",
        "rural_urban": "rural_urban",
        "count": "integer",
    }.get(str(question_type or "").strip().lower())

    if expected_domain in {None, "yes_no"}:
        if token in {"yes", "y", "true", "affirmative"}:
            return "yes"
        if token in {"no", "n", "false", "negative"}:
            return "no"
    if expected_domain in {None, "rural_urban"}:
        words = set(token.split())
        if "urban" in words and "rural" not in words:
            return "urban"
        if "rural" in words and "urban" not in words:
            return "rural"
    if expected_domain in {None, "integer"} and re.fullmatch(r"0|[1-9]\d*", token):
        number = int(token)
        return "201+" if expected_domain == "integer" and number > 200 else token
    return None


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load and validate every JSONL record before applying a CLI limit."""
    records: List[Dict[str, Any]] = []
    seen_question_ids = set()
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on manifest line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object.")
            missing = [key for key in ("question_id", "image_id", "question", "question_type", "ground_truth") if key not in record]
            if missing:
                raise ValueError(f"Manifest line {line_number} is missing: {', '.join(missing)}")
            question_id = str(record["question_id"])
            if question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id in manifest: {question_id}")
            seen_question_ids.add(question_id)
            question_type = str(record["question_type"]).strip().lower()
            if question_type not in ALLOWED_QUESTION_TYPES:
                raise ValueError(f"Unsupported question_type on line {line_number}: {question_type}")
            normalized_label = normalize_answer(record["ground_truth"], question_type)
            if normalized_label is None:
                raise ValueError(f"Invalid ground_truth on line {line_number}: {record['ground_truth']!r}")
            records.append(record)
    if not records:
        raise ValueError("The manifest contains no records.")
    return records


def resolve_image_path(record: Mapping[str, Any], image_root: Path) -> Path:
    """Ignore manifest image_path and resolve strictly from the local image ID."""
    image_id = str(record["image_id"]).strip()
    if not image_id or not re.fullmatch(r"[A-Za-z0-9._-]+", image_id):
        raise ValueError(f"Unsafe or empty image_id: {image_id!r}")
    return image_root.expanduser().resolve() / f"{image_id}.tif"


def _stretch_band(band: np.ma.MaskedArray) -> np.ndarray:
    values = np.asarray(np.ma.getdata(band), dtype=np.float64)
    valid = np.isfinite(values) & ~np.ma.getmaskarray(band)
    if not valid.any():
        raise ValueError("A selected TIFF band contains no finite pixels.")
    selected = values[valid]
    low, high = (float(item) for item in np.percentile(selected, [2.0, 98.0]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("A selected TIFF band has invalid percentile limits.")
    if high <= low:
        low, high = float(selected.min()), float(selected.max())
    if high <= low:
        output = np.full(values.shape, 127, dtype=np.uint8)
        output[~valid] = 0
        return output
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~valid] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def convert_tiff_to_png(source_path: Path, converted_dir: Path, image_id: Any) -> Tuple[Path, bool]:
    """Convert the first three bands to cached deterministic display RGB."""
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Local RSVQA image does not exist: {source_path}")
    safe_id = str(image_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", safe_id):
        raise ValueError(f"Unsafe or empty image_id: {safe_id!r}")
    converted_dir.mkdir(parents=True, exist_ok=True)
    output_path = converted_dir / f"{safe_id}.png"
    if output_path.is_file() and output_path.stat().st_size > 0 and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        return output_path, True

    with rasterio.open(source_path) as dataset:
        if dataset.count < 3:
            raise ValueError(f"RSVQA TIFF requires at least three bands: {source_path}")
        bands = dataset.read((1, 2, 3), masked=True)
    stretched = np.stack([_stretch_band(bands[index]) for index in range(3)], axis=-1)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(stretched).save(temporary, format="PNG")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path, False


def normalize_endpoint(value: str) -> str:
    value = value.strip()
    markdown = re.fullmatch(r"\[[^]]+\]\((https?://[^)]+)\)", value)
    if markdown:
        value = markdown.group(1)
    if not re.match(r"^https?://", value):
        raise ValueError("--endpoint must be an HTTP or HTTPS URL.")
    return value.rstrip("/")


def _json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> Optional[int]:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _reuse_observations(payload: Mapping[str, Any]) -> Dict[str, Any]:
    trace = payload.get("execution_trace")
    if not isinstance(trace, list):
        trace = []
    tools = [str(item.get("tool", "")) for item in trace if isinstance(item, dict)]
    image_cache_hits = 0
    image_cache_misses = 0
    grounding_model_reuses = 0
    for item in trace:
        if not isinstance(item, dict):
            continue
        parameters = item.get("parameters")
        if not isinstance(parameters, dict):
            continue
        if item.get("tool") == "sve_image_embedding":
            image_cache_hits += int(parameters.get("cache") == "hit")
            image_cache_misses += int(parameters.get("cache") == "miss")
        if item.get("tool") == "grounder_model_load":
            grounding_model_reuses += int(parameters.get("reused") is True)
    return {
        "response_cached": payload.get("cached") if "cached" in payload else None,
        "sve_model_load_count": tools.count("sve_model_load"),
        "sve_model_reuse_count": tools.count("sve_model_reuse"),
        "sve_image_cache_hits": image_cache_hits,
        "sve_image_cache_misses": image_cache_misses,
        "grounder_model_reuse_count": grounding_model_reuses,
        "grounding_used": any("ground" in tool for tool in tools),
    }


def _base_prediction(record: Mapping[str, Any], manifest_index: int, source_path: Path) -> Dict[str, Any]:
    question_type = str(record["question_type"]).strip().lower()
    return {
        "manifest_index": manifest_index,
        "question_id": str(record["question_id"]),
        "image_id": str(record["image_id"]),
        "source_image_path": str(source_path),
        "converted_image_path": None,
        "question": str(record["question"]),
        "question_type": question_type,
        "ground_truth": str(record["ground_truth"]),
        "normalized_ground_truth": normalize_answer(record["ground_truth"], question_type),
        "answer": None,
        "normalized_prediction": None,
        "correct": False,
        "outcome": "ERROR",
        "completed": False,
        "endpoint_error": None,
        "error_stage": None,
        "http_status": None,
        "confidence": None,
        "task": None,
        "status": None,
        "result_status": None,
        "processing_time_ms": None,
        "client_elapsed_ms": None,
        "effective_latency_ms": None,
        "model_used": None,
        "cached": None,
        "warnings": [],
        "reuse_observations": {},
        "conversion_cache_hit": None,
    }


def _error_prediction(base: Dict[str, Any], stage: str, message: str, **updates: Any) -> Dict[str, Any]:
    result = dict(base)
    result.update(updates)
    result.update({"endpoint_error": message, "error_stage": stage, "outcome": "ERROR", "completed": False})
    return result


def _response_error_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("code") or _json_string(detail))
        return str(detail)
    if payload not in (None, ""):
        return str(payload)
    return fallback


def execute_record(
    record: Mapping[str, Any],
    manifest_index: int,
    config: BenchmarkConfig,
    session: requests.Session,
) -> Dict[str, Any]:
    source_path = resolve_image_path(record, config.image_root)
    base = _base_prediction(record, manifest_index, source_path)
    try:
        png_path, conversion_cache_hit = convert_tiff_to_png(source_path, config.converted_dir, record["image_id"])
        base["converted_image_path"] = str(png_path)
        base["conversion_cache_hit"] = conversion_cache_hit
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        return _error_prediction(base, "image_conversion", str(error))

    request_started = time.perf_counter()
    try:
        with png_path.open("rb") as image_handle:
            response = session.post(
                config.endpoint,
                data={
                    "analysis_type": "ground_truth",
                    "model_name": "satquery-agent",
                    "question": str(record["question"]),
                },
                files={"image": (png_path.name, image_handle, "image/png")},
                timeout=config.timeout,
            )
    except requests.RequestException as error:
        elapsed = max(0, round((time.perf_counter() - request_started) * 1000))
        return _error_prediction(base, "endpoint_request", str(error), client_elapsed_ms=elapsed, effective_latency_ms=elapsed)

    elapsed = max(0, round((time.perf_counter() - request_started) * 1000))
    try:
        payload = response.json()
    except ValueError:
        return _error_prediction(
            base,
            "endpoint_response",
            "Endpoint returned a non-JSON response.",
            http_status=response.status_code,
            client_elapsed_ms=elapsed,
            effective_latency_ms=elapsed,
        )
    if not isinstance(payload, dict):
        return _error_prediction(
            base,
            "endpoint_response",
            "Endpoint JSON response must be an object.",
            http_status=response.status_code,
            client_elapsed_ms=elapsed,
            effective_latency_ms=elapsed,
        )
    if not response.ok:
        return _error_prediction(
            base,
            "endpoint_http",
            _response_error_message(payload, f"HTTP {response.status_code}"),
            http_status=response.status_code,
            client_elapsed_ms=elapsed,
            effective_latency_ms=elapsed,
            status=payload.get("status"),
            result_status=payload.get("result_status"),
        )

    question_type = base["question_type"]
    prediction = normalize_answer(payload.get("answer"), question_type)
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
    processing_time = _optional_int(payload.get("processing_time_ms"))
    confidence = _optional_float(payload.get("confidence"))
    correct = prediction is not None and prediction == base["normalized_ground_truth"]
    result = dict(base)
    result.update(
        {
            "answer": payload.get("answer"),
            "normalized_prediction": prediction,
            "correct": correct,
            "outcome": "CORRECT" if correct else "NULL" if prediction is None else "WRONG",
            "completed": True,
            "http_status": response.status_code,
            "confidence": confidence,
            "task": payload.get("task"),
            "status": payload.get("status"),
            "result_status": payload.get("result_status"),
            "processing_time_ms": processing_time,
            "client_elapsed_ms": elapsed,
            "effective_latency_ms": processing_time if processing_time is not None else elapsed,
            "model_used": payload.get("model_used"),
            "cached": payload.get("cached") if "cached" in payload else None,
            "warnings": [str(item) for item in warnings],
            "reuse_observations": _reuse_observations(payload),
        }
    )
    return result


def _prediction_to_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    output = {key: row.get(key) for key in PREDICTION_FIELDS}
    output["warnings_json"] = _json_string(row.get("warnings", []))
    reuse = dict(row.get("reuse_observations", {}))
    reuse["conversion_cache_hit"] = row.get("conversion_cache_hit")
    output["reuse_observations_json"] = _json_string(reuse)
    for key in ("correct", "completed"):
        output[key] = "true" if row.get(key) else "false"
    for key, value in list(output.items()):
        if value is None:
            output[key] = ""
    return output


def _prediction_from_csv(row: Mapping[str, str]) -> Dict[str, Any]:
    warnings: List[str] = []
    reuse: Dict[str, Any] = {}
    try:
        decoded = json.loads(row.get("warnings_json") or "[]")
        if isinstance(decoded, list):
            warnings = [str(item) for item in decoded]
    except json.JSONDecodeError:
        pass
    try:
        decoded_reuse = json.loads(row.get("reuse_observations_json") or "{}")
        if isinstance(decoded_reuse, dict):
            reuse = decoded_reuse
    except json.JSONDecodeError:
        pass
    result: Dict[str, Any] = dict(row)
    result.update(
        {
            "manifest_index": _optional_int(row.get("manifest_index")) or 0,
            "correct": _as_bool(row.get("correct")),
            "completed": _as_bool(row.get("completed")),
            "http_status": _optional_int(row.get("http_status")),
            "confidence": _optional_float(row.get("confidence")),
            "processing_time_ms": _optional_int(row.get("processing_time_ms")),
            "client_elapsed_ms": _optional_int(row.get("client_elapsed_ms")),
            "effective_latency_ms": _optional_int(row.get("effective_latency_ms")),
            "cached": None if row.get("cached") in (None, "") else _as_bool(row.get("cached")),
            "answer": row.get("answer") or None,
            "normalized_prediction": row.get("normalized_prediction") or None,
            "endpoint_error": row.get("endpoint_error") or None,
            "error_stage": row.get("error_stage") or None,
            "warnings": warnings,
            "reuse_observations": {key: value for key, value in reuse.items() if key != "conversion_cache_hit"},
            "conversion_cache_hit": reuse.get("conversion_cache_hit"),
        }
    )
    return result


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question_id" not in reader.fieldnames:
            raise ValueError(f"Existing predictions CSV has no question_id column: {path}")
        return {
            str(row["question_id"]): _prediction_from_csv(row)
            for row in reader
            if row.get("question_id")
        }


def write_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    ordered = sorted(rows, key=lambda item: int(item.get("manifest_index", 0)))
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row in ordered:
                writer.writerow(_prediction_to_csv(row))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress_line(index: int, total: int, row: Mapping[str, Any], resumed: bool = False) -> str:
    outcome = "RESUMED" if resumed else str(row.get("outcome") or "ERROR")
    prediction = row.get("normalized_prediction")
    if not row.get("completed"):
        prediction = "error"
    elif prediction is None:
        prediction = "null"
    latency = row.get("effective_latency_ms")
    latency_text = f"{latency} ms" if latency is not None else "n/a"
    return (
        f"[{index:02d}/{total:02d}] {outcome:<7} | {str(row.get('question_type')):<11} | "
        f"gt={row.get('normalized_ground_truth')} | pred={prediction} | {latency_text}"
    )


def run_records(
    records: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
    session: Optional[requests.Session] = None,
    progress_stream: TextIO = sys.stdout,
) -> List[Dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_predictions(config.predictions_path) if config.resume else {}
    selected_ids = {str(record["question_id"]) for record in records}
    predictions = {key: value for key, value in existing.items() if key in selected_ids}
    client = session or requests.Session()
    owns_session = session is None
    try:
        for index, record in enumerate(records, start=1):
            question_id = str(record["question_id"])
            prior = predictions.get(question_id)
            if prior is not None and prior.get("completed"):
                print(_progress_line(index, len(records), prior, resumed=True), file=progress_stream, flush=True)
                continue
            row = execute_record(record, index, config, client)
            predictions[question_id] = row
            write_predictions(config.predictions_path, predictions.values())
            print(_progress_line(index, len(records), row), file=progress_stream, flush=True)
    finally:
        if owns_session:
            client.close()
    return sorted(predictions.values(), key=lambda item: int(item["manifest_index"]))


def aggregate_metrics(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    completed = sum(bool(item.get("completed")) for item in predictions)
    correct = sum(bool(item.get("correct")) for item in predictions)
    null_answers = sum(bool(item.get("completed")) and item.get("normalized_prediction") is None for item in predictions)
    endpoint_errors = total - completed
    confidence_values = [
        value for value in (_optional_float(item.get("confidence")) for item in predictions) if value is not None
    ]
    latency_values = [
        value
        for item in predictions
        for value in [_optional_float(item.get("effective_latency_ms"))]
        if value is not None and item.get("completed")
    ]
    by_type: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in predictions:
        grouped[str(item.get("question_type"))].append(item)
    for question_type in sorted(grouped):
        rows = grouped[question_type]
        type_correct = sum(bool(item.get("correct")) for item in rows)
        by_type[question_type] = {
            "samples": len(rows),
            "completed": sum(bool(item.get("completed")) for item in rows),
            "correct": type_correct,
            "accuracy": type_correct / len(rows) if rows else None,
            "accuracy_percent": round(100.0 * type_correct / len(rows), 2) if rows else None,
            "null_answers": sum(bool(item.get("completed")) and item.get("normalized_prediction") is None for item in rows),
            "endpoint_errors": sum(not bool(item.get("completed")) for item in rows),
        }
    completed_rows = [item for item in predictions if item.get("completed")]
    coldest = completed_rows[0] if completed_rows else None
    slowest = max(completed_rows, key=lambda item: _optional_float(item.get("effective_latency_ms")) or -1) if completed_rows else None
    reuse_totals: Counter = Counter()
    conversion_cache_hits = 0
    conversion_cache_misses = 0
    for item in predictions:
        cache_hit = item.get("conversion_cache_hit")
        conversion_cache_hits += int(cache_hit is True)
        conversion_cache_misses += int(cache_hit is False)
        reuse = item.get("reuse_observations")
        if isinstance(reuse, dict):
            for key in (
                "sve_model_load_count",
                "sve_model_reuse_count",
                "sve_image_cache_hits",
                "sve_image_cache_misses",
                "grounder_model_reuse_count",
            ):
                reuse_totals[key] += int(reuse.get(key) or 0)
            reuse_totals["response_cached_count"] += int(reuse.get("response_cached") is True)
            reuse_totals["grounding_request_count"] += int(reuse.get("grounding_used") is True)

    def request_summary(item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        return {
            "question_id": item.get("question_id"),
            "image_id": item.get("image_id"),
            "question_type": item.get("question_type"),
            "processing_time_ms": item.get("processing_time_ms"),
            "client_elapsed_ms": item.get("client_elapsed_ms"),
            "effective_latency_ms": item.get("effective_latency_ms"),
            "model_used": item.get("model_used"),
        }

    return {
        "total_samples": total,
        "completed_count": completed,
        "correct_count": correct,
        "overall_accuracy": correct / total if total else None,
        "overall_accuracy_percent": round(100.0 * correct / total, 2) if total else None,
        "accuracy_by_question_type": by_type,
        "null_answer_count": null_answers,
        "endpoint_error_count": endpoint_errors,
        "average_confidence": round(statistics.fmean(confidence_values), 6) if confidence_values else None,
        "confidence_sample_count": len(confidence_values),
        "average_latency_ms": round(statistics.fmean(latency_values), 3) if latency_values else None,
        "median_latency_ms": round(statistics.median(latency_values), 3) if latency_values else None,
        "p95_latency_ms": round(float(np.percentile(latency_values, 95)), 3) if latency_values else None,
        "latency_sample_count": len(latency_values),
        "coldest_request": request_summary(coldest),
        "slowest_request": request_summary(slowest),
        "cache_reuse_observations": {
            "converted_png_cache_hits": conversion_cache_hits,
            "converted_png_cache_misses": conversion_cache_misses,
            **dict(reuse_totals),
        },
    }


def _confusion_key(item: Mapping[str, Any]) -> str:
    if not item.get("completed"):
        return "<error>"
    return str(item.get("normalized_prediction")) if item.get("normalized_prediction") is not None else "<null>"


def write_confusion_matrices(predictions: Sequence[Mapping[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    labels = sorted(
        {str(item.get("normalized_ground_truth")) for item in predictions}
        | {_confusion_key(item) for item in predictions},
        key=lambda value: ({"no": 0, "yes": 1, "rural": 2, "urban": 3}.get(value, 4), value),
    )
    overall_path = output_dir / "confusion_matrix_overall.csv"
    by_type_path = output_dir / "confusion_matrix_by_type.csv"
    overall_counts: Dict[str, Counter] = defaultdict(Counter)
    type_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for item in predictions:
        actual = str(item.get("normalized_ground_truth"))
        predicted = _confusion_key(item)
        overall_counts[actual][predicted] += 1
        type_counts[(str(item.get("question_type")), actual)][predicted] += 1
    with overall_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ground_truth", "total", *labels])
        writer.writeheader()
        for actual in labels:
            counts = overall_counts[actual]
            writer.writerow({"ground_truth": actual, "total": sum(counts.values()), **{label: counts[label] for label in labels}})
    with by_type_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_type", "ground_truth", "total", *labels])
        writer.writeheader()
        for question_type, actual in sorted(type_counts):
            counts = type_counts[(question_type, actual)]
            writer.writerow(
                {"question_type": question_type, "ground_truth": actual, "total": sum(counts.values()), **{label: counts[label] for label in labels}}
            )
    return overall_path, by_type_path


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _public_prediction(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"warnings_json", "reuse_observations_json"}}


def summary_text(summary: Mapping[str, Any], generated_files: Sequence[Path]) -> str:
    lines = [
        "RSVQA-LR Smoke Benchmark",
        "========================",
        f"Completed: {summary['completed_count']}/{summary['total_samples']}",
        f"Overall accuracy: {summary['overall_accuracy_percent']:.2f}%" if summary.get("overall_accuracy_percent") is not None else "Overall accuracy: n/a",
        f"Null answers: {summary['null_answer_count']}",
        f"Endpoint errors: {summary['endpoint_error_count']}",
        f"Average confidence: {summary['average_confidence'] if summary.get('average_confidence') is not None else 'n/a'}",
        f"Average latency: {summary['average_latency_ms'] if summary.get('average_latency_ms') is not None else 'n/a'} ms",
        f"Median latency: {summary['median_latency_ms'] if summary.get('median_latency_ms') is not None else 'n/a'} ms",
        f"P95 latency: {summary['p95_latency_ms'] if summary.get('p95_latency_ms') is not None else 'n/a'} ms",
        "",
        "Accuracy by question type:",
    ]
    for question_type, values in summary["accuracy_by_question_type"].items():
        lines.append(
            f"  {question_type}: {values['correct']}/{values['samples']} ({values['accuracy_percent']:.2f}%)"
        )
    lines.extend(["", "Cache/reuse observations:"])
    for key, value in summary["cache_reuse_observations"].items():
        lines.append(f"  {key}: {value}")
    for name in ("coldest_request", "slowest_request"):
        value = summary.get(name)
        lines.append(f"{name.replace('_', ' ').title()}: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["", "Generated files:"])
    lines.extend(f"  {path}" for path in generated_files)
    return "\n".join(lines) + "\n"


def write_outputs(
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    config: BenchmarkConfig,
    manifest_path: Path,
) -> List[Path]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.predictions_path
    write_predictions(predictions_path, predictions)
    errors_path = output_dir / "errors.json"
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.txt"
    overall_confusion, type_confusion = write_confusion_matrices(predictions, output_dir)
    errors = [
        {
            "manifest_index": item.get("manifest_index"),
            "question_id": item.get("question_id"),
            "image_id": item.get("image_id"),
            "question": item.get("question"),
            "stage": item.get("error_stage"),
            "http_status": item.get("http_status"),
            "error": item.get("endpoint_error"),
        }
        for item in predictions
        if not item.get("completed")
    ]
    _write_json(errors_path, errors)
    generated_files = [predictions_path, results_path, summary_path, errors_path, overall_confusion, type_confusion]
    results = {
        "benchmark": "RSVQA-LR SatQuery smoke benchmark",
        "manifest": str(manifest_path.expanduser().resolve()),
        "image_root": str(config.image_root.expanduser().resolve()),
        "endpoint": config.endpoint,
        "output_dir": str(output_dir.resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "predictions": [_public_prediction(item) for item in predictions],
    }
    _write_json(results_path, results)
    summary_path.write_text(summary_text(summary, generated_files), encoding="utf-8")
    return generated_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="RSVQA JSONL manifest")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Directory containing <image_id>.tif files")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="SatQuery /api/analysis/image URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds")
    parser.add_argument("--limit", type=int, help="Run only the first N records after loading the complete manifest")
    parser.add_argument("--resume", action="store_true", help="Skip completed question IDs already in predictions.csv")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    try:
        manifest_path = args.manifest.expanduser().resolve()
        records = load_manifest(manifest_path)
        selected = records[: args.limit] if args.limit is not None else records
        config = BenchmarkConfig(
            image_root=args.image_root.expanduser().resolve(),
            endpoint=normalize_endpoint(args.endpoint),
            output_dir=args.output_dir.expanduser().resolve(),
            timeout=args.timeout,
            resume=args.resume,
        )
        print(
            f"Loaded {len(records)} records; running {len(selected)} against {config.endpoint} "
            f"({'resume enabled' if config.resume else 'fresh run'}).",
            flush=True,
        )
        predictions = run_records(selected, config)
        summary = aggregate_metrics(predictions)
        generated_files = write_outputs(predictions, summary, config, manifest_path)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Benchmark setup failed: {error}", file=sys.stderr)
        return 2

    print("\nFinal summary")
    print(f"Completed: {summary['completed_count']}/{summary['total_samples']}")
    print(f"Overall accuracy: {summary['overall_accuracy_percent']:.2f}%")
    for question_type, values in summary["accuracy_by_question_type"].items():
        print(f"  {question_type}: {values['correct']}/{values['samples']} ({values['accuracy_percent']:.2f}%)")
    print(f"Null answers: {summary['null_answer_count']} | Endpoint errors: {summary['endpoint_error_count']}")
    print(f"Average latency: {summary['average_latency_ms']} ms | P95 latency: {summary['p95_latency_ms']} ms")
    print(f"Artifacts: {config.output_dir}")
    return 0 if summary["endpoint_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
