#!/usr/bin/env python3
"""Run the durable official RSVQA-LR test benchmark against SatQuery.

The HTTP request contains only the rendered image and official question. Ground
truth and question-family metadata remain evaluator-side and are used only after
the endpoint response has been persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TextIO
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_rsvqa_smoke as smoke


EXPECTED_QUESTIONS = 10_004
EXPECTED_IMAGES = 100
MODEL_USED = "RSVQA Specialist v1"
TASKS = ("presence", "comp", "count", "rural_urban")
DEFAULT_MANIFEST = Path("artifacts/rsvqa_full_specialist_v1/rsvqa_lr_test_manifest.jsonl")
DEFAULT_IMAGE_ROOT = Path("datasets/RSVQA/images/Images_LR")
DEFAULT_ENDPOINT = "http://127.0.0.1:8010/api/analysis/image"
DEFAULT_OUTPUT_DIR = Path("artifacts/rsvqa_full_specialist_v1")
CHECKPOINT_SHA256 = "71c0ab56ee650813bd495e8a3bc777353b6907a097af860e417f60523efe56ad"
PREDICTION_FIELDS = (
    "manifest_index", "question_id", "image_id", "source_image_path", "converted_image_path",
    "question", "question_type", "ground_truth", "normalized_ground_truth_exact",
    "normalized_ground_truth_overflow", "answer", "normalized_prediction", "exact_correct",
    "overflow_correct", "outcome", "completed", "endpoint_error", "error_stage", "http_status",
    "attempts", "confidence", "task", "status", "result_status", "processing_time_ms",
    "client_elapsed_ms", "effective_latency_ms", "model_used", "specialist_used", "fallback_used",
    "cached", "warnings_json", "conversion_cache_hit",
)
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
COUNT_BUCKETS = (
    "0", "1", "2-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+", "<null>", "<error>",
)


@dataclass(frozen=True)
class FullBenchmarkConfig:
    manifest: Path
    image_root: Path
    endpoint: str
    output_dir: Path
    timeout: float = 120.0
    resume: bool = False
    checkpoint_every: int = 25
    max_retries: int = 3
    limit: Optional[int] = None

    @property
    def converted_dir(self) -> Path:
        return self.output_dir / "converted"

    @property
    def partial_predictions(self) -> Path:
        return self.output_dir / "predictions.partial.csv"

    @property
    def partial_errors(self) -> Path:
        return self.output_dir / "errors.partial.json"

    @property
    def run_state(self) -> Path:
        return self.output_dir / "run_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest_from_official_exports(
    questions_path: Path,
    answers_path: Path,
    images_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Losslessly join the three official exports; never infer questions or labels."""
    questions_payload = json.loads(questions_path.read_text(encoding="utf-8"))
    answers_payload = json.loads(answers_path.read_text(encoding="utf-8"))
    images_payload = json.loads(images_path.read_text(encoding="utf-8"))
    questions = [item for item in questions_payload.get("questions", []) if item.get("active") is True]
    answers = [item for item in answers_payload.get("answers", []) if item.get("active") is True]
    images = [item for item in images_payload.get("images", []) if item.get("active") is True]
    if len(questions) != EXPECTED_QUESTIONS or len(answers) != EXPECTED_QUESTIONS or len(images) != EXPECTED_IMAGES:
        raise ValueError("Official export counts do not match the RSVQA-LR test split")
    answer_by_question = {int(item["question_id"]): item for item in answers}
    image_by_id = {int(item["id"]): item for item in images}
    if len(answer_by_question) != EXPECTED_QUESTIONS or len(image_by_id) != EXPECTED_IMAGES:
        raise ValueError("Official exports contain duplicate active identifiers")
    records = []
    for question in questions:
        question_id = int(question["id"])
        image_id = int(question["img_id"])
        answer = answer_by_question.get(question_id)
        image = image_by_id.get(image_id)
        if answer is None or image is None or list(question.get("answers_ids") or []) != [int(answer["id"])]:
            raise ValueError(f"Official export linkage failed for question {question_id}")
        records.append({
            "question_id": question_id,
            "image_id": image_id,
            "image_path": f"/content/drive/MyDrive/SatQuery_Benchmarks/RSVQA_LR/images/Images_LR/{image_id}.tif",
            "original_image_name": image.get("original_name"),
            "question": question["question"],
            "question_type": question["type"],
            "ground_truth": answer["answer"],
            "sensor": image.get("sensor"),
            "image_type": image.get("type"),
            "resolution_x": image.get("res_x"),
            "resolution_y": image.get("res_y"),
        })
    if len({item["question_id"] for item in records}) != EXPECTED_QUESTIONS:
        raise ValueError("Joined manifest contains duplicate question IDs")
    atomic_text(output_path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    return {
        "records": len(records),
        "unique_images": len({item["image_id"] for item in records}),
        "task_counts": dict(Counter(item["question_type"] for item in records)),
        "source_sha256": {
            questions_path.name: sha256_file(questions_path),
            answers_path.name: sha256_file(answers_path),
            images_path.name: sha256_file(images_path),
        },
        "manifest_sha256": sha256_file(output_path),
    }


def normalize_answer(value: Any, question_type: str, *, overflow_aware: bool = False) -> Optional[str]:
    """Normalize official answer tokens while keeping exact labels distinct from 201+."""
    if value is None or isinstance(value, bool):
        return None
    raw = str(value).strip().lower()
    raw = re.sub(r"[.!?,;:]+$", "", raw).strip()
    task = question_type.strip().lower()
    if task in {"presence", "comp"}:
        return raw if raw in {"yes", "no"} else None
    if task == "rural_urban":
        return raw if raw in {"rural", "urban"} else None
    if task == "count":
        if raw == "201+":
            return raw
        if not re.fullmatch(r"0|[1-9]\d*", raw):
            return None
        return "201+" if overflow_aware and int(raw) > 200 else raw
    return None


def load_manifest(path: Path, image_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on manifest line {line_number}") from error
            required = {"question_id", "image_id", "question", "question_type", "ground_truth"}
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"Manifest line {line_number} is missing required official fields")
            question_id = str(item["question_id"])
            task = str(item["question_type"]).strip().lower()
            if question_id in ids or task not in TASKS:
                raise ValueError(f"Duplicate ID or unsupported task on manifest line {line_number}")
            ids.add(question_id)
            if normalize_answer(item["ground_truth"], task) is None:
                raise ValueError(f"Invalid official answer on manifest line {line_number}")
            records.append(item)
    if len(records) != EXPECTED_QUESTIONS:
        raise ValueError(f"Official manifest must contain {EXPECTED_QUESTIONS} records, found {len(records)}")
    image_ids = {str(item["image_id"]) for item in records}
    if len(image_ids) != EXPECTED_IMAGES:
        raise ValueError(f"Official manifest must contain {EXPECTED_IMAGES} unique images, found {len(image_ids)}")
    missing = sorted(image_id for image_id in image_ids if not (image_root / f"{image_id}.tif").is_file())
    if missing:
        raise ValueError(f"Official test images are missing: {missing[:10]}")
    return records


def health_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, "/api/agent/health", "", ""))


def check_backend_health(config: FullBenchmarkConfig, session: requests.Session) -> dict[str, Any]:
    response = session.get(health_url(config.endpoint), timeout=config.timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("SatQuery backend health check did not return ok")
    return payload


def post_question(
    record: Mapping[str, Any],
    png_path: Path,
    config: FullBenchmarkConfig,
    session: requests.Session,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str], Optional[int], int, int]:
    """Post only image/question fields and retry transient failures."""
    total_attempts = config.max_retries + 1
    last_error: Optional[str] = None
    last_stage: Optional[str] = None
    last_status: Optional[int] = None
    overall_started = time.perf_counter()
    for attempt in range(1, total_attempts + 1):
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
            last_status = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if response.ok and isinstance(payload, dict):
                elapsed = max(0, round((time.perf_counter() - overall_started) * 1000))
                return payload, None, None, response.status_code, attempt, elapsed
            last_stage = "endpoint_http" if not response.ok else "endpoint_response"
            last_error = smoke._response_error_message(payload, f"HTTP {response.status_code}")
            if response.status_code not in TRANSIENT_HTTP:
                break
        except requests.RequestException as error:
            last_stage, last_error = "endpoint_request", str(error)
        if attempt < total_attempts:
            time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))
    elapsed = max(0, round((time.perf_counter() - overall_started) * 1000))
    return None, last_error or "Endpoint request failed", last_stage or "endpoint_request", last_status, min(total_attempts, attempt), elapsed


def execute_record(
    record: Mapping[str, Any],
    manifest_index: int,
    config: FullBenchmarkConfig,
    session: requests.Session,
) -> dict[str, Any]:
    task = str(record["question_type"]).strip().lower()
    image_id = str(record["image_id"])
    source = config.image_root / f"{image_id}.tif"
    base = {
        "manifest_index": manifest_index,
        "question_id": str(record["question_id"]),
        "image_id": image_id,
        "source_image_path": str(source),
        "converted_image_path": None,
        "question": str(record["question"]),
        "question_type": task,
        "ground_truth": str(record["ground_truth"]),
        "normalized_ground_truth_exact": normalize_answer(record["ground_truth"], task),
        "normalized_ground_truth_overflow": normalize_answer(record["ground_truth"], task, overflow_aware=True),
        "answer": None,
        "normalized_prediction": None,
        "exact_correct": False,
        "overflow_correct": False,
        "outcome": "ERROR",
        "completed": False,
        "endpoint_error": None,
        "error_stage": None,
        "http_status": None,
        "attempts": 0,
        "confidence": None,
        "task": None,
        "status": None,
        "result_status": None,
        "processing_time_ms": None,
        "client_elapsed_ms": None,
        "effective_latency_ms": None,
        "model_used": "<endpoint_error>",
        "specialist_used": False,
        "fallback_used": False,
        "cached": None,
        "warnings": [],
        "conversion_cache_hit": None,
    }
    try:
        png_path, cache_hit = smoke.convert_tiff_to_png(source, config.converted_dir, image_id)
        base.update(converted_image_path=str(png_path), conversion_cache_hit=cache_hit)
    except Exception as error:
        base.update(endpoint_error=str(error), error_stage="image_conversion")
        return base
    payload, error, stage, status_code, attempts, elapsed = post_question(record, png_path, config, session)
    base.update(http_status=status_code, attempts=attempts, client_elapsed_ms=elapsed)
    if payload is None:
        base.update(endpoint_error=error, error_stage=stage, effective_latency_ms=elapsed)
        return base
    prediction = normalize_answer(payload.get("answer"), task)
    processing_ms = smoke._optional_int(payload.get("processing_time_ms"))
    confidence = smoke._optional_float(payload.get("confidence"))
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    exact_correct = prediction is not None and prediction == base["normalized_ground_truth_exact"]
    overflow_correct = prediction is not None and prediction == base["normalized_ground_truth_overflow"]
    model_used = str(payload.get("model_used") or "<missing>")
    specialist_used = model_used == MODEL_USED
    base.update(
        answer=payload.get("answer"),
        normalized_prediction=prediction,
        exact_correct=exact_correct,
        overflow_correct=overflow_correct,
        outcome="CORRECT" if exact_correct else "NULL" if prediction is None else "WRONG",
        completed=True,
        confidence=confidence,
        task=payload.get("task"),
        status=payload.get("status"),
        result_status=payload.get("result_status"),
        processing_time_ms=processing_ms,
        effective_latency_ms=processing_ms if processing_ms is not None else elapsed,
        model_used=model_used,
        specialist_used=specialist_used,
        fallback_used=not specialist_used,
        cached=payload.get("cached") if "cached" in payload else None,
        warnings=[str(item) for item in warnings],
    )
    return base


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {field: row.get(field) for field in PREDICTION_FIELDS}
    output["warnings_json"] = json.dumps(row.get("warnings") or [], ensure_ascii=False)
    for field in ("exact_correct", "overflow_correct", "completed", "specialist_used", "fallback_used"):
        output[field] = "true" if row.get(field) else "false"
    for key, value in list(output.items()):
        if value is None:
            output[key] = ""
    return output


def _from_csv(row: Mapping[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    for field in ("exact_correct", "overflow_correct", "completed", "specialist_used", "fallback_used"):
        output[field] = smoke._as_bool(row.get(field))
    for field in ("manifest_index", "http_status", "attempts", "processing_time_ms", "client_elapsed_ms", "effective_latency_ms"):
        output[field] = smoke._optional_int(row.get(field))
    output["confidence"] = smoke._optional_float(row.get("confidence"))
    output["normalized_prediction"] = row.get("normalized_prediction") or None
    output["endpoint_error"] = row.get("endpoint_error") or None
    output["conversion_cache_hit"] = None if not row.get("conversion_cache_hit") else smoke._as_bool(row.get("conversion_cache_hit"))
    try:
        output["warnings"] = json.loads(row.get("warnings_json") or "[]")
    except json.JSONDecodeError:
        output["warnings"] = []
    return output


def write_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row.get("manifest_index") or 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(row) for row in ordered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PREDICTION_FIELDS):
            raise ValueError(f"Prediction schema mismatch: {path}")
        rows = [_from_csv(row) for row in reader]
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Partial predictions contain duplicate question IDs")
    return dict(zip(ids, rows))


def errors_for(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in ("manifest_index", "question_id", "image_id", "question", "error_stage", "http_status", "attempts", "endpoint_error", "model_used")}
        for row in rows if not row.get("completed")
    ]


def checkpoint(
    config: FullBenchmarkConfig,
    predictions: Mapping[str, Mapping[str, Any]],
    state: dict[str, Any],
) -> None:
    rows = list(predictions.values())
    write_predictions(config.partial_predictions, rows)
    atomic_json(config.partial_errors, errors_for(rows))
    state.update(
        updated_at=utc_now(),
        processed_records=len(rows),
        successful_responses=sum(bool(row.get("completed")) for row in rows),
        endpoint_errors=sum(not bool(row.get("completed")) for row in rows),
        exact_correct=sum(bool(row.get("exact_correct")) for row in rows),
        overflow_correct=sum(bool(row.get("overflow_correct")) for row in rows),
    )
    atomic_json(config.run_state, state)


def warm_up(
    records: Sequence[Mapping[str, Any]],
    config: FullBenchmarkConfig,
    session: requests.Session,
    stream: TextIO = sys.stdout,
) -> list[dict[str, Any]]:
    selected = []
    for task in TASKS:
        record = next(item for item in records if item["question_type"] == task)
        image_id = str(record["image_id"])
        png, _ = smoke.convert_tiff_to_png(config.image_root / f"{image_id}.tif", config.converted_dir, image_id)
        payload, error, _, _, attempts, elapsed = post_question(record, png, config, session)
        if payload is None or payload.get("model_used") != MODEL_USED:
            raise ValueError(f"Warm-up failed for {task}: {error or payload.get('model_used') if payload else error}")
        result = {"task": task, "question_id": str(record["question_id"]), "model_used": payload.get("model_used"), "attempts": attempts, "elapsed_ms": elapsed}
        selected.append(result)
        print(f"Warm-up {task}: {MODEL_USED} ({elapsed} ms)", file=stream, flush=True)
    return selected


def live_progress(processed: int, total: int, predictions: Iterable[Mapping[str, Any]]) -> str:
    rows = list(predictions)
    accuracy = sum(bool(row.get("exact_correct")) for row in rows) / len(rows) if rows else 0.0
    completed = [row for row in rows if row.get("completed")]
    coverage = sum(row.get("normalized_prediction") is not None for row in completed) / len(rows) if rows else 0.0
    latencies = [float(row["effective_latency_ms"]) for row in completed if row.get("effective_latency_ms") is not None]
    average = statistics.fmean(latencies) if latencies else 0.0
    return f"[{processed:04d}/{total}] accuracy={accuracy:.2%} | coverage={coverage:.2%} | avg={average:.0f} ms"


def run_records(
    records: Sequence[Mapping[str, Any]],
    config: FullBenchmarkConfig,
    session: Optional[requests.Session] = None,
    stream: TextIO = sys.stdout,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing_path = config.partial_predictions if config.partial_predictions.is_file() else config.output_dir / "predictions.csv"
    existing = load_predictions(existing_path) if config.resume else {}
    selected_ids = {str(record["question_id"]) for record in records}
    predictions = {qid: row for qid, row in existing.items() if qid in selected_ids}
    client = session or requests.Session()
    owns_session = session is None
    state = {
        "status": "running",
        "started_at": utc_now(),
        "benchmark_started_monotonic": time.perf_counter(),
        "manifest": str(config.manifest),
        "manifest_sha256": sha256_file(config.manifest),
        "expected_records": len(records),
        "expected_official_records": EXPECTED_QUESTIONS,
        "endpoint": config.endpoint,
        "image_root": str(config.image_root),
        "checkpoint_every": config.checkpoint_every,
        "max_retries": config.max_retries,
        "resume": config.resume,
        "command": reproducibility_command(config),
    }
    started = time.perf_counter()
    try:
        for index, record in enumerate(records, start=1):
            qid = str(record["question_id"])
            if qid in predictions:
                continue
            predictions[qid] = execute_record(record, index, config, client)
            processed = len(predictions)
            if processed % config.checkpoint_every == 0:
                checkpoint(config, predictions, state)
            if processed % 100 == 0 or processed == len(records):
                print(live_progress(processed, len(records), predictions.values()), file=stream, flush=True)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        checkpoint(config, predictions, state)
        raise
    finally:
        if owns_session:
            client.close()
    state["benchmark_wall_clock_seconds"] = round(time.perf_counter() - started, 3)
    state["status"] = "predictions_complete" if len(predictions) == len(records) else "partial"
    checkpoint(config, predictions, state)
    return sorted(predictions.values(), key=lambda row: int(row["manifest_index"])), state


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return round(float(np.percentile(values, q)), 3) if values else None


def count_bucket(value: Optional[str], *, error: bool = False) -> str:
    if error:
        return "<error>"
    if value is None:
        return "<null>"
    if value == "201+":
        return value
    if not re.fullmatch(r"0|[1-9]\d*", value):
        return "<null>"
    number = int(value)
    if number == 0:
        return "0"
    if number == 1:
        return "1"
    for low, high in ((2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200)):
        if low <= number <= high:
            return f"{low}-{high}"
    return "201+"


def wording_pattern(question: str) -> str:
    value = re.sub(r"\s+", " ", question.lower()).strip()
    rules = (
        ("rural-or-urban", r"rural.*urban|urban.*rural"),
        ("equal-to", r"equal to|same as"),
        ("less-than", r"\b(?:less|fewer)\b.*\bthan\b"),
        ("more-than", r"\b(?:more|greater)\b.*\bthan\b"),
        ("how-many", r"^how many\b"),
        ("number-of", r"\bnumber of\b"),
        ("amount-of", r"\bamount of\b"),
        ("is-there", r"^(?:is|are) there\b"),
        ("visible", r"\bvisible\b"),
        ("present", r"\bpresent\b"),
        ("contain/show", r"\b(?:contain|show|see)\b"),
    )
    return next((label for label, pattern in rules if re.search(pattern, value)), "other")


def canonical_target(question: str) -> str:
    try:
        from satquery_agent.specialists.vqa import get_vqa

        intent = get_vqa().classify_question(question)
        values = [value for value in (intent.target, intent.secondary_target) if value]
        return " vs ".join(values) if values else "scene"
    except Exception:
        return "unresolved"


def confidence_range(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    if value < 0.5:
        return "0.00-0.49"
    if value < 0.75:
        return "0.50-0.74"
    if value < 0.9:
        return "0.75-0.89"
    return "0.90-1.00"


def majority_baselines(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for task in TASKS:
        labels = [normalize_answer(item["ground_truth"], task) for item in records if item["question_type"] == task]
        label, count = Counter(labels).most_common(1)[0]
        result[task] = {"label": label, "correct": count, "samples": len(labels), "accuracy": count / len(labels)}
    return result


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], wall_seconds: float) -> dict[str, Any]:
    total = len(rows)
    successful = [row for row in rows if row.get("completed")]
    answered = [row for row in successful if row.get("normalized_prediction") is not None]
    exact_correct = [row for row in rows if row.get("exact_correct")]
    overflow_correct = [row for row in rows if row.get("overflow_correct")]
    specialist = [row for row in successful if row.get("specialist_used")]
    fallback = [row for row in successful if row.get("fallback_used")]
    confidence_all = [float(row["confidence"]) for row in successful if row.get("confidence") is not None]
    confidence_correct = [float(row["confidence"]) for row in successful if row.get("confidence") is not None and row.get("exact_correct")]
    confidence_incorrect = [float(row["confidence"]) for row in successful if row.get("confidence") is not None and not row.get("exact_correct")]
    latency = [float(row["effective_latency_ms"]) for row in successful if row.get("effective_latency_ms") is not None]
    by_type = {}
    for task in TASKS:
        selected = [row for row in rows if row.get("question_type") == task]
        samples = len(selected)
        by_type[task] = {
            "samples": samples,
            "successful_responses": sum(bool(row.get("completed")) for row in selected),
            "correct": sum(bool(row.get("exact_correct")) for row in selected),
            "overflow_correct": sum(bool(row.get("overflow_correct")) for row in selected),
            "accuracy": sum(bool(row.get("exact_correct")) for row in selected) / samples if samples else None,
            "overflow_aware_accuracy": sum(bool(row.get("overflow_correct")) for row in selected) / samples if samples else None,
            "coverage": sum(bool(row.get("completed")) and row.get("normalized_prediction") is not None for row in selected) / samples if samples else None,
            "null_answers": sum(bool(row.get("completed")) and row.get("normalized_prediction") is None for row in selected),
            "endpoint_errors": sum(not bool(row.get("completed")) for row in selected),
        }
    return {
        "total_records": total,
        "successful_responses": len(successful),
        "endpoint_errors": total - len(successful),
        "exact_correct": len(exact_correct),
        "overflow_correct": len(overflow_correct),
        "exact_match_accuracy": len(exact_correct) / total if total else 0.0,
        "overflow_aware_accuracy": len(overflow_correct) / total if total else 0.0,
        "coverage": len(answered) / total if total else 0.0,
        "null_answers": len(successful) - len(answered),
        "null_answer_rate": (len(successful) - len(answered)) / total if total else 0.0,
        "endpoint_error_rate": (total - len(successful)) / total if total else 0.0,
        "specialist_use_rate": len(specialist) / total if total else 0.0,
        "fallback_use_rate": len(fallback) / total if total else 0.0,
        "average_confidence": statistics.fmean(confidence_all) if confidence_all else None,
        "average_confidence_correct": statistics.fmean(confidence_correct) if confidence_correct else None,
        "average_confidence_incorrect": statistics.fmean(confidence_incorrect) if confidence_incorrect else None,
        "average_latency_ms": statistics.fmean(latency) if latency else None,
        "median_latency_ms": statistics.median(latency) if latency else None,
        "p90_latency_ms": percentile(latency, 90),
        "p95_latency_ms": percentile(latency, 95),
        "p99_latency_ms": percentile(latency, 99),
        "slowest_request": max(successful, key=lambda row: float(row.get("effective_latency_ms") or -1), default=None),
        "throughput_questions_per_second": total / wall_seconds if wall_seconds > 0 else None,
        "total_wall_clock_seconds": wall_seconds,
        "by_type": by_type,
        "macro_task_accuracy": statistics.fmean(item["accuracy"] for item in by_type.values() if item["accuracy"] is not None),
    }


def count_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("question_type") == "count"]
    buckets = {}
    for bucket in COUNT_BUCKETS[:9]:
        values = [row for row in selected if count_bucket(str(row.get("normalized_ground_truth_exact"))) == bucket]
        buckets[bucket] = {
            "samples": len(values),
            "correct": sum(bool(row.get("exact_correct")) for row in values),
            "accuracy": sum(bool(row.get("exact_correct")) for row in values) / len(values) if values else None,
        }
    numeric_pairs = []
    for row in selected:
        actual, predicted = row.get("normalized_ground_truth_exact"), row.get("normalized_prediction")
        if isinstance(actual, str) and isinstance(predicted, str) and actual.isdigit() and predicted.isdigit():
            numeric_pairs.append(abs(int(actual) - int(predicted)))
    return {
        "samples": len(selected),
        "exact_count_accuracy": sum(bool(row.get("exact_correct")) for row in selected) / len(selected),
        "overflow_aware_accuracy": sum(bool(row.get("overflow_correct")) for row in selected) / len(selected),
        "accuracy_by_ground_truth_bucket": buckets,
        "overflow_ground_truth_count": sum(count_bucket(str(row.get("normalized_ground_truth_exact"))) == "201+" for row in selected),
        "prediction_distribution": dict(sorted(Counter(row.get("normalized_prediction") or "<null>" for row in selected).items())),
        "ground_truth_distribution": dict(sorted(Counter(str(row.get("normalized_ground_truth_exact")) for row in selected).items())),
        "numeric_pair_count": len(numeric_pairs),
        "mean_absolute_error": statistics.fmean(numeric_pairs) if numeric_pairs else None,
        "median_absolute_error": statistics.median(numeric_pairs) if numeric_pairs else None,
        "within_1": sum(error <= 1 for error in numeric_pairs) / len(numeric_pairs) if numeric_pairs else None,
        "within_2": sum(error <= 2 for error in numeric_pairs) / len(numeric_pairs) if numeric_pairs else None,
        "within_5": sum(error <= 5 for error in numeric_pairs) / len(numeric_pairs) if numeric_pairs else None,
    }


def write_matrix(path: Path, rows: Sequence[Mapping[str, Any]], labels: Sequence[str], *, bucketed: bool = False) -> None:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if bucketed:
            actual = count_bucket(row.get("normalized_ground_truth_exact"))
            predicted = count_bucket(row.get("normalized_prediction"), error=not bool(row.get("completed")))
        else:
            actual = str(row.get("normalized_ground_truth_exact"))
            predicted = str(row.get("normalized_prediction")) if row.get("normalized_prediction") is not None else "<null>"
            if not row.get("completed"):
                predicted = "<error>"
        counts[actual][predicted] += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ground_truth", "total", *labels, "<other>"])
        writer.writeheader()
        for actual in labels:
            values = counts[actual]
            other = sum(count for predicted, count in values.items() if predicted not in labels)
            writer.writerow({"ground_truth": actual, "total": sum(values.values()), **{label: values[label] for label in labels}, "<other>": other})
        unlisted_actual = [actual for actual in counts if actual not in labels]
        if unlisted_actual:
            values = sum((counts[actual] for actual in unlisted_actual), Counter())
            other = sum(count for predicted, count in values.items() if predicted not in labels)
            writer.writerow({"ground_truth": "<other>", "total": sum(values.values()), **{label: values[label] for label in labels}, "<other>": other})


def write_group_metrics(path: Path, rows: Sequence[Mapping[str, Any]], key_name: str, key_fn: Any) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    fields = [key_name, "samples", "correct", "accuracy", "coverage", "endpoint_errors", "average_confidence"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            confidence = [float(row["confidence"]) for row in values if row.get("confidence") is not None]
            writer.writerow({
                key_name: key,
                "samples": len(values),
                "correct": sum(bool(row.get("exact_correct")) for row in values),
                "accuracy": sum(bool(row.get("exact_correct")) for row in values) / len(values),
                "coverage": sum(bool(row.get("completed")) and row.get("normalized_prediction") is not None for row in values) / len(values),
                "endpoint_errors": sum(not bool(row.get("completed")) for row in values),
                "average_confidence": statistics.fmean(confidence) if confidence else "",
            })


def write_error_analysis(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    failures = [row for row in rows if not row.get("exact_correct")]
    fields = [
        "question_id", "image_id", "question_type", "question", "wording_pattern", "canonical_target",
        "ground_truth", "prediction", "confidence", "confidence_range", "count_bucket", "model_used",
        "fallback_used", "endpoint_error",
    ]
    enriched = []
    for row in failures:
        enriched.append({
            "question_id": row.get("question_id"), "image_id": row.get("image_id"),
            "question_type": row.get("question_type"), "question": row.get("question"),
            "wording_pattern": wording_pattern(str(row.get("question"))),
            "canonical_target": canonical_target(str(row.get("question"))),
            "ground_truth": row.get("normalized_ground_truth_exact"),
            "prediction": row.get("normalized_prediction") or "<null>",
            "confidence": row.get("confidence"), "confidence_range": confidence_range(row.get("confidence")),
            "count_bucket": count_bucket(row.get("normalized_ground_truth_exact")) if row.get("question_type") == "count" else "n/a",
            "model_used": row.get("model_used"), "fallback_used": row.get("fallback_used"),
            "endpoint_error": row.get("endpoint_error"),
        })
    with (output_dir / "failure_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(enriched)
    high_confidence = [item for item in enriched if isinstance(item["confidence"], (int, float)) and item["confidence"] >= 0.9]
    count_predictions = Counter(item["prediction"] for item in enriched if item["question_type"] == "count")
    count_failure_total = sum(count_predictions.values())
    dominant_count = count_predictions.most_common(1)[0] if count_predictions else (None, 0)
    dominant_count_rate = dominant_count[1] / count_failure_total if count_failure_total else 0.0
    return {
        "incorrect_or_error_records": len(enriched),
        "high_confidence_errors": len(high_confidence),
        "highest_confidence_error_category": Counter(item["question_type"] for item in high_confidence).most_common(1)[0][0] if high_confidence else None,
        "errors_by_question_type": dict(Counter(item["question_type"] for item in enriched)),
        "errors_by_ground_truth": dict(Counter(item["ground_truth"] for item in enriched)),
        "errors_by_prediction": dict(Counter(item["prediction"] for item in enriched)),
        "errors_by_confidence_range": dict(Counter(item["confidence_range"] for item in enriched)),
        "errors_by_count_bucket": dict(Counter(item["count_bucket"] for item in enriched if item["question_type"] == "count")),
        "recurring_wording_patterns": Counter(item["wording_pattern"] for item in enriched).most_common(20),
        "recurring_entity_failures": Counter(item["canonical_target"] for item in enriched).most_common(20),
        "image_specific_clusters": Counter(item["image_id"] for item in enriched).most_common(20),
        "fallback_cases": sum(bool(item["fallback_used"]) for item in enriched),
        "preprocessing_issues": sum(
            bool(item["endpoint_error"])
            and "conversion" in str(item["endpoint_error"]).lower()
            for item in enriched
        ),
        "count_prediction_distribution": dict(count_predictions),
        "dominant_count_failure_prediction": dominant_count[0],
        "dominant_count_failure_rate": dominant_count_rate,
        "count_class_collapse_detected": dominant_count_rate >= 0.5,
    }


def load_smoke_references(project_root: Path) -> dict[str, Any]:
    """Load already-produced Smoke-50 results without treating them as full-test data."""
    references = {}
    for key, relative in (
        ("old_heuristic_smoke50", "artifacts/rsvqa_smoke/results.json"),
        ("specialist_smoke50", "artifacts/rsvqa_smoke_specialist_v1/results.json"),
    ):
        path = project_root / relative
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary") or {}
        references[key] = {
            "scope": "Smoke-50 only",
            "path": str(path),
            "total_samples": summary.get("total_samples"),
            "overall_accuracy": summary.get("overall_accuracy"),
            "coverage": (
                (summary.get("total_samples", 0) - summary.get("null_answer_count", 0))
                / summary.get("total_samples", 1)
            ),
            "average_latency_ms": summary.get("average_latency_ms"),
            "accuracy_by_question_type": summary.get("accuracy_by_question_type"),
        }
    return references


def write_confidence_analysis(rows: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    bins = ((0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.000001))
    output = []
    for low, high in bins:
        values = [row for row in rows if row.get("confidence") is not None and low <= float(row["confidence"]) < high]
        output.append({
            "confidence_bin": f"{low:.2f}-{min(high, 1.0):.2f}",
            "samples": len(values),
            "average_confidence": statistics.fmean(float(row["confidence"]) for row in values) if values else None,
            "accuracy": sum(bool(row.get("exact_correct")) for row in values) / len(values) if values else None,
            "correct": sum(bool(row.get("exact_correct")) for row in values),
            "incorrect": sum(not bool(row.get("exact_correct")) for row in values),
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    return output


def _colors() -> list[str]:
    return ["#1677ff", "#12b886", "#f59f00", "#e64980", "#845ef7", "#15aabf"]


def write_bar_chart(asset_dir: Path, name: str, title: str, labels: Sequence[str], values: Sequence[float], *, percent: bool = True) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 120, 60, 90, 150
    plot_w, plot_h = width - margin_left - margin_right, height - margin_top - margin_bottom
    maximum = 1.0 if percent else max(values or [1.0]) * 1.1 or 1.0
    colors = _colors()
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 25), title, fill="#172b4d", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#6b778c", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="#6b778c", width=2)
    for tick in range(6):
        value = maximum * tick / 5
        y = margin_top + plot_h - plot_h * tick / 5
        draw.line((margin_left, y, margin_left + plot_w, y), fill="#e9ecef", width=1)
        label = f"{value:.0%}" if percent else f"{value:.0f}"
        draw.text((20, y - 10), label, fill="#495057", font=small)
    slot = plot_w / max(1, len(values))
    bar_w = slot * 0.58
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_left + slot * index + (slot - bar_w) / 2
        x1 = x0 + bar_w
        y0 = margin_top + plot_h - plot_h * value / maximum
        draw.rounded_rectangle((x0, y0, x1, margin_top + plot_h), radius=8, fill=colors[index % len(colors)])
        value_label = f"{value:.1%}" if percent else f"{value:.1f}"
        draw.text((x0, max(margin_top, y0 - 30)), value_label, fill="#172b4d", font=small)
        draw.text((x0, margin_top + plot_h + 20), label[:22], fill="#172b4d", font=small)
    image.save(asset_dir / f"{name}.png", format="PNG")

    def escape(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{margin_left}" y="50" font-family="sans-serif" font-size="28" fill="#172b4d">{escape(title)}</text>']
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#6b778c" stroke-width="2"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#6b778c" stroke-width="2"/>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_left + slot * index + (slot - bar_w) / 2
        y0 = margin_top + plot_h - plot_h * value / maximum
        svg.extend([f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{margin_top + plot_h - y0:.1f}" rx="8" fill="{colors[index % len(colors)]}"/>', f'<text x="{x0:.1f}" y="{max(margin_top, y0 - 12):.1f}" font-family="sans-serif" font-size="18" fill="#172b4d">{escape(f"{value:.1%}" if percent else f"{value:.1f}")}</text>', f'<text x="{x0:.1f}" y="{margin_top + plot_h + 35}" font-family="sans-serif" font-size="17" fill="#172b4d">{escape(label[:22])}</text>'])
    svg.append("</svg>")
    atomic_text(asset_dir / f"{name}.svg", "\n".join(svg) + "\n")


def write_histogram(asset_dir: Path, name: str, title: str, values: Sequence[float]) -> None:
    if not values:
        write_bar_chart(asset_dir, name, title, ["No data"], [0.0], percent=False)
        return
    counts, edges = np.histogram(values, bins=12)
    labels = [f"{edges[index]:.0f}-{edges[index + 1]:.0f}" for index in range(len(counts))]
    write_bar_chart(asset_dir, name, title, labels, [float(value) for value in counts], percent=False)


def reproducibility_command(config: FullBenchmarkConfig) -> str:
    parts = [
        "venv/bin/python", "scripts/run_rsvqa_full.py",
        "--manifest", str(config.manifest), "--image-root", str(config.image_root),
        "--endpoint", config.endpoint, "--output-dir", str(config.output_dir),
        "--timeout", f"{config.timeout:g}", "--checkpoint-every", str(config.checkpoint_every),
        "--max-retries", str(config.max_retries),
    ]
    if config.resume:
        parts.append("--resume")
    if config.limit is not None:
        parts.extend(("--limit", str(config.limit)))
    return shlex.join(parts)


def write_summary(summary: Mapping[str, Any], count: Mapping[str, Any], baselines: Mapping[str, Any]) -> str:
    lines = [
        "Official RSVQA-LR Test Benchmark — SatQuery RSVQA Specialist v1",
        "===============================================================",
        f"Processed records: {summary['total_records']}",
        f"Exact-match accuracy: {summary['exact_match_accuracy']:.4%}",
        f"Overflow-aware accuracy: {summary['overflow_aware_accuracy']:.4%}",
        f"Macro task accuracy: {summary['macro_task_accuracy']:.4%}",
        f"Coverage: {summary['coverage']:.4%}",
        f"Null answers: {summary['null_answers']}",
        f"Endpoint errors: {summary['endpoint_errors']}",
        f"Specialist-use rate: {summary['specialist_use_rate']:.4%}",
        f"Fallback-use rate: {summary['fallback_use_rate']:.4%}",
        f"Average latency: {summary['average_latency_ms']:.3f} ms",
        f"P95 latency: {summary['p95_latency_ms']:.3f} ms",
        f"Total wall-clock duration: {summary['total_wall_clock_seconds']:.3f} seconds",
        "", "Per-task exact accuracy:",
    ]
    for task in TASKS:
        item = summary["by_type"][task]
        lines.append(f"  {task}: {item['correct']}/{item['samples']} ({item['accuracy']:.4%}); coverage={item['coverage']:.4%}")
    lines.extend([
        "", "Count metrics:",
        f"  Exact accuracy: {count['exact_count_accuracy']:.4%}",
        f"  MAE (numeric pairs only): {count['mean_absolute_error']}",
        f"  Within ±1 / ±2 / ±5: {count['within_1']} / {count['within_2']} / {count['within_5']}",
        "", "Majority baselines:",
    ])
    for task in TASKS:
        item = baselines[task]
        lines.append(f"  {task}: {item['label']} ({item['accuracy']:.4%})")
    return "\n".join(lines) + "\n"


def write_report(
    config: FullBenchmarkConfig,
    summary: Mapping[str, Any],
    count: Mapping[str, Any],
    baselines: Mapping[str, Any],
    validation: Mapping[str, Any],
    error_analysis: Mapping[str, Any],
    smoke_references: Mapping[str, Any],
    unique_images: int,
) -> str:
    validation_tasks = (validation.get("final_validation") or {}).get("per_task_accuracy") or {}
    lines = [
        "# Official RSVQA-LR Test Benchmark — SatQuery RSVQA Specialist v1", "",
        "## Scope and reproducibility", "",
        f"This measurement uses the official RSVQA-LR test split: **{summary['total_records']:,} questions across {unique_images} images**. The model is **SatQuery RSVQA Specialist v1**. No training, routing, inference, vocabulary, preprocessing, checkpoint, or benchmark label was changed for this run.", "",
        "Architecture: OpenCLIP ViT-L-14 (`laion2b_s32b_b82k`) with the frozen SatQuery Vision Encoder v1 adapter, projected image/question features, elementwise product and absolute-difference fusion, then exported presence, comparison, rural/urban, and count heads.", "",
        f"Checkpoint SHA-256: `{CHECKPOINT_SHA256}`.", "",
        "Preprocessing reads the first three TIFF bands, applies the verified per-band 2nd–98th percentile stretch, clips to [0,1], converts to uint8 RGB PNG, and applies the OpenCLIP ViT-L-14 validation transform. Converted images are cached by image ID.", "",
        "Exact match compares normalized yes/no, rural/urban, non-negative integer strings, and the literal `201+` class. Official labels above 200 remain their exact integer values. Overflow-aware accuracy is reported separately by mapping official values above 200 to the exported `201+` policy.", "",
        "Reproduction command:", "", f"```bash\n{reproducibility_command(config)}\n```", "",
        "## Test results", "",
        "| Metric | Result |", "|---|---:|",
        f"| Exact-match accuracy | {summary['exact_match_accuracy']:.2%} |",
        f"| Overflow-aware accuracy | {summary['overflow_aware_accuracy']:.2%} |",
        f"| Macro task accuracy | {summary['macro_task_accuracy']:.2%} |",
        f"| Coverage | {summary['coverage']:.2%} |",
        f"| Null answers | {summary['null_answers']} |",
        f"| Endpoint errors | {summary['endpoint_errors']} |",
        f"| Specialist-use rate | {summary['specialist_use_rate']:.2%} |",
        f"| Fallback-use rate | {summary['fallback_use_rate']:.2%} |", "",
        "| Task | Samples | Exact accuracy | Coverage | Majority baseline | Exceeds baseline |", "|---|---:|---:|---:|---:|:---:|",
    ]
    for task in TASKS:
        item, baseline = summary["by_type"][task], baselines[task]
        lines.append(f"| {task} | {item['samples']:,} | {item['accuracy']:.2%} | {item['coverage']:.2%} | {baseline['label']} ({baseline['accuracy']:.2%}) | {'yes' if item['accuracy'] > baseline['accuracy'] else 'no'} |")
    lines.extend(["", "## Latency", "", "| Metric | Value |", "|---|---:|", f"| Average | {summary['average_latency_ms']:.1f} ms |", f"| Median | {summary['median_latency_ms']:.1f} ms |", f"| P90 | {summary['p90_latency_ms']:.1f} ms |", f"| P95 | {summary['p95_latency_ms']:.1f} ms |", f"| P99 | {summary['p99_latency_ms']:.1f} ms |", f"| Throughput | {summary['throughput_questions_per_second']:.2f} questions/s |", f"| Wall-clock duration | {summary['total_wall_clock_seconds']:.1f} s |", "",
        "## Count analysis", "", f"Exact count accuracy is **{count['exact_count_accuracy']:.2%}** and overflow-aware count accuracy is **{count['overflow_aware_accuracy']:.2%}**. Numeric-only MAE is **{count['mean_absolute_error']}** across {count['numeric_pair_count']:,} pairs; `201+` is excluded from numeric error. Within ±1 / ±2 / ±5 is **{count['within_1']:.2%} / {count['within_2']:.2%} / {count['within_5']:.2%}**.", "",
        "Counts are learned RSVQA benchmark labels. They are not physical object inventories, segmentation measurements, or grounding results.", "",
        "## Validation and prior smoke references", "",
        "Exported model-card results below are validation metrics, not test results:", "", "| Task | Exported validation | Official test |", "|---|---:|---:|",
    ])
    for task in TASKS:
        lines.append(f"| {task} | {float(validation_tasks.get(task, 0)):.2%} | {summary['by_type'][task]['accuracy']:.2%} |")
    lines.extend(["", "The earlier heuristic and specialist Smoke-50 runs are small-set references only and are not substitutes for this full-test measurement.", ""])
    heuristic_smoke = smoke_references.get("old_heuristic_smoke50")
    specialist_smoke = smoke_references.get("specialist_smoke50")
    if heuristic_smoke and specialist_smoke:
        lines.extend([
            "| Smoke-50 system | Accuracy | Coverage | Average latency |",
            "|---|---:|---:|---:|",
            f"| Old heuristic | {heuristic_smoke['overall_accuracy']:.2%} | {heuristic_smoke['coverage']:.2%} | {heuristic_smoke['average_latency_ms']:.1f} ms |",
            f"| RSVQA Specialist v1 | {specialist_smoke['overall_accuracy']:.2%} | {specialist_smoke['coverage']:.2%} | {specialist_smoke['average_latency_ms']:.1f} ms |",
            "",
            "These rows compare the two systems on the same 50-question smoke manifest. The official 10,004-question result above must not be directly subtracted from the heuristic Smoke-50 accuracy.", "",
        ])
    lines.extend([
        "## Error analysis", "", f"Incorrect/error records: **{error_analysis['incorrect_or_error_records']:,}**. High-confidence errors (confidence ≥0.90): **{error_analysis['high_confidence_errors']:,}**. The highest-confidence error category is **{error_analysis['highest_confidence_error_category']}**.", "",
        f"Among count failures, the most common prediction is **{error_analysis['dominant_count_failure_prediction']}** ({error_analysis['dominant_count_failure_rate']:.2%}); count-class collapse flag: **{error_analysis['count_class_collapse_detected']}**.", "",
        "Detailed wording, entity, image, confidence, count-bucket, and fallback groupings are saved in the CSV and JSON artifacts. Analysis was performed only after predictions were produced.", "",
        "## Fallback behavior and limitations", "", f"The specialist answered {summary['specialist_use_rate']:.2%} of all records; fallback use was {summary['fallback_use_rate']:.2%}. Endpoint failures are retained as explicit rows and never converted into model predictions.", "",
        "Confidence values are task-head softmax scores and are not calibrated probabilities. This benchmark does not establish physical object-count accuracy, segmentation accuracy, grounding accuracy, or generalization beyond RSVQA-LR. Results apply to the saved official split, deterministic rendering pipeline, checkpoint, and runtime configuration.", ""])
    return "\n".join(lines)


def generate_outputs(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    config: FullBenchmarkConfig,
    state: dict[str, Any],
) -> dict[str, Any]:
    output = config.output_dir
    wall_seconds = float(state.get("benchmark_wall_clock_seconds") or 0.0)
    summary = aggregate_metrics(rows, wall_seconds)
    counts = count_metrics(rows)
    baselines = majority_baselines(records)
    validation = json.loads((Path(__file__).resolve().parents[1] / "models/rsvqa_specialist_v1/validation_metrics.json").read_text(encoding="utf-8"))
    smoke_references = load_smoke_references(Path(__file__).resolve().parents[1])
    error_analysis = write_error_analysis(rows, output)
    write_group_metrics(output / "metrics_by_type.csv", rows, "question_type", lambda row: row.get("question_type"))
    write_group_metrics(output / "metrics_by_wording.csv", rows, "wording_pattern", lambda row: wording_pattern(str(row.get("question"))))
    write_group_metrics(output / "metrics_by_target.csv", rows, "canonical_target", lambda row: canonical_target(str(row.get("question"))))
    confidence = write_confidence_analysis(rows, output / "confidence_analysis.csv")
    latency_values = [float(row["effective_latency_ms"]) for row in rows if row.get("completed") and row.get("effective_latency_ms") is not None]
    latency_by_type = {}
    for task in TASKS:
        task_latencies = [
            float(row["effective_latency_ms"])
            for row in rows
            if row.get("question_type") == task
            and row.get("completed")
            and row.get("effective_latency_ms") is not None
        ]
        latency_by_type[task] = statistics.fmean(task_latencies) if task_latencies else None
    atomic_json(output / "latency_analysis.json", {key: summary[key] for key in ("average_latency_ms", "median_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms", "slowest_request", "throughput_questions_per_second", "total_wall_clock_seconds")} | {"average_by_type_ms": latency_by_type})
    atomic_json(output / "count_metrics.json", counts)
    with (output / "count_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["answer", "ground_truth_count", "prediction_count"])
        writer.writeheader()
        labels = sorted(set(counts["ground_truth_distribution"]) | set(counts["prediction_distribution"]), key=lambda value: (value == "201+", int(value) if value.isdigit() else 10**9, value))
        for label in labels:
            writer.writerow({"answer": label, "ground_truth_count": counts["ground_truth_distribution"].get(label, 0), "prediction_count": counts["prediction_distribution"].get(label, 0)})
    for task, filename, labels in (
        ("presence", "confusion_presence.csv", ("no", "yes", "<null>", "<error>")),
        ("comp", "confusion_comparison.csv", ("no", "yes", "<null>", "<error>")),
        ("rural_urban", "confusion_rural_urban.csv", ("rural", "urban", "<null>", "<error>")),
    ):
        write_matrix(output / filename, [row for row in rows if row.get("question_type") == task], labels)
    count_rows = [row for row in rows if row.get("question_type") == "count"]
    top25 = [label for label, _ in Counter(str(row.get("normalized_ground_truth_exact")) for row in count_rows).most_common(25)]
    write_matrix(output / "confusion_count_top25.csv", count_rows, [*top25, "<null>", "<error>"])
    write_matrix(output / "confusion_count_buckets.csv", count_rows, COUNT_BUCKETS, bucketed=True)

    asset_dir = output / "report_assets"
    write_bar_chart(asset_dir, "overall_accuracy", "Official RSVQA-LR test: accuracy and coverage", ["Exact accuracy", "Overflow-aware", "Coverage"], [summary["exact_match_accuracy"], summary["overflow_aware_accuracy"], summary["coverage"]])
    write_bar_chart(asset_dir, "per_task_accuracy", "Exact-match accuracy by task", list(TASKS), [summary["by_type"][task]["accuracy"] for task in TASKS])
    validation_tasks = validation["final_validation"]["per_task_accuracy"]
    chart_labels, chart_values = [], []
    for task in TASKS:
        chart_labels.extend([f"{task} val", f"{task} test"]); chart_values.extend([validation_tasks[task], summary["by_type"][task]["accuracy"]])
    write_bar_chart(asset_dir, "validation_vs_test", "Exported validation vs official test accuracy", chart_labels, chart_values)
    write_histogram(asset_dir, "latency_distribution", "Endpoint latency distribution (ms)", latency_values)
    write_bar_chart(asset_dir, "confidence_calibration_style", "Observed accuracy by softmax-confidence bin", [item["confidence_bin"] for item in confidence], [float(item["accuracy"] or 0) for item in confidence])
    bucket_items = counts["accuracy_by_ground_truth_bucket"]
    write_bar_chart(asset_dir, "count_performance_by_bucket", "Exact count accuracy by ground-truth bucket", list(bucket_items), [float(item["accuracy"] or 0) for item in bucket_items.values()])
    write_bar_chart(asset_dir, "correct_incorrect_distribution", "Exact result distribution", ["Correct", "Incorrect", "Endpoint error"], [float(summary["exact_correct"]), float(summary["total_records"] - summary["exact_correct"] - summary["endpoint_errors"]), float(summary["endpoint_errors"])], percent=False)

    write_predictions(output / "predictions.csv", rows)
    atomic_json(output / "errors.json", errors_for(rows))
    results = {
        "benchmark": "Official RSVQA-LR complete test benchmark",
        "dataset_split": "official test",
        "model": MODEL_USED,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "manifest": str(config.manifest),
        "manifest_sha256": sha256_file(config.manifest),
        "image_root": str(config.image_root),
        "unique_test_images": len({str(record["image_id"]) for record in records}),
        "summary": summary,
        "count_metrics": counts,
        "majority_baselines": baselines,
        "validation_metrics_reference": validation,
        "smoke50_references": smoke_references,
        "error_analysis": error_analysis,
        "warmups": state.get("warmups", []),
        "validation_gates": {},
        "reproducibility_command": reproducibility_command(config),
    }
    task_counts_manifest = Counter(str(record["question_type"]) for record in records)
    task_counts_rows = Counter(str(row.get("question_type")) for row in rows)
    gates = {
        "expected_questions": len(records) == (config.limit or EXPECTED_QUESTIONS),
        "official_manifest_questions": EXPECTED_QUESTIONS,
        "processed_records": len(rows) == len(records),
        "duplicate_question_ids": len(rows) - len({str(row.get("question_id")) for row in rows}),
        "missing_question_ids": len({str(record["question_id"]) for record in records} - {str(row.get("question_id")) for row in rows}),
        "all_model_used_present": all(bool(row.get("model_used")) for row in rows),
        "task_family_counts_match": task_counts_manifest == task_counts_rows,
        "ground_truth_sent_to_endpoint": False,
        "question_type_sent_to_model": False,
        "preprocessing": "smoke benchmark first-three-band per-band 2nd-98th percentile stretch",
    }
    results["validation_gates"] = gates
    atomic_json(output / "results.json", results)
    atomic_text(output / "summary.txt", write_summary(summary, counts, baselines))
    atomic_text(output / "benchmark_report.md", write_report(config, summary, counts, baselines, validation, error_analysis, smoke_references, results["unique_test_images"]))
    state.update(status="complete", completed_at=utc_now(), validation_gates=gates)
    atomic_json(config.run_state, state)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or args.checkpoint_every <= 0 or args.max_retries < 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("timeout/checkpoint-every must be positive; max-retries non-negative; limit positive")
    config = FullBenchmarkConfig(
        manifest=args.manifest.expanduser().resolve(), image_root=args.image_root.expanduser().resolve(),
        endpoint=smoke.normalize_endpoint(args.endpoint), output_dir=args.output_dir.expanduser().resolve(),
        timeout=args.timeout, resume=args.resume, checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries, limit=args.limit,
    )
    try:
        all_records = load_manifest(config.manifest, config.image_root)
        records = all_records[:config.limit] if config.limit else all_records
        with requests.Session() as session:
            health = check_backend_health(config, session)
            print(f"Backend health: {health.get('status')} ({health.get('module')})", flush=True)
            warmups = warm_up(all_records, config, session)
            rows, state = run_records(records, config, session=session)
        state["warmups"] = warmups
        results = generate_outputs(records, rows, config, state)
    except KeyboardInterrupt:
        print("Benchmark interrupted; durable partial artifacts were saved. Resume with --resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, csv.Error, requests.RequestException) as error:
        print(f"Full benchmark failed validation/setup: {error}", file=sys.stderr)
        return 2
    summary = results["summary"]
    print(f"Completed {summary['total_records']}/{len(records)}; exact={summary['exact_match_accuracy']:.2%}; coverage={summary['coverage']:.2%}; errors={summary['endpoint_errors']}")
    print(f"Artifacts: {config.output_dir}")
    return 0 if summary["endpoint_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
