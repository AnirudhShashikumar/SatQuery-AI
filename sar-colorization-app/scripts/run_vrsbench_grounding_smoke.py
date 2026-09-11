#!/usr/bin/env python3
"""Benchmark the unchanged SatQuery Grounding DINO path on VRSBench Smoke-100."""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satquery_agent.grounding_evaluation import (  # noqa: E402
    GroundingEvaluationSample,
    LocalGroundingDinoCandidateProvider,
)
from satquery_agent.models import ImageFormat, ImageMetadata  # noqa: E402
from satquery_agent.specialists.grounder import (  # noqa: E402
    CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_RELIABILITY_POLICY,
    GroundingConfig,
    RemoteSensingGrounder,
    evaluate_detection_quality,
)
from scripts.run_rsvqa_full import atomic_json, atomic_text, percentile, write_bar_chart, write_histogram  # noqa: E402


EXPECTED_SAMPLES = 100
EXPECTED_CLASSES = 26
EXPECTED_CLIPPED = 40
DEFAULT_MANIFEST = Path("datasets/VRSBench/grounding_smoke100/vrsbench_grounding_smoke100.jsonl")
DEFAULT_IMAGE_ROOT = Path("datasets/VRSBench/grounding_smoke100/images")
DEFAULT_OUTPUT = Path("artifacts/vrsbench_grounding_smoke_baseline")
DEFAULT_ENDPOINT = "in-process://satquery/rs_grounder"
IOU_THRESHOLDS = (0.25, 0.50, 0.75)
FAILURE_CATEGORIES = (
    "null_prediction", "low_overlap", "wrong_instance", "oversized_box", "undersized_box",
    "class_confusion", "phrase_relation_failure", "boundary_clipping_case", "unknown",
)
PREDICTION_FIELDS = (
    "manifest_index", "smoke_id", "image", "image_path", "object_class", "referring_sentence",
    "ground_truth_bbox", "bbox_was_clipped", "object_position", "relative_position", "object_size",
    "relative_size", "is_unique", "completed", "error", "latency_ms", "predicted_box_count",
    "predicted_boxes_json", "best_prediction_index", "best_prediction_bbox", "best_prediction_label",
    "best_prediction_score", "best_iou", "highest_confidence_index", "highest_confidence_score",
    "model_reused", "device", "checkpoint", "failure_category",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest: Path
    image_root: Path
    endpoint: str
    output_dir: Path
    timeout: float = 120.0
    resume: bool = False
    limit: Optional[int] = None
    box_threshold: float = DEFAULT_CONFIG.box_threshold
    text_threshold: float = DEFAULT_CONFIG.text_threshold

    @property
    def partial_predictions(self) -> Path:
        return self.output_dir / "predictions.partial.csv"

    @property
    def run_state(self) -> Path:
        return self.output_dir / "run_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalized_xyxy(box: Sequence[Any], width: int, height: int, *, already_normalized: bool = False) -> list[float]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("A valid xyxy box and positive image dimensions are required")
    values = [finite_number(value, "box coordinate") for value in box]
    if not already_normalized:
        values = [values[0] / width, values[1] / height, values[2] / width, values[3] / height]
    left, top, right, bottom = [min(1.0, max(0.0, value)) for value in values]
    if right <= left or bottom <= top:
        raise ValueError("Box must have positive normalized area")
    return [left, top, right, bottom]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def best_iou_prediction(predictions: Sequence[Mapping[str, Any]], ground_truth: Sequence[float]) -> Optional[dict[str, Any]]:
    if not predictions:
        return None
    ranked = []
    for index, prediction in enumerate(predictions):
        iou = box_iou(prediction["bbox_normalized"], ground_truth)
        ranked.append((iou, float(prediction["score"]), -index, index, prediction))
    iou, _score, _negative_index, index, prediction = max(ranked)
    return {**prediction, "index": index, "iou": iou}


def load_manifest(path: Path, image_root: Path) -> list[dict[str, Any]]:
    rows, identifiers = [], set()
    required = {
        "smoke_id", "image", "object_class", "referring_sentence", "ground_truth_bbox",
        "bbox_was_clipped", "object_position", "object_size", "relative_position", "is_unique",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on manifest line {line_number}") from error
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"Manifest line {line_number} lacks required fields")
            smoke_id = str(row["smoke_id"])
            if smoke_id in identifiers:
                raise ValueError(f"Duplicate smoke_id: {smoke_id}")
            identifiers.add(smoke_id)
            image_path = image_root / Path(str(row["image"])).name
            if not image_path.is_file():
                raise ValueError(f"Missing Smoke-100 image: {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            normalized_xyxy(row["ground_truth_bbox"], width, height, already_normalized=True)
            rows.append({**row, "resolved_image_path": str(image_path), "image_width": width, "image_height": height})
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"Smoke-100 must contain {EXPECTED_SAMPLES} records, found {len(rows)}")
    if len({str(row["object_class"]) for row in rows}) != EXPECTED_CLASSES:
        raise ValueError(f"Smoke-100 must contain {EXPECTED_CLASSES} classes")
    if sum(bool(row["bbox_was_clipped"]) for row in rows) != EXPECTED_CLIPPED:
        raise ValueError(f"Smoke-100 must contain {EXPECTED_CLIPPED} clipped boxes")
    return rows


def image_metadata(record: Mapping[str, Any]) -> ImageMetadata:
    path = Path(str(record["resolved_image_path"]))
    suffix = path.suffix.lower()
    image_format = ImageFormat.PNG if suffix == ".png" else ImageFormat.JPEG
    return ImageMetadata(
        file_id=str(record["smoke_id"]), original_name=path.name, safe_name=path.name,
        format=image_format, mime_type="image/png" if suffix == ".png" else "image/jpeg",
        size_bytes=path.stat().st_size, width=int(record["image_width"]), height=int(record["image_height"]),
        band_count=3, dtype="uint8", is_georeferenced=False, color_interpretation=["r", "g", "b"],
    )


class InProcessGroundingRunner:
    """Thin benchmark adapter over the repository's existing candidate provider and quality gate."""

    def __init__(self, box_threshold: float, text_threshold: float) -> None:
        config = GroundingConfig(
            box_threshold=box_threshold, text_threshold=text_threshold,
            nms_iou_threshold=DEFAULT_CONFIG.nms_iou_threshold,
            maximum_detections=DEFAULT_CONFIG.maximum_detections,
            maximum_source_dimension=DEFAULT_CONFIG.maximum_source_dimension,
        )
        self.provider = LocalGroundingDinoCandidateProvider(
            device="auto", reuse_model=True, candidate_floor=box_threshold
        )
        grounder = RemoteSensingGrounder(config=config)
        grounder.load()
        self.provider._grounder = grounder
        self.config = config
        self.inference_calls = 0
        self.reuse_count = 0

    @property
    def load_count(self) -> int:
        grounder = self.provider._grounder
        return int(grounder._load_count) if grounder is not None else 0

    def predict(self, record: Mapping[str, Any]) -> dict[str, Any]:
        grounder = self.provider._grounder
        was_ready = bool(grounder is not None and grounder.health().status == "ready")
        sentence = str(record["referring_sentence"]).strip()
        sample = GroundingEvaluationSample(
            sample_id=str(record["smoke_id"]), image_path=Path(str(record["resolved_image_path"])),
            target=sentence, query=sentence,
            boxes=(tuple(float(value) for value in record["ground_truth_bbox"]),), target_present=True,
            annotation_source=str(record.get("annotation_file") or "VRSBench"), notes=None,
            width=int(record["image_width"]), height=int(record["image_height"]),
        )
        inference = self.provider(sample)
        self.inference_calls += 1
        if was_ready:
            self.reuse_count += 1
        metadata = image_metadata(record)
        accepted, rejected = evaluate_detection_quality(
            boxes=[candidate.box for candidate in inference.candidates],
            scores=[candidate.score for candidate in inference.candidates],
            labels=[candidate.label for candidate in inference.candidates],
            metadata=metadata,
            target_phrase=str(record["object_class"]).strip().lower(),
            config=self.config,
            policy=DEFAULT_RELIABILITY_POLICY,
        )
        predictions = [
            {
                "bbox_normalized": normalized_xyxy(item.bbox_pixels, metadata.width, metadata.height),
                "bbox_pixels": list(item.bbox_pixels), "score": float(item.score), "label": item.label,
            }
            for item in accepted
        ]
        return {
            "predictions": predictions, "raw_candidate_count": len(inference.candidates),
            "rejected_candidate_count": len(rejected), "model_reused": was_ready,
            "device": inference.device, "checkpoint": inference.checkpoint,
        }


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def label_matches_class(label: str, object_class: str) -> bool:
    label_text, class_text = _normalized_text(label), _normalized_text(object_class)
    aliases = {
        "airplane": ("airplane", "plane", "aircraft"), "vehicle": ("vehicle", "car", "truck", "automobile"),
        "ship": ("ship", "boat", "vessel"), "golffield": ("golf field", "golf course"),
        "windmill": ("windmill", "wind turbine"), "trainstation": ("train station", "railway station"),
    }
    candidates = aliases.get(class_text.replace(" ", ""), (class_text,))
    return any(candidate in label_text or label_text in candidate for candidate in candidates if candidate)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def failure_category(record: Mapping[str, Any], best: Optional[Mapping[str, Any]]) -> str:
    if best is None:
        return "null_prediction"
    if float(best["iou"]) >= 0.5:
        return "success"
    truth, predicted = record["ground_truth_bbox"], best["bbox_normalized"]
    if not label_matches_class(str(best.get("label") or ""), str(record["object_class"])):
        return "class_confusion"
    if bool(record["bbox_was_clipped"]):
        return "boundary_clipping_case"
    truth_area, predicted_area = box_area(truth), box_area(predicted)
    ratio = predicted_area / truth_area if truth_area > 0 else math.inf
    if ratio >= 2.0:
        return "oversized_box"
    if ratio <= 0.5:
        return "undersized_box"
    relation = bool(str(record.get("relative_position") or "").strip()) or bool(re.search(r"\b(?:closest|left|right|top|bottom|above|below|between|next to)\b", str(record["referring_sentence"]).lower()))
    if not bool(record.get("is_unique")) and relation:
        return "phrase_relation_failure"
    predicted_center = ((predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2)
    if not (truth[0] <= predicted_center[0] <= truth[2] and truth[1] <= predicted_center[1] <= truth[3]):
        return "wrong_instance" if not bool(record.get("is_unique")) else "low_overlap"
    return "low_overlap" if float(best["iou"]) < 0.5 else "unknown"


def evaluate_record(record: Mapping[str, Any], index: int, runner: InProcessGroundingRunner) -> dict[str, Any]:
    base = {
        "manifest_index": index, "smoke_id": str(record["smoke_id"]), "image": record["image"],
        "image_path": record["resolved_image_path"], "object_class": record["object_class"],
        "referring_sentence": record["referring_sentence"], "ground_truth_bbox": list(record["ground_truth_bbox"]),
        "bbox_was_clipped": bool(record["bbox_was_clipped"]), "object_position": record.get("object_position") or "",
        "relative_position": record.get("relative_position") or "", "object_size": record.get("object_size") or "",
        "relative_size": record.get("relative_size") or "", "is_unique": bool(record.get("is_unique")),
        "completed": False, "error": None, "latency_ms": None, "predicted_box_count": 0,
        "predicted_boxes": [], "best_prediction_index": None, "best_prediction_bbox": None,
        "best_prediction_label": None, "best_prediction_score": None, "best_iou": 0.0,
        "highest_confidence_index": None, "highest_confidence_score": None, "model_reused": None,
        "device": None, "checkpoint": CHECKPOINT, "failure_category": "unknown",
    }
    started = time.perf_counter()
    try:
        result = runner.predict(record)
        predictions = result["predictions"]
        truth = normalized_xyxy(record["ground_truth_bbox"], int(record["image_width"]), int(record["image_height"]), already_normalized=True)
        best = best_iou_prediction(predictions, truth)
        highest_index = max(range(len(predictions)), key=lambda item: predictions[item]["score"]) if predictions else None
        base.update(
            completed=True, predicted_box_count=len(predictions), predicted_boxes=predictions,
            best_prediction_index=best["index"] if best else None,
            best_prediction_bbox=best["bbox_normalized"] if best else None,
            best_prediction_label=best["label"] if best else None,
            best_prediction_score=best["score"] if best else None,
            best_iou=best["iou"] if best else 0.0,
            highest_confidence_index=highest_index,
            highest_confidence_score=predictions[highest_index]["score"] if highest_index is not None else None,
            model_reused=result["model_reused"], device=result["device"], checkpoint=result["checkpoint"],
            failure_category=failure_category(record, best),
        )
    except Exception as error:
        base.update(error=f"{type(error).__name__}: {error}", failure_category="unknown")
    base["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return base


def _csv_value(field: str, value: Any) -> Any:
    if field in {"ground_truth_bbox", "predicted_boxes_json", "best_prediction_bbox"}:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value


def write_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["manifest_index"]))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row in ordered:
                shaped = {field: row.get(field) for field in PREDICTION_FIELDS}
                shaped["predicted_boxes_json"] = row.get("predicted_boxes") or []
                writer.writerow({field: _csv_value(field, shaped[field]) for field in PREDICTION_FIELDS})
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        parsed: dict[str, Any] = dict(row)
        for field in ("ground_truth_bbox", "predicted_boxes_json", "best_prediction_bbox"):
            parsed[field if field != "predicted_boxes_json" else "predicted_boxes"] = json.loads(row[field]) if row[field] else ([] if field == "predicted_boxes_json" else None)
        for field in ("bbox_was_clipped", "is_unique", "completed", "model_reused"):
            parsed[field] = _boolean(row[field]) if row[field] else None
        for field in ("manifest_index", "predicted_box_count", "best_prediction_index", "highest_confidence_index"):
            parsed[field] = int(row[field]) if row[field] else None
        for field in ("latency_ms", "best_prediction_score", "best_iou", "highest_confidence_score"):
            parsed[field] = float(row[field]) if row[field] else None
        parsed["error"] = row["error"] or None
        output[str(parsed["smoke_id"])] = parsed
    if len(output) != len(rows):
        raise ValueError("Partial predictions contain duplicate smoke IDs")
    return output


def pending_records(records: Sequence[Mapping[str, Any]], completed_ids: Iterable[str]) -> list[Mapping[str, Any]]:
    completed = {str(value) for value in completed_ids}
    return [record for record in records if str(record["smoke_id"]) not in completed]


def preflight(records: Sequence[Mapping[str, Any]], runner: InProcessGroundingRunner) -> dict[str, Any]:
    synthetic = box_iou([0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75])
    if not math.isclose(synthetic, 1 / 7, rel_tol=1e-9):
        raise ValueError("Synthetic IoU validation failed")
    selected, seen = [], set()
    for record in records:
        object_class = str(record["object_class"])
        if object_class not in seen:
            selected.append(record); seen.add(object_class)
        if len(selected) == 4:
            break
    warmups = []
    accepted_box_count = 0
    for record in selected:
        started = time.perf_counter()
        result = runner.predict(record)
        accepted_box_count += len(result["predictions"])
        for prediction in result["predictions"]:
            normalized_xyxy(prediction["bbox_normalized"], 1, 1, already_normalized=True)
        warmups.append({
            "smoke_id": str(record["smoke_id"]), "object_class": record["object_class"],
            "accepted_boxes": len(result["predictions"]), "coordinates_normalized": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3), "model_reused": result["model_reused"],
        })
    resume_ok = [str(item["smoke_id"]) for item in pending_records(records[:2], {str(records[0]["smoke_id"])})] == [str(records[1]["smoke_id"])]
    if runner.load_count != 1 or not resume_ok or accepted_box_count == 0:
        raise ValueError("Model single-load or resume validation failed")
    return {
        "warmups": warmups, "synthetic_iou": synthetic, "model_load_count": runner.load_count,
        "normalized_coordinate_check": True, "normalized_accepted_box_count": accepted_box_count,
        "resume_check": resume_ok,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    total = len(rows)
    completed = [row for row in rows if row.get("completed")]
    non_null = [row for row in completed if int(row.get("predicted_box_count") or 0) > 0]
    ious = [float(row.get("best_iou") or 0.0) for row in rows]
    non_null_ious = [float(row.get("best_iou") or 0.0) for row in non_null]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    confidences = [float(row["best_prediction_score"]) for row in non_null if row.get("best_prediction_score") is not None]
    return {
        "samples": total, "completed_samples": len(completed), "inference_errors": total - len(completed),
        "null_predictions": len(completed) - len(non_null),
        "null_prediction_rate": (len(completed) - len(non_null)) / total if total else 0.0,
        "mean_iou_all": statistics.fmean(ious) if ious else 0.0,
        "mean_iou_non_null": statistics.fmean(non_null_ious) if non_null_ious else None,
        "median_iou": statistics.median(ious) if ious else 0.0,
        **{f"accuracy_at_{threshold:.2f}": sum(iou >= threshold for iou in ious) / total if total else 0.0 for threshold in IOU_THRESHOLDS},
        "average_predicted_boxes": statistics.fmean(float(row.get("predicted_box_count") or 0) for row in rows) if rows else 0.0,
        "average_confidence": statistics.fmean(confidences) if confidences else None,
        "average_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95),
        "total_runtime_seconds": runtime_seconds,
        "throughput_samples_per_second": total / runtime_seconds if runtime_seconds > 0 else None,
    }


def group_metrics(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda item: item[0]):
        metrics = aggregate(values, 0.0)
        output.append({
            "group": group, "samples": metrics["samples"], "mean_iou": metrics["mean_iou_all"],
            "accuracy_at_0.25": metrics["accuracy_at_0.25"], "accuracy_at_0.50": metrics["accuracy_at_0.50"],
            "accuracy_at_0.75": metrics["accuracy_at_0.75"], "null_rate": metrics["null_prediction_rate"],
            "average_boxes": metrics["average_predicted_boxes"], "average_confidence": metrics["average_confidence"],
        })
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_horizontal_chart(asset_dir: Path, name: str, title: str, labels: Sequence[str], values: Sequence[float], *, percent: bool = True) -> None:
    width, row_height = 1400, 38
    height = max(420, 120 + len(labels) * row_height)
    left, right, top = 310, 80, 80
    plot_width = width - left - right
    maximum = 1.0 if percent else max(values or [1.0]) or 1.0
    font = ImageFont.load_default(size=18)
    heading = ImageFont.load_default(size=24)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 24), title, fill="#172b4d", font=heading)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="45" font-family="sans-serif" font-size="25" fill="#172b4d">{title}</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_height
        bar_width = plot_width * float(value) / maximum
        draw.text((10, y + 6), str(label)[:38], fill="#172b4d", font=font)
        draw.rounded_rectangle((left, y + 3, left + bar_width, y + 29), radius=5, fill="#1677ff")
        display = f"{value:.1%}" if percent else f"{value:.2f}"
        draw.text((left + bar_width + 8, y + 6), display, fill="#172b4d", font=font)
        safe = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        svg.extend([f'<text x="10" y="{y + 23}" font-family="sans-serif" font-size="17" fill="#172b4d">{safe}</text>', f'<rect x="{left}" y="{y + 3}" width="{bar_width:.2f}" height="26" rx="5" fill="#1677ff"/>', f'<text x="{left + bar_width + 8:.2f}" y="{y + 23}" font-family="sans-serif" font-size="17" fill="#172b4d">{display}</text>'])
    svg.append("</svg>")
    asset_dir.mkdir(parents=True, exist_ok=True)
    image.save(asset_dir / f"{name}.png", "PNG")
    atomic_text(asset_dir / f"{name}.svg", "\n".join(svg) + "\n")


def recommendation(summary: Mapping[str, Any]) -> dict[str, str]:
    acc25, acc50 = float(summary["accuracy_at_0.25"]), float(summary["accuracy_at_0.50"])
    null_rate = float(summary["null_prediction_rate"])
    if acc50 >= 0.60 and acc25 >= 0.75 and null_rate <= 0.10:
        return {"choice": "A", "label": "current Grounding DINO is sufficient"}
    if acc50 >= 0.35 or (acc25 >= 0.55 and null_rate >= 0.15):
        return {"choice": "B", "label": "threshold tuning first"}
    return {"choice": "C", "label": "fine-tune/replace grounding specialist"}


def reproducibility_command(config: BenchmarkConfig) -> str:
    parts = [
        "venv/bin/python", "scripts/run_vrsbench_grounding_smoke.py", "--manifest", str(config.manifest),
        "--image-root", str(config.image_root), "--endpoint", config.endpoint, "--output-dir", str(config.output_dir),
        "--timeout", f"{config.timeout:g}",
    ]
    if config.resume:
        parts.append("--resume")
    if config.limit is not None:
        parts.extend(("--limit", str(config.limit)))
    if config.box_threshold != DEFAULT_CONFIG.box_threshold:
        parts.extend(("--box-threshold", str(config.box_threshold)))
    if config.text_threshold != DEFAULT_CONFIG.text_threshold:
        parts.extend(("--text-threshold", str(config.text_threshold)))
    return shlex.join(parts)


def finalize(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], config: BenchmarkConfig, state: dict[str, Any]) -> dict[str, Any]:
    output = config.output_dir
    summary = aggregate(rows, float(state["benchmark_runtime_seconds"]))
    by_class = group_metrics(rows, lambda row: row["object_class"])
    clipped = group_metrics(rows, lambda row: "clipped" if row["bbox_was_clipped"] else "unclipped")
    positions = []
    for dimension in ("object_position", "relative_position"):
        for item in group_metrics(rows, lambda row, key=dimension: row.get(key) or "unspecified"):
            positions.append({"dimension": dimension, **item})
    sizes = []
    for dimension in ("object_size", "relative_size"):
        for item in group_metrics(rows, lambda row, key=dimension: row.get(key) or "unspecified"):
            sizes.append({"dimension": dimension, **item})
    uniqueness = group_metrics(rows, lambda row: "unique" if row["is_unique"] else "non_unique")
    common_fields = ("group", "samples", "mean_iou", "accuracy_at_0.25", "accuracy_at_0.50", "accuracy_at_0.75", "null_rate", "average_boxes", "average_confidence")
    # Rename the generic group field for the class-specific consumer.
    with (output / "metrics_by_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("object_class", *common_fields[1:])); writer.writeheader()
        for item in by_class: writer.writerow({"object_class": item["group"], **{key: item[key] for key in common_fields[1:]}})
    write_csv(output / "metrics_by_position.csv", positions, ("dimension", *common_fields))
    write_csv(output / "metrics_by_size.csv", sizes, ("dimension", *common_fields))
    write_csv(output / "metrics_by_uniqueness.csv", uniqueness, common_fields)
    write_csv(output / "clipped_vs_unclipped.csv", clipped, common_fields)
    taxonomy = [{
        "smoke_id": row["smoke_id"], "object_class": row["object_class"], "failure_category": row["failure_category"],
        "best_iou": row["best_iou"], "prediction_label": row.get("best_prediction_label"),
        "prediction_score": row.get("best_prediction_score"), "bbox_was_clipped": row["bbox_was_clipped"],
        "is_unique": row["is_unique"], "predicted_box_count": row["predicted_box_count"],
    } for row in rows]
    write_csv(output / "confusion_or_failure_taxonomy.csv", taxonomy, tuple(taxonomy[0]))
    write_predictions(output / "predictions.csv", rows)
    atomic_json(output / "errors.json", [{"smoke_id": row["smoke_id"], "error": row["error"]} for row in rows if row.get("error")])

    assets = output / "report_assets"
    ious = [float(row.get("best_iou") or 0.0) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    write_histogram(assets, "iou_distribution", "Best-IoU distribution (null = 0)", ious)
    write_bar_chart(assets, "accuracy_at_thresholds", "Grounding accuracy at IoU thresholds", ["IoU@0.25", "IoU@0.50", "IoU@0.75"], [summary["accuracy_at_0.25"], summary["accuracy_at_0.50"], summary["accuracy_at_0.75"]])
    ordered_class = sorted(by_class, key=lambda item: (-item["accuracy_at_0.50"], item["group"]))
    write_horizontal_chart(assets, "per_class_accuracy_at_0_50", "Per-class accuracy@0.50", [item["group"] for item in ordered_class], [item["accuracy_at_0.50"] for item in ordered_class])
    write_horizontal_chart(assets, "null_rate_by_class", "Null-prediction rate by class", [item["group"] for item in ordered_class], [item["null_rate"] for item in ordered_class])
    write_histogram(assets, "latency_distribution", "In-process grounding latency (ms)", latencies)
    write_bar_chart(assets, "clipped_vs_unclipped", "Clipped vs unclipped accuracy@0.50", [item["group"] for item in clipped], [item["accuracy_at_0.50"] for item in clipped])

    rec = recommendation(summary)
    results = {
        "benchmark": "VRSBench Grounding Smoke-100 baseline", "model": CHECKPOINT,
        "inference_path": "existing LocalGroundingDinoCandidateProvider plus production evaluate_detection_quality",
        "thresholds": {"box": config.box_threshold, "text": config.text_threshold, "reliability_score": DEFAULT_RELIABILITY_POLICY.minimum_alignment_score, "nms_iou": DEFAULT_CONFIG.nms_iou_threshold},
        "summary": summary, "per_class": by_class, "clipped_vs_unclipped": clipped,
        "position_analysis": positions, "size_analysis": sizes, "uniqueness_analysis": uniqueness,
        "failure_taxonomy": {
            category: Counter(str(row["failure_category"]) for row in rows)[category]
            for category in (*FAILURE_CATEGORIES, "success")
        },
        "preflight": state["preflight"], "model_load_count": state["model_load_count"],
        "model_reuse_count": state["model_reuse_count"], "measured_model_reuse_count": sum(bool(row.get("model_reused")) for row in rows),
        "recommendation": rec, "reproducibility_command": reproducibility_command(config),
        "validation_gates": {
            "expected_samples": len(records) == (config.limit or EXPECTED_SAMPLES), "processed_samples": len(rows) == len(records),
            "duplicate_smoke_ids": len(rows) - len({str(row["smoke_id"]) for row in rows}),
            "missing_smoke_ids": len({str(record["smoke_id"]) for record in records} - {str(row["smoke_id"]) for row in rows}),
            "class_count": len(by_class), "clipped_samples": sum(bool(row["bbox_was_clipped"]) for row in rows),
            "model_loaded_once": state["model_load_count"] == 1, "normalized_boxes": all(all(0 <= value <= 1 for value in prediction["bbox_normalized"]) for row in rows for prediction in row.get("predicted_boxes", [])),
            "production_inference_modified": False,
        },
    }
    atomic_json(output / "results.json", results)
    best = sorted(by_class, key=lambda item: (-item["accuracy_at_0.50"], -item["mean_iou"], item["group"]))[:5]
    worst = sorted(by_class, key=lambda item: (item["accuracy_at_0.50"], item["mean_iou"], item["group"]))[:5]
    clipped_map = {item["group"]: item for item in clipped}
    lines = [
        "VRSBench Grounding Smoke-100 — SatQuery baseline", "=" * 54,
        f"Completed: {summary['completed_samples']}/{summary['samples']}", f"Mean IoU (all): {summary['mean_iou_all']:.6f}",
        f"Mean IoU (non-null): {summary['mean_iou_non_null']}", f"Median IoU: {summary['median_iou']:.6f}",
        f"Accuracy@0.25/@0.50/@0.75: {summary['accuracy_at_0.25']:.4%} / {summary['accuracy_at_0.50']:.4%} / {summary['accuracy_at_0.75']:.4%}",
        f"Null rate: {summary['null_prediction_rate']:.4%}", f"Average latency: {summary['average_latency_ms']:.3f} ms",
        f"P95 latency: {summary['p95_latency_ms']:.3f} ms", f"Runtime: {summary['total_runtime_seconds']:.3f} s",
        f"Model loads/reuses: {state['model_load_count']}/{state['model_reuse_count']}", f"Recommendation: {rec['choice']}. {rec['label']}",
    ]
    atomic_text(output / "summary.txt", "\n".join(lines) + "\n")
    report = [
        "# VRSBench Grounding Smoke-100 — SatQuery Grounding DINO baseline", "",
        "This is a measurement-only run of the unchanged `IDEA-Research/grounding-dino-tiny` checkpoint. Full VRSBench referring sentences were passed to the repository's existing in-process candidate provider. The current processor thresholds, reliability gate, NMS, and detection cap were retained.", "",
        f"Thresholds: box **{config.box_threshold}**, text **{config.text_threshold}**, reliability score **{DEFAULT_RELIABILITY_POLICY.minimum_alignment_score}**, NMS IoU **{DEFAULT_CONFIG.nms_iou_threshold}**. Model loads: **{state['model_load_count']}**; reuses including preflight: **{state['model_reuse_count']}**.", "",
        "## Results", "", "| Metric | Value |", "|---|---:|",
        f"| Completed | {summary['completed_samples']}/{summary['samples']} |", f"| Mean IoU, null=0 | {summary['mean_iou_all']:.4f} |",
        f"| Non-null mean IoU | {summary['mean_iou_non_null'] if summary['mean_iou_non_null'] is not None else 'n/a'} |",
        f"| Accuracy@0.25 | {summary['accuracy_at_0.25']:.2%} |", f"| Accuracy@0.50 | {summary['accuracy_at_0.50']:.2%} |",
        f"| Accuracy@0.75 | {summary['accuracy_at_0.75']:.2%} |", f"| Null rate | {summary['null_prediction_rate']:.2%} |",
        f"| Average / P95 latency | {summary['average_latency_ms']:.1f} / {summary['p95_latency_ms']:.1f} ms |", f"| Runtime | {summary['total_runtime_seconds']:.1f} s |", "",
        "The benchmark prediction is always the accepted box with highest ground-truth IoU, not necessarily the highest-confidence box. This oracle matching is evaluation-only.", "",
        "## Class extremes", "", "Best five by accuracy@0.50: " + ", ".join(f"{item['group']} ({item['accuracy_at_0.50']:.0%})" for item in best) + ".", "",
        "Worst five by accuracy@0.50: " + ", ".join(f"{item['group']} ({item['accuracy_at_0.50']:.0%})" for item in worst) + ".", "",
        "## Clipping", "", f"Clipped accuracy@0.50: **{clipped_map['clipped']['accuracy_at_0.50']:.2%}**; unclipped: **{clipped_map['unclipped']['accuracy_at_0.50']:.2%}**. Detailed mean IoU and null rates are in `clipped_vs_unclipped.csv`.", "",
        "## Failure taxonomy", "", "Categories use only returned labels/scores, box geometry, clipping metadata, uniqueness, and explicit relation metadata. They are diagnostic buckets, not claims about hidden model cognition.", "",
        "## Recommendation", "", f"**{rec['choice']}. {rec['label']}** under the documented rule encoded in the evaluator. Threshold tuning cannot be claimed to improve results until a separate, explicitly authorized threshold sweep is run.", "",
        "## Limitations and reproducibility", "", "Scores are text-region alignment scores, not calibrated confidence. Smoke-100 is small and does not establish general grounding performance. Best-IoU matching is optimistic when several boxes are returned. Clipped annotations can constrain achievable overlap. No training, threshold change, or production behavior modification occurred.", "",
        "```bash", reproducibility_command(config), "```", "",
    ]
    atomic_text(output / "benchmark_report.md", "\n".join(report))
    state.update(status="complete", completed_at=utc_now(), validation_gates=results["validation_gates"])
    atomic_json(config.run_state, state)
    return results


def run(records: Sequence[Mapping[str, Any]], config: BenchmarkConfig, runner: InProcessGroundingRunner, preflight_result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_predictions(config.partial_predictions) if config.resume else {}
    existing = {key: row for key, row in existing.items() if key in {str(item["smoke_id"]) for item in records}}
    state = {
        "status": "running", "started_at": utc_now(), "manifest": str(config.manifest), "image_root": str(config.image_root),
        "endpoint": config.endpoint, "expected_samples": len(records), "resume": config.resume,
        "box_threshold": config.box_threshold, "text_threshold": config.text_threshold, "preflight": preflight_result,
        "command": reproducibility_command(config),
    }
    started = time.perf_counter()
    try:
        for index, record in enumerate(records, 1):
            smoke_id = str(record["smoke_id"])
            if smoke_id in existing:
                continue
            existing[smoke_id] = evaluate_record(record, index, runner)
            write_predictions(config.partial_predictions, existing.values())
            state.update(
                updated_at=utc_now(), processed_samples=len(existing),
                completed_samples=sum(bool(row.get("completed")) for row in existing.values()),
                null_predictions=sum(bool(row.get("completed")) and not int(row.get("predicted_box_count") or 0) for row in existing.values()),
            )
            atomic_json(config.run_state, state)
            row = existing[smoke_id]
            print(f"[{len(existing):03d}/{len(records)}] class={record['object_class']} boxes={row['predicted_box_count']} IoU={float(row['best_iou']):.3f}", flush=True)
    except KeyboardInterrupt:
        state["status"] = "interrupted"; atomic_json(config.run_state, state)
        raise
    state.update(
        status="predictions_complete", benchmark_runtime_seconds=round(time.perf_counter() - started, 3),
        model_load_count=runner.load_count, model_reuse_count=runner.reuse_count,
    )
    atomic_json(config.run_state, state)
    return sorted(existing.values(), key=lambda row: int(row["manifest_index"])), state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Provenance label; this evaluator uses the existing in-process grounder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0, help="Reserved for endpoint compatibility")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--text-threshold", type=float)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    box_threshold = DEFAULT_CONFIG.box_threshold if args.box_threshold is None else args.box_threshold
    text_threshold = DEFAULT_CONFIG.text_threshold if args.text_threshold is None else args.text_threshold
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0) or not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise SystemExit("timeout/limit must be positive and thresholds must be between 0 and 1")
    config = BenchmarkConfig(
        manifest=args.manifest.expanduser().resolve(), image_root=args.image_root.expanduser().resolve(),
        endpoint=args.endpoint, output_dir=args.output_dir.expanduser().resolve(), timeout=args.timeout,
        resume=args.resume, limit=args.limit, box_threshold=box_threshold, text_threshold=text_threshold,
    )
    try:
        all_records = load_manifest(config.manifest, config.image_root)
        records = all_records[:config.limit] if config.limit else all_records
        runner = InProcessGroundingRunner(config.box_threshold, config.text_threshold)
        validation = preflight(all_records, runner)
        print(f"Preflight: 4 classes; model_load_count={runner.load_count}; normalized_boxes=true; synthetic_iou={validation['synthetic_iou']:.6f}; resume=true", flush=True)
        rows, state = run(records, config, runner, validation)
        results = finalize(records, rows, config, state)
    except KeyboardInterrupt:
        print("Interrupted; resume with --resume.", file=sys.stderr); return 130
    except (OSError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr); return 2
    summary = results["summary"]
    print(f"Completed {summary['completed_samples']}/{summary['samples']}; mean_iou={summary['mean_iou_all']:.4f}; acc@0.50={summary['accuracy_at_0.50']:.2%}; null={summary['null_prediction_rate']:.2%}")
    return 0 if summary["inference_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
