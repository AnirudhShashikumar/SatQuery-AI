#!/usr/bin/env python3
"""Evaluate SatQuery Grounding Specialist v1 step-100 on VRSBench Smoke-100."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
import torch
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satquery_agent.specialists.grounder import CHECKPOINT as BASE_CHECKPOINT  # noqa: E402
from satquery_agent.specialists.grounder import RemoteSensingGrounder  # noqa: E402
from scripts.run_rsvqa_full import atomic_json, atomic_text, percentile, write_bar_chart, write_histogram  # noqa: E402
from scripts.run_vrsbench_grounding_smoke import (  # noqa: E402
    DEFAULT_IMAGE_ROOT, DEFAULT_MANIFEST, EXPECTED_CLASSES, EXPECTED_CLIPPED, EXPECTED_SAMPLES,
    load_manifest, normalized_xyxy, write_horizontal_chart,
)


DEFAULT_CHECKPOINT = Path("models/grounding_specialist_v1_1/grounding_specialist_v1_1_head.pt")
DEFAULT_OUTPUT = Path("artifacts/vrsbench_grounding_smoke_specialist_step100")
DEFAULT_BASELINE_RESULTS = Path("artifacts/vrsbench_grounding_smoke_baseline/results.json")
DEFAULT_BASELINE_PREDICTIONS = Path("artifacts/vrsbench_grounding_smoke_baseline/predictions.csv")
IOU_THRESHOLDS = (0.25, 0.50, 0.75)
AREA_BUCKETS = ((0.0, 0.01, "tiny"), (0.01, 0.05, "small"), (0.05, 0.20, "medium"), (0.20, 0.50, "large"), (0.50, math.inf, "very_large"))
PREDICTION_FIELDS = (
    "manifest_index", "smoke_id", "image", "image_path", "object_class", "referring_sentence",
    "ground_truth_bbox", "bbox_was_clipped", "completed", "error", "selected_query_index",
    "query_logit", "confidence", "base_box_cxcywh", "refined_box_cxcywh", "predicted_bbox",
    "iou", "predicted_box_area", "ground_truth_box_area", "area_ratio", "oversized", "undersized",
    "latency_ms", "device", "base_checkpoint", "specialist_checkpoint_sha256",
)


class SatQueryGroundingHead(nn.Module):
    """The exact exported pilot architecture; the Grounding DINO base stays separate and frozen."""

    def __init__(self, hidden_dim: int = 256, intermediate_dim: int = 256, max_box_delta: float = 0.15) -> None:
        super().__init__()
        self.max_box_delta = float(max_box_delta)
        self.query_scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, intermediate_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(intermediate_dim, 1),
        )
        self.box_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, intermediate_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(intermediate_dim, 4), nn.Tanh(),
        )

    def forward(self, hidden_state: torch.Tensor, base_boxes_cxcywh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_logits = self.query_scorer(hidden_state).squeeze(-1)
        refined = base_boxes_cxcywh + self.box_refiner(hidden_state) * self.max_box_delta
        centers = refined[..., :2].clamp(0.0, 1.0)
        sizes = refined[..., 2:].clamp(1e-4, 1.0)
        return query_logits, torch.cat((centers, sizes), dim=-1)


@dataclass(frozen=True)
class SpecialistConfig:
    checkpoint: Path
    manifest: Path
    image_root: Path
    output_dir: Path
    device: str = "auto"
    batch_size: int = 1
    resume: bool = False
    limit: Optional[int] = None

    @property
    def partial_predictions(self) -> Path:
        return self.output_dir / "predictions.partial.csv"

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


def load_specialist_checkpoint(path: Path, device: torch.device) -> tuple[SatQueryGroundingHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("specialist_state_dict"), dict):
        raise ValueError("Checkpoint lacks specialist_state_dict")
    if payload.get("base_model") != BASE_CHECKPOINT:
        raise ValueError(f"Checkpoint base_model must be {BASE_CHECKPOINT}")
    head = SatQueryGroundingHead()
    head.load_state_dict(payload["specialist_state_dict"], strict=True)
    head.to(device).eval()
    return head, payload


def sanitize_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    centers = boxes[..., :2].clamp(0.0, 1.0)
    sizes = boxes[..., 2:].clamp(1e-4, 1.0)
    return torch.cat((centers, sizes), dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    boxes = sanitize_cxcywh(boxes)
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    converted = torch.stack(
        (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2),
        dim=-1,
    )
    return converted.clamp(0.0, 1.0)


def paired_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp(min=0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp(min=0).prod(dim=-1)
    union = first_area + second_area - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def select_highest_logit(query_logits: torch.Tensor, refined_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_logits.ndim != 2 or refined_boxes.ndim != 3 or refined_boxes.shape[:2] != query_logits.shape:
        raise ValueError("Expected logits [batch, queries] and boxes [batch, queries, 4]")
    indices = query_logits.argmax(dim=1)
    batch_indices = torch.arange(query_logits.shape[0], device=query_logits.device)
    return indices, query_logits[batch_indices, indices], refined_boxes[batch_indices, indices]


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def area_bucket(area: float) -> str:
    return next(label for low, high, label in AREA_BUCKETS if low <= area < high)


def query_index_analysis(indices: Sequence[int], predicted_areas: Sequence[float]) -> dict[str, Any]:
    total = len(indices)
    frequencies = Counter(int(index) for index in indices)
    ranked = frequencies.most_common()
    top_coverage = {
        f"top_{amount}_coverage": sum(count for _, count in ranked[:amount]) / total if total else 0.0
        for amount in (1, 5, 10)
    }
    size_frequencies = Counter(area_bucket(float(area)) for area in predicted_areas)
    dominant_size = size_frequencies.most_common(1)[0] if size_frequencies else (None, 0)
    dominant_size_rate = dominant_size[1] / total if total else 0.0
    reasons = []
    if top_coverage["top_1_coverage"] > 0.25:
        reasons.append("one_query_index_above_25_percent")
    if top_coverage["top_5_coverage"] > 0.60:
        reasons.append("top_five_query_indices_above_60_percent")
    if dominant_size_rate > 0.60:
        reasons.append("one_box_size_pattern_above_60_percent")
    return {
        "samples": total, "unique_selected_query_indices": len(frequencies),
        "most_frequent_query_indices": [{"query_index": index, "count": count, "percentage": count / total if total else 0.0} for index, count in ranked[:25]],
        **top_coverage, "box_size_pattern_distribution": dict(size_frequencies),
        "dominant_box_size_pattern": dominant_size[0], "dominant_box_size_pattern_rate": dominant_size_rate,
        "collapse_detected": bool(reasons), "collapse_reasons": reasons,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    total = len(rows)
    completed = [row for row in rows if row.get("completed")]
    ious = [float(row.get("iou") or 0.0) for row in rows]
    confidence = [float(row["confidence"]) for row in completed if row.get("confidence") is not None]
    latency = [float(row["latency_ms"]) for row in completed if row.get("latency_ms") is not None]
    return {
        "samples": total, "completed_samples": len(completed), "inference_failures": total - len(completed),
        "null_or_failure_rate": (total - len(completed)) / total if total else 0.0,
        "mean_iou": statistics.fmean(ious) if ious else 0.0, "median_iou": statistics.median(ious) if ious else 0.0,
        **{f"accuracy_at_{threshold:.2f}": sum(iou >= threshold for iou in ious) / total if total else 0.0 for threshold in IOU_THRESHOLDS},
        "average_confidence": statistics.fmean(confidence) if confidence else None,
        "average_latency_ms": statistics.fmean(latency) if latency else None,
        "median_latency_ms": statistics.median(latency) if latency else None,
        "p95_latency_ms": percentile(latency, 95), "runtime_seconds": runtime_seconds,
        "throughput_samples_per_second": total / runtime_seconds if runtime_seconds > 0 else None,
    }


def group_metrics(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    output = []
    for group, values in sorted(groups.items()):
        metrics = aggregate(values, 0.0)
        output.append({
            "group": group, "samples": len(values), "mean_iou": metrics["mean_iou"],
            "accuracy_at_0.25": metrics["accuracy_at_0.25"], "accuracy_at_0.50": metrics["accuracy_at_0.50"],
            "accuracy_at_0.75": metrics["accuracy_at_0.75"], "average_confidence": metrics["average_confidence"],
        })
    return output


def geometry_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("completed")]
    predicted = [float(row["predicted_box_area"]) for row in completed]
    truth = [float(row["ground_truth_box_area"]) for row in completed]
    return {
        "samples": len(completed),
        "average_predicted_box_area": statistics.fmean(predicted) if predicted else None,
        "average_ground_truth_box_area": statistics.fmean(truth) if truth else None,
        "oversized_prediction_rate": sum(bool(row.get("oversized")) for row in completed) / len(completed) if completed else None,
        "undersized_prediction_rate": sum(bool(row.get("undersized")) for row in completed) / len(completed) if completed else None,
        "predicted_area_bucket_distribution": dict(Counter(area_bucket(value) for value in predicted)),
        "ground_truth_area_bucket_distribution": dict(Counter(area_bucket(value) for value in truth)),
    }


def baseline_comparison(specialist: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = (
        ("mean_iou", "mean_iou", "mean_iou_all"),
        ("accuracy_at_0.25", "accuracy_at_0.25", "accuracy_at_0.25"),
        ("accuracy_at_0.50", "accuracy_at_0.50", "accuracy_at_0.50"),
        ("accuracy_at_0.75", "accuracy_at_0.75", "accuracy_at_0.75"),
        ("null_rate", "null_or_failure_rate", "null_prediction_rate"),
        ("average_latency_ms", "average_latency_ms", "average_latency_ms"),
        ("p95_latency_ms", "p95_latency_ms", "p95_latency_ms"),
    )
    return [
        {
            "metric": label, "baseline": float(baseline[baseline_key]), "specialist": float(specialist[specialist_key]),
            "absolute_change": float(specialist[specialist_key]) - float(baseline[baseline_key]),
            "relative_change": ((float(specialist[specialist_key]) / float(baseline[baseline_key])) - 1.0) if float(baseline[baseline_key]) != 0 else None,
        }
        for label, specialist_key, baseline_key in pairs
    ]


class SpecialistRunner:
    def __init__(self, config: SpecialistConfig) -> None:
        self.device = self._resolve_device(config.device)
        grounder = RemoteSensingGrounder()
        processor, base_model = grounder._load_components(local_only=True)
        self.processor = processor
        self.base_model = base_model.to(self.device).eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_parameter_count = sum(parameter.numel() for parameter in self.base_model.parameters())
        self.trainable_base_parameter_count = sum(parameter.numel() for parameter in self.base_model.parameters() if parameter.requires_grad)
        self.head, self.checkpoint_payload = load_specialist_checkpoint(config.checkpoint, self.device)
        self.checkpoint_hash = sha256_file(config.checkpoint)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available(): return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available(): return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is unavailable")
        return torch.device(requested)

    def predict_batch(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        images = []
        try:
            for record in records:
                with Image.open(str(record["resolved_image_path"])) as source:
                    images.append(source.convert("RGB"))
            texts = [str(record["referring_sentence"]) for record in records]
            inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
            device_inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                base_outputs = self.base_model(**device_inputs)
                hidden_state = base_outputs.last_hidden_state
                base_boxes = base_outputs.pred_boxes
                if hidden_state.shape[-1] != 256:
                    raise ValueError(f"Grounding DINO hidden dimension mismatch: {hidden_state.shape[-1]}")
                query_logits, refined_boxes = self.head(hidden_state, base_boxes)
                indices, selected_logits, selected_cxcywh = select_highest_logit(query_logits, refined_boxes)
                selected_xyxy = cxcywh_to_xyxy(selected_cxcywh)
                batch_indices = torch.arange(len(records), device=self.device)
                selected_base = base_boxes[batch_indices, indices]
            return [
                {
                    "selected_query_index": int(indices[index].cpu()),
                    "query_logit": float(selected_logits[index].cpu()),
                    "confidence": float(selected_logits[index].sigmoid().cpu()),
                    "base_box_cxcywh": [float(value) for value in selected_base[index].cpu().tolist()],
                    "refined_box_cxcywh": [float(value) for value in selected_cxcywh[index].cpu().tolist()],
                    "predicted_bbox": [float(value) for value in selected_xyxy[index].cpu().tolist()],
                }
                for index in range(len(records))
            ]
        finally:
            for image in images:
                image.close()


def shape_row(record: Mapping[str, Any], manifest_index: int, prediction: Optional[Mapping[str, Any]], error: Optional[str], latency_ms: float, runner: SpecialistRunner) -> dict[str, Any]:
    truth = normalized_xyxy(record["ground_truth_bbox"], int(record["image_width"]), int(record["image_height"]), already_normalized=True)
    base = {
        "manifest_index": manifest_index, "smoke_id": str(record["smoke_id"]), "image": record["image"],
        "image_path": record["resolved_image_path"], "object_class": record["object_class"],
        "referring_sentence": record["referring_sentence"], "ground_truth_bbox": truth,
        "bbox_was_clipped": bool(record["bbox_was_clipped"]), "completed": prediction is not None,
        "error": error, "selected_query_index": None, "query_logit": None, "confidence": None,
        "base_box_cxcywh": None, "refined_box_cxcywh": None, "predicted_bbox": None, "iou": 0.0,
        "predicted_box_area": None, "ground_truth_box_area": box_area(truth), "area_ratio": None,
        "oversized": False, "undersized": False, "latency_ms": latency_ms, "device": str(runner.device),
        "base_checkpoint": BASE_CHECKPOINT, "specialist_checkpoint_sha256": runner.checkpoint_hash,
    }
    if prediction is None:
        return base
    predicted = normalized_xyxy(prediction["predicted_bbox"], 1, 1, already_normalized=True)
    iou = float(paired_iou(torch.tensor([predicted]), torch.tensor([truth]))[0])
    predicted_area, truth_area = box_area(predicted), box_area(truth)
    ratio = predicted_area / truth_area if truth_area > 0 else math.inf
    base.update(prediction)
    base.update(
        predicted_bbox=predicted, iou=iou, predicted_box_area=predicted_area,
        area_ratio=ratio, oversized=ratio >= 2.0, undersized=ratio <= 0.5,
    )
    return base


def _csv_value(field: str, value: Any) -> Any:
    if field in {"ground_truth_bbox", "base_box_cxcywh", "refined_box_cxcywh", "predicted_bbox"}:
        return "" if value is None else json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value


def write_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["manifest_index"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS); writer.writeheader()
            writer.writerows({field: _csv_value(field, row.get(field)) for field in PREDICTION_FIELDS} for row in ordered)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file(): return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        parsed: dict[str, Any] = dict(row)
        for field in ("ground_truth_bbox", "base_box_cxcywh", "refined_box_cxcywh", "predicted_bbox"):
            parsed[field] = json.loads(row[field]) if row[field] else None
        for field in ("bbox_was_clipped", "completed", "oversized", "undersized"):
            parsed[field] = row[field].lower() == "true"
        for field in ("manifest_index", "selected_query_index"):
            parsed[field] = int(row[field]) if row[field] else None
        for field in ("query_logit", "confidence", "iou", "predicted_box_area", "ground_truth_box_area", "area_ratio", "latency_ms"):
            parsed[field] = float(row[field]) if row[field] else None
        parsed["error"] = row["error"] or None
        output[str(parsed["smoke_id"])] = parsed
    if len(output) != len(rows): raise ValueError("Partial predictions contain duplicate smoke IDs")
    return output


def pending_records(records: Sequence[Mapping[str, Any]], completed_ids: Iterable[str]) -> list[Mapping[str, Any]]:
    done = {str(item) for item in completed_ids}
    return [record for record in records if str(record["smoke_id"]) not in done]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def run(records: Sequence[Mapping[str, Any]], config: SpecialistConfig, runner: SpecialistRunner) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_predictions(config.partial_predictions) if config.resume else {}
    permitted = {str(record["smoke_id"]) for record in records}
    existing = {key: row for key, row in existing.items() if key in permitted}
    state = {
        "status": "running", "started_at": utc_now(), "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": runner.checkpoint_hash, "manifest": str(config.manifest), "image_root": str(config.image_root),
        "device": str(runner.device), "batch_size": config.batch_size, "resume": config.resume,
        "expected_samples": len(records), "base_parameter_count": runner.base_parameter_count,
        "trainable_base_parameter_count": runner.trainable_base_parameter_count,
        "strict_specialist_load": True, "oracle_best_iou_selection": False,
        "production_thresholds_applied": False, "command": reproducibility_command(config),
    }
    started = time.perf_counter()
    remaining = pending_records(records, existing)
    for offset in range(0, len(remaining), config.batch_size):
        batch = remaining[offset:offset + config.batch_size]
        batch_started = time.perf_counter()
        try:
            predictions = runner.predict_batch(batch)
            error = None
        except Exception as exception:
            predictions = [None] * len(batch)
            error = f"{type(exception).__name__}: {exception}"
        elapsed_per_sample = (time.perf_counter() - batch_started) * 1000 / len(batch)
        for record, prediction in zip(batch, predictions):
            index = next(index for index, item in enumerate(records, 1) if str(item["smoke_id"]) == str(record["smoke_id"]))
            row = shape_row(record, index, prediction, error, elapsed_per_sample, runner)
            existing[str(record["smoke_id"])] = row
            print(f"[{len(existing):03d}/{len(records)}] class={record['object_class']} query={row['selected_query_index']} IoU={float(row['iou'] or 0):.3f}", flush=True)
        write_predictions(config.partial_predictions, existing.values())
        state.update(
            updated_at=utc_now(), processed_samples=len(existing),
            completed_samples=sum(bool(row.get("completed")) for row in existing.values()),
            inference_failures=sum(not bool(row.get("completed")) for row in existing.values()),
        )
        atomic_json(config.run_state, state)
    state.update(status="predictions_complete", runtime_seconds=round(time.perf_counter() - started, 3))
    atomic_json(config.run_state, state)
    return sorted(existing.values(), key=lambda row: int(row["manifest_index"])), state


def recommendation(summary: Mapping[str, Any], comparison: Sequence[Mapping[str, Any]], collapse: Mapping[str, Any]) -> dict[str, str]:
    changes = {row["metric"]: float(row["absolute_change"]) for row in comparison}
    if summary["accuracy_at_0.50"] >= 0.50 and summary["accuracy_at_0.75"] >= 0.25 and not collapse["collapse_detected"]:
        return {"choice": "C", "label": "ready for full training"}
    if changes["accuracy_at_0.50"] > 0 or changes["mean_iou"] >= 0.05:
        return {"choice": "B", "label": "continue to 1,000-step balanced training"}
    return {"choice": "A", "label": "stop and redesign"}


def reproducibility_command(config: SpecialistConfig) -> str:
    parts = [
        "venv/bin/python", "scripts/run_vrsbench_grounding_specialist_smoke.py", "--checkpoint", str(config.checkpoint),
        "--manifest", str(config.manifest), "--image-root", str(config.image_root), "--output-dir", str(config.output_dir),
        "--device", config.device, "--batch-size", str(config.batch_size),
    ]
    if config.resume: parts.append("--resume")
    if config.limit is not None: parts.extend(("--limit", str(config.limit)))
    return shlex.join(parts)


def finalize(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], config: SpecialistConfig, state: dict[str, Any]) -> dict[str, Any]:
    output = config.output_dir
    summary = aggregate(rows, float(state["runtime_seconds"]))
    per_class = group_metrics(rows, lambda row: row["object_class"])
    clipped = group_metrics(rows, lambda row: "clipped" if row["bbox_was_clipped"] else "unclipped")
    query = query_index_analysis(
        [int(row["selected_query_index"]) for row in rows if row.get("completed")],
        [float(row["predicted_box_area"]) for row in rows if row.get("completed")],
    )
    geometry = geometry_analysis(rows)
    baseline_payload = json.loads(DEFAULT_BASELINE_RESULTS.read_text(encoding="utf-8"))
    baseline_summary = baseline_payload["summary"]
    with DEFAULT_BASELINE_PREDICTIONS.open("r", encoding="utf-8", newline="") as handle:
        baseline_ids = {str(row["smoke_id"]) for row in csv.DictReader(handle)}
    specialist_ids = {str(row["smoke_id"]) for row in rows}
    if len(records) == EXPECTED_SAMPLES and baseline_ids != specialist_ids:
        raise ValueError("Baseline and specialist predictions do not cover the same Smoke-100 IDs")
    comparison = baseline_comparison(summary, baseline_summary)
    rec = recommendation(summary, comparison, query)

    write_predictions(output / "predictions.csv", rows)
    fields = ("group", "samples", "mean_iou", "accuracy_at_0.25", "accuracy_at_0.50", "accuracy_at_0.75", "average_confidence")
    with (output / "metrics_by_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("object_class", *fields[1:])); writer.writeheader()
        for item in per_class: writer.writerow({"object_class": item["group"], **{field: item[field] for field in fields[1:]}})
    write_csv(output / "clipped_vs_unclipped.csv", clipped, fields)
    write_csv(output / "baseline_comparison.csv", comparison, ("metric", "baseline", "specialist", "absolute_change", "relative_change"))
    atomic_json(output / "query_index_analysis.json", query)
    atomic_json(output / "geometry_analysis.json", geometry)
    atomic_json(output / "errors.json", [{"smoke_id": row["smoke_id"], "error": row["error"]} for row in rows if row.get("error")])

    assets = output / "report_assets"
    ious = [float(row.get("iou") or 0.0) for row in rows]
    write_histogram(assets, "iou_distribution", "Specialist deployed-prediction IoU", ious)
    write_bar_chart(assets, "accuracy_threshold_comparison", "Baseline vs specialist accuracy", ["B @0.25", "S @0.25", "B @0.50", "S @0.50", "B @0.75", "S @0.75"], [baseline_summary["accuracy_at_0.25"], summary["accuracy_at_0.25"], baseline_summary["accuracy_at_0.50"], summary["accuracy_at_0.50"], baseline_summary["accuracy_at_0.75"], summary["accuracy_at_0.75"]])
    ordered_class = sorted(per_class, key=lambda item: (-item["accuracy_at_0.50"], -item["mean_iou"], item["group"]))
    write_horizontal_chart(assets, "per_class_accuracy_at_0_50", "Specialist per-class accuracy@0.50", [item["group"] for item in ordered_class], [item["accuracy_at_0.50"] for item in ordered_class])
    query_rows = query["most_frequent_query_indices"][:25]
    write_horizontal_chart(assets, "selected_query_index_frequency", "Selected specialist query indices", [str(item["query_index"]) for item in query_rows], [float(item["count"]) for item in query_rows], percent=False)
    write_bar_chart(assets, "predicted_vs_ground_truth_area", "Average normalized box area", ["Predicted", "Ground truth"], [geometry["average_predicted_box_area"], geometry["average_ground_truth_box_area"]], percent=False)
    write_bar_chart(assets, "latency_comparison", "Average and P95 latency (ms)", ["Baseline avg", "Specialist avg", "Baseline P95", "Specialist P95"], [baseline_summary["average_latency_ms"], summary["average_latency_ms"], baseline_summary["p95_latency_ms"], summary["p95_latency_ms"]], percent=False)

    results = {
        "benchmark": "SatQuery Grounding Specialist v1 Pilot step-100 — VRSBench Smoke-100",
        "specialist_checkpoint": str(config.checkpoint), "specialist_checkpoint_sha256": sha256_file(config.checkpoint),
        "checkpoint_metadata": {key: runner_value for key, runner_value in state.items() if key in {"base_parameter_count", "trainable_base_parameter_count", "strict_specialist_load"}},
        "base_model": BASE_CHECKPOINT, "prediction_policy": "highest specialist query logit; no oracle selection",
        "production_thresholds_applied": False, "summary": summary, "per_class": per_class,
        "clipped_vs_unclipped": clipped, "query_index_analysis": query, "geometry_analysis": geometry,
        "baseline_comparison": comparison, "baseline_note": "Same manifest; baseline used production filtering and could return null, while the specialist always selects one learned query.",
        "recommendation": rec, "reproducibility_command": reproducibility_command(config),
        "validation_gates": {
            "expected_samples": len(records) == (config.limit or EXPECTED_SAMPLES), "processed_samples": len(rows) == len(records),
            "duplicate_smoke_ids": len(rows) - len(specialist_ids), "missing_smoke_ids": len({str(record["smoke_id"]) for record in records} - specialist_ids),
            "class_count": len(per_class), "clipped_samples": sum(bool(row["bbox_was_clipped"]) for row in rows),
            "same_manifest_as_baseline": baseline_ids == specialist_ids if len(records) == EXPECTED_SAMPLES else None,
            "strict_specialist_state_dict": True, "base_model_frozen": state["trainable_base_parameter_count"] == 0,
            "highest_logit_selection": True, "oracle_best_iou_selection": False, "production_thresholds_applied": False,
            "one_prediction_per_completed_sample": all(row.get("predicted_bbox") is not None for row in rows if row.get("completed")),
            "production_behavior_modified": False,
        },
    }
    atomic_json(output / "results.json", results)
    best = ordered_class[:5]
    worst = sorted(per_class, key=lambda item: (item["accuracy_at_0.50"], item["mean_iou"], item["group"]))[:5]
    clipped_map = {item["group"]: item for item in clipped}
    changes = {item["metric"]: item for item in comparison}
    atomic_text(output / "summary.txt", "\n".join([
        "SatQuery Grounding Specialist v1 Pilot step-100 — VRSBench Smoke-100",
        "=" * 70, f"Completed: {summary['completed_samples']}/{summary['samples']}",
        f"Mean / median IoU: {summary['mean_iou']:.6f} / {summary['median_iou']:.6f}",
        f"Accuracy@0.25/@0.50/@0.75: {summary['accuracy_at_0.25']:.4%} / {summary['accuracy_at_0.50']:.4%} / {summary['accuracy_at_0.75']:.4%}",
        f"Failure rate: {summary['null_or_failure_rate']:.4%}", f"Average / P95 latency: {summary['average_latency_ms']:.3f} / {summary['p95_latency_ms']:.3f} ms",
        f"Unique query indices / top-5 coverage: {query['unique_selected_query_indices']} / {query['top_5_coverage']:.4%}",
        f"Collapse detected: {query['collapse_detected']} ({', '.join(query['collapse_reasons']) or 'none'})",
        f"Recommendation: {rec['choice']}. {rec['label']}",
    ]) + "\n")
    report = [
        "# SatQuery Grounding Specialist v1 Pilot — VRSBench Smoke-100", "",
        "This evaluation uses the same 100 records as the unchanged Grounding DINO baseline. The base `IDEA-Research/grounding-dino-tiny` model was frozen, the exported specialist state was loaded strictly, and every complete referring sentence was passed to the base processor.", "",
        "The deployed specialist prediction is the query with the highest specialist logit. Ground-truth IoU was computed only after selection. No best-IoU oracle selection, production reliability gate, box threshold, or text threshold was applied.", "",
        "## Results", "", "| Metric | Specialist | Baseline | Change |", "|---|---:|---:|---:|",
        f"| Mean IoU | {summary['mean_iou']:.4f} | {baseline_summary['mean_iou_all']:.4f} | {changes['mean_iou']['absolute_change']:+.4f} |",
        f"| Accuracy@0.25 | {summary['accuracy_at_0.25']:.2%} | {baseline_summary['accuracy_at_0.25']:.2%} | {changes['accuracy_at_0.25']['absolute_change']:+.2%} |",
        f"| Accuracy@0.50 | {summary['accuracy_at_0.50']:.2%} | {baseline_summary['accuracy_at_0.50']:.2%} | {changes['accuracy_at_0.50']['absolute_change']:+.2%} |",
        f"| Accuracy@0.75 | {summary['accuracy_at_0.75']:.2%} | {baseline_summary['accuracy_at_0.75']:.2%} | {changes['accuracy_at_0.75']['absolute_change']:+.2%} |",
        f"| Null/failure rate | {summary['null_or_failure_rate']:.2%} | {baseline_summary['null_prediction_rate']:.2%} | {changes['null_rate']['absolute_change']:+.2%} |",
        f"| Average latency | {summary['average_latency_ms']:.1f} ms | {baseline_summary['average_latency_ms']:.1f} ms | {changes['average_latency_ms']['absolute_change']:+.1f} ms |", "",
        "The baseline used production filtering and could return null. The specialist always selects one learned query unless inference fails; its lower null rate is therefore not directly equivalent to better localization.", "",
        "## Class and clipping analysis", "", "Best five by accuracy@0.50: " + ", ".join(f"{item['group']} ({item['accuracy_at_0.50']:.0%})" for item in best) + ".", "",
        "Worst five by accuracy@0.50: " + ", ".join(f"{item['group']} ({item['accuracy_at_0.50']:.0%})" for item in worst) + ".", "",
        f"Clipped accuracy@0.50: **{clipped_map['clipped']['accuracy_at_0.50']:.2%}**; unclipped: **{clipped_map['unclipped']['accuracy_at_0.50']:.2%}**.", "",
        "## Query and geometry analysis", "", f"The specialist selected **{query['unique_selected_query_indices']}** unique query indices. Top-1 / top-5 / top-10 coverage is **{query['top_1_coverage']:.2%} / {query['top_5_coverage']:.2%} / {query['top_10_coverage']:.2%}**. Collapse detected: **{query['collapse_detected']}** ({', '.join(query['collapse_reasons']) or 'none'}).", "",
        f"Average predicted / ground-truth normalized box area is **{geometry['average_predicted_box_area']:.4f} / {geometry['average_ground_truth_box_area']:.4f}**. Oversized / undersized rates are **{geometry['oversized_prediction_rate']:.2%} / {geometry['undersized_prediction_rate']:.2%}**.", "",
        "## Recommendation", "", f"**{rec['choice']}. {rec['label']}** under the transparent evaluation rule encoded in this script.", "",
        "## Reproducibility", "", "```bash", reproducibility_command(config), "```", "",
        "This Smoke-100 pilot does not establish full-dataset generalization or calibrated confidence. No training, threshold tuning, checkpoint change, or production behavior modification occurred.", "",
    ]
    atomic_text(output / "benchmark_report.md", "\n".join(report))
    state.update(status="complete", completed_at=utc_now(), validation_gates=results["validation_gates"])
    atomic_json(config.run_state, state)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("batch-size and limit must be positive")
    config = SpecialistConfig(
        checkpoint=args.checkpoint.expanduser().resolve(), manifest=args.manifest.expanduser().resolve(),
        image_root=args.image_root.expanduser().resolve(), output_dir=args.output_dir.expanduser().resolve(),
        device=args.device, batch_size=args.batch_size, resume=args.resume, limit=args.limit,
    )
    try:
        all_records = load_manifest(config.manifest, config.image_root)
        records = all_records[:config.limit] if config.limit else all_records
        runner = SpecialistRunner(config)
        if runner.trainable_base_parameter_count != 0:
            raise ValueError("Frozen Grounding DINO base contains trainable parameters")
        print(f"Loaded specialist strictly; base frozen; device={runner.device}; checkpoint_sha256={runner.checkpoint_hash}", flush=True)
        rows, state = run(records, config, runner)
        results = finalize(records, rows, config, state)
    except KeyboardInterrupt:
        print("Interrupted; resume with --resume.", file=sys.stderr); return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Specialist benchmark failed: {error}", file=sys.stderr); return 2
    summary = results["summary"]
    print(f"Completed {summary['completed_samples']}/{summary['samples']}; mean_iou={summary['mean_iou']:.4f}; acc@0.50={summary['accuracy_at_0.50']:.2%}; failures={summary['inference_failures']}")
    return 0 if summary["inference_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
