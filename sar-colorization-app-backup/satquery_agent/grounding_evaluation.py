"""Offline, local reliability evaluation for GeoVision Grounding DINO outputs.

This module never mutates production configuration. It evaluates a supplied
candidate pool with an explicit reliability policy and labelled local boxes.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .models import ImageFormat, ImageMetadata
from .specialists.grounder import (
    CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_RELIABILITY_POLICY,
    GroundingReliabilityPolicy,
    RemoteSensingGrounder,
    _grounding_target_sizes,
    _values,
    evaluate_detection_quality,
)


Box = Tuple[float, float, float, float]
SUPPORTED_INITIAL_TARGETS = (
    "stadium",
    "aircraft",
    "building",
    "bridge",
    "road",
    "water body",
    "river",
    "lake",
    "ship",
    "vegetation",
)
IOU_THRESHOLDS = (0.25, 0.50)
SCORE_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
AREA_RATIO_GRID = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
FULL_FRAME_RATIO = 0.98
_SAFE_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_IMAGE_FORMATS = {
    ".png": ImageFormat.PNG,
    ".jpg": ImageFormat.JPEG,
    ".jpeg": ImageFormat.JPEG,
    ".tif": ImageFormat.TIFF,
    ".tiff": ImageFormat.TIFF,
}


class GroundingEvaluationError(ValueError):
    """Raised for invalid local annotations or unsafe evaluation output."""


@dataclass(frozen=True)
class GroundingEvaluationSample:
    sample_id: str
    image_path: Path
    target: str
    query: str
    boxes: Tuple[Box, ...]
    target_present: bool
    annotation_source: str
    notes: Optional[str]
    width: int
    height: int


@dataclass(frozen=True)
class GroundingEvaluationDataset:
    samples: Tuple[GroundingEvaluationSample, ...]


@dataclass(frozen=True)
class ModelCandidate:
    score: float
    box: Box
    label: str


@dataclass(frozen=True)
class CandidateInference:
    candidates: Tuple[ModelCandidate, ...]
    device: str = "synthetic"
    checkpoint: str = CHECKPOINT
    error_code: Optional[str] = None


CandidateProvider = Callable[[GroundingEvaluationSample], CandidateInference]


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise GroundingEvaluationError(f"{field} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GroundingEvaluationError(f"{field} must be a finite number.") from error
    if not math.isfinite(number):
        raise GroundingEvaluationError(f"{field} must be a finite number.")
    return number


def _safe_annotation_source(value: Any) -> str:
    source = str(value or "").strip()
    if not source:
        raise GroundingEvaluationError("Each sample must declare an annotation source.")
    if len(source) > 120 or source.startswith(("/", "~")) or "\\" in source:
        raise GroundingEvaluationError("Annotation source must be a short label, not a filesystem path.")
    return source


def _load_box(value: Any, sample_id: str, width: int, height: int) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise GroundingEvaluationError(f"Sample '{sample_id}' contains an invalid reference box.")
    left, top, right, bottom = (_finite_number(item, f"box coordinate for '{sample_id}'") for item in value)
    if left < 0 or top < 0 or right > width or bottom > height:
        raise GroundingEvaluationError(f"Sample '{sample_id}' contains a reference box outside the image bounds.")
    if right <= left or bottom <= top:
        raise GroundingEvaluationError(f"Sample '{sample_id}' contains a zero- or negative-area reference box.")
    return left, top, right, bottom


def load_grounding_evaluation_dataset(dataset_dir: Path) -> GroundingEvaluationDataset:
    root = dataset_dir.expanduser().resolve()
    annotation_path = root / "annotations.json"
    if not annotation_path.is_file():
        raise GroundingEvaluationError("The dataset must contain annotations.json.")
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GroundingEvaluationError("annotations.json could not be read as valid JSON.") from error
    rows = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise GroundingEvaluationError("annotations.json must contain a 'samples' array.")

    samples: List[GroundingEvaluationSample] = []
    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GroundingEvaluationError("Every dataset sample must be a JSON object.")
        sample_id = str(row.get("id") or "").strip()
        if not _SAFE_SAMPLE_ID.fullmatch(sample_id):
            raise GroundingEvaluationError("Sample IDs must contain only letters, numbers, underscores, and hyphens.")
        if sample_id in seen_ids:
            raise GroundingEvaluationError(f"Duplicate sample ID '{sample_id}'.")
        seen_ids.add(sample_id)

        image_value = row.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            raise GroundingEvaluationError(f"Sample '{sample_id}' must declare an image.")
        relative_image = Path(image_value)
        if relative_image.is_absolute():
            raise GroundingEvaluationError(f"Sample '{sample_id}' must use a dataset-relative image name.")
        image_path = (root / relative_image).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise GroundingEvaluationError(f"Sample '{sample_id}' image escapes the dataset directory.") from error
        if not image_path.is_file():
            raise GroundingEvaluationError(f"Sample '{sample_id}' references a missing image.")
        if image_path.suffix.lower() not in _IMAGE_FORMATS:
            raise GroundingEvaluationError(f"Sample '{sample_id}' uses an unsupported image format.")
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except OSError as error:
            raise GroundingEvaluationError(f"Sample '{sample_id}' image could not be decoded.") from error
        if width <= 0 or height <= 0:
            raise GroundingEvaluationError(f"Sample '{sample_id}' image has invalid dimensions.")

        target = str(row.get("target") or "").strip().lower()
        query = str(row.get("query") or "").strip()
        if not target or not query:
            raise GroundingEvaluationError(f"Sample '{sample_id}' must declare target and query text.")
        raw_boxes = row.get("boxes")
        if not isinstance(raw_boxes, list):
            raise GroundingEvaluationError(f"Sample '{sample_id}' must explicitly declare a boxes array.")
        boxes = tuple(_load_box(value, sample_id, width, height) for value in raw_boxes)
        explicit_presence = row.get("target_present")
        if explicit_presence is None:
            if not boxes:
                raise GroundingEvaluationError(
                    f"Sample '{sample_id}' with no boxes must explicitly set target_present to false."
                )
            target_present = True
        elif not isinstance(explicit_presence, bool):
            raise GroundingEvaluationError(f"Sample '{sample_id}' target_present must be boolean.")
        else:
            target_present = explicit_presence
        if target_present and not boxes:
            raise GroundingEvaluationError(f"Positive sample '{sample_id}' must contain at least one reference box.")
        if not target_present and boxes:
            raise GroundingEvaluationError(f"Negative sample '{sample_id}' must use an empty boxes array.")

        notes_value = row.get("notes")
        notes = None if notes_value is None else str(notes_value)
        samples.append(
            GroundingEvaluationSample(
                sample_id=sample_id,
                image_path=image_path,
                target=target,
                query=query,
                boxes=boxes,
                target_present=target_present,
                annotation_source=_safe_annotation_source(row.get("source")),
                notes=notes,
                width=width,
                height=height,
            )
        )
    return GroundingEvaluationDataset(samples=tuple(samples))


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_point_in_reference(candidate: Sequence[float], reference: Sequence[float]) -> bool:
    center_x = (float(candidate[0]) + float(candidate[2])) / 2.0
    center_y = (float(candidate[1]) + float(candidate[3])) / 2.0
    return float(reference[0]) <= center_x <= float(reference[2]) and float(reference[1]) <= center_y <= float(reference[3])


def _metadata(sample: GroundingEvaluationSample) -> ImageMetadata:
    return ImageMetadata(
        file_id=sample.sample_id,
        original_name="evaluation-image",
        safe_name="evaluation-image",
        format=_IMAGE_FORMATS[sample.image_path.suffix.lower()],
        mime_type="application/octet-stream",
        size_bytes=0,
        width=sample.width,
        height=sample.height,
        band_count=3,
        dtype="unknown",
        is_georeferenced=False,
        color_interpretation=["r", "g", "b"],
    )


def _candidate_record(
    sample: GroundingEvaluationSample,
    label: str,
    box: Sequence[Optional[float]],
    score: Optional[float],
    area_ratio: Optional[float],
    accepted: bool,
    rejection_reasons: Sequence[str],
) -> Dict[str, Any]:
    finite_box = len(box) == 4 and all(value is not None and math.isfinite(float(value)) for value in box)
    ious = [box_iou(box, reference) for reference in sample.boxes] if finite_box else []
    best_iou = max(ious) if ious else (None if not sample.boxes else 0.0)
    matched_reference_index = ious.index(best_iou) if ious and best_iou is not None else None
    matched_reference_box = list(sample.boxes[matched_reference_index]) if matched_reference_index is not None else None
    center_inclusion = any(center_point_in_reference(box, reference) for reference in sample.boxes) if finite_box and sample.boxes else None
    if accepted and not sample.target_present:
        outcome = "accepted_false_positive"
    elif accepted and (best_iou or 0.0) < IOU_THRESHOLDS[0]:
        outcome = "accepted_false_localization"
    elif accepted:
        outcome = "accepted_true_positive"
    else:
        outcome = "rejected_candidate"
    return {
        "prediction_label": label,
        "alignment_score": score,
        "box_source_xyxy": [round(float(value), 6) if value is not None else None for value in box],
        "box_area_ratio": area_ratio,
        "accepted": accepted,
        "rejection_reasons": list(rejection_reasons),
        "reference_ious": [round(value, 8) for value in ious],
        "best_iou": round(best_iou, 8) if best_iou is not None else None,
        "matched_reference_index": matched_reference_index,
        "matched_reference_box": matched_reference_box,
        "center_point_inclusion": center_inclusion,
        "meets_iou_25": best_iou is not None and best_iou >= 0.25,
        "meets_iou_50": best_iou is not None and best_iou >= 0.50,
        "full_frame_prediction": area_ratio is not None and area_ratio >= FULL_FRAME_RATIO,
        "outcome": outcome,
    }


def _evaluate_sample(
    sample: GroundingEvaluationSample,
    inference: CandidateInference,
    policy: GroundingReliabilityPolicy,
) -> Dict[str, Any]:
    if inference.error_code:
        return {
            "sample_id": sample.sample_id,
            "target": sample.target,
            "annotation_source": sample.annotation_source,
            "target_present": sample.target_present,
            "reference_box_count": len(sample.boxes),
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "no_model_candidate": None,
            "correct_rejection": None,
            "missed_target": None,
            "inference_error_code": inference.error_code,
            "candidates": [],
        }
    boxes = [candidate.box for candidate in inference.candidates]
    scores = [candidate.score for candidate in inference.candidates]
    labels = [candidate.label for candidate in inference.candidates]
    accepted, rejected = evaluate_detection_quality(
        boxes,
        scores,
        labels,
        _metadata(sample),
        sample.target,
        DEFAULT_CONFIG,
        policy,
    )
    records: List[Dict[str, Any]] = []
    for detection in accepted:
        records.append(
            _candidate_record(
                sample,
                detection.label,
                detection.bbox_pixels,
                detection.score,
                detection.quality.box_area_ratio,
                True,
                [],
            )
        )
    for candidate in rejected:
        records.append(
            _candidate_record(
                sample,
                candidate.label,
                candidate.bbox_source_xyxy,
                candidate.score,
                candidate.box_area_ratio,
                False,
                candidate.rejection_reasons,
            )
        )
    matched = any(record["accepted"] and record["meets_iou_25"] for record in records)
    accepted_count = sum(record["accepted"] for record in records)
    return {
        "sample_id": sample.sample_id,
        "target": sample.target,
        "annotation_source": sample.annotation_source,
        "target_present": sample.target_present,
        "reference_box_count": len(sample.boxes),
        "candidate_count": len(records),
        "accepted_count": accepted_count,
        "rejected_count": len(records) - accepted_count,
        "no_model_candidate": not records,
        "correct_rejection": (not sample.target_present and accepted_count == 0),
        "missed_target": (sample.target_present and not matched),
        "inference_error_code": None,
        "candidates": records,
    }


def _greedy_matches(sample: Dict[str, Any], threshold: float) -> int:
    candidates = [item for item in sample["candidates"] if item["accepted"]]
    pairs: List[Tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        for reference_index, iou in enumerate(candidate["reference_ious"]):
            if iou >= threshold:
                pairs.append((iou, candidate_index, reference_index))
    used_candidates = set()
    used_references = set()
    for _iou_value, candidate_index, reference_index in sorted(pairs, reverse=True):
        if candidate_index in used_candidates or reference_index in used_references:
            continue
        used_candidates.add(candidate_index)
        used_references.add(reference_index)
    return len(used_candidates)


def _distribution(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
    }


def aggregate_grounding_evaluation(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    available = [sample for sample in samples if not sample["inference_error_code"]]
    candidates = [candidate for sample in available for candidate in sample["candidates"]]
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    rejected = [candidate for candidate in candidates if not candidate["accepted"]]
    positive = [sample for sample in available if sample["target_present"]]
    negative = [sample for sample in available if not sample["target_present"]]
    reference_count = sum(sample["reference_box_count"] for sample in positive)
    tp25 = sum(_greedy_matches(sample, 0.25) for sample in positive)
    tp50 = sum(_greedy_matches(sample, 0.50) for sample in positive)
    false_positive_count = sum(
        candidate["accepted"] for sample in negative for candidate in sample["candidates"]
    )
    accepted_positive_count = sum(
        candidate["accepted"] for sample in positive for candidate in sample["candidates"]
    )
    false_localization_count = max(0, accepted_positive_count - tp25)
    missed_target_count = sum(bool(sample["missed_target"]) for sample in positive)
    accepted_count = len(accepted)
    candidate_count = len(candidates)
    best_ious = [candidate["best_iou"] for candidate in accepted if candidate["best_iou"] is not None]
    return {
        "evaluated_sample_count": len(available),
        "unavailable_sample_count": len(samples) - len(available),
        "positive_sample_count": len(positive),
        "negative_sample_count": len(negative),
        "reference_box_count": reference_count,
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "true_positive_count_iou_25": tp25,
        "true_positive_count_iou_50": tp50,
        "accepted_precision_iou_25": tp25 / accepted_count if accepted_count else None,
        "accepted_precision_iou_50": tp50 / accepted_count if accepted_count else None,
        "accepted_recall_iou_25": tp25 / reference_count if reference_count else None,
        "accepted_recall_iou_50": tp50 / reference_count if reference_count else None,
        "false_positive_count": false_positive_count,
        "false_localization_count": false_localization_count,
        "false_localization_rate": false_localization_count / accepted_positive_count if accepted_positive_count else None,
        "missed_target_count": missed_target_count,
        "missed_target_rate": missed_target_count / len(positive) if positive else None,
        "no_prediction_count": sum(bool(sample["no_model_candidate"]) for sample in available),
        "correct_rejection_count": sum(bool(sample["correct_rejection"]) for sample in negative),
        "weak_rejected_negative_count": sum(
            bool(sample["correct_rejection"] and sample["rejected_count"] > 0) for sample in negative
        ),
        "mean_accepted_best_iou": statistics.fmean(best_ious) if best_ious else None,
        "median_accepted_best_iou": statistics.median(best_ious) if best_ious else None,
        "score_distribution": _distribution(candidate["alignment_score"] for candidate in candidates),
        "box_area_ratio_distribution": _distribution(candidate["box_area_ratio"] for candidate in candidates),
        "acceptance_rate": accepted_count / candidate_count if candidate_count else None,
        "rejection_rate": len(rejected) / candidate_count if candidate_count else None,
        "full_frame_candidate_count": sum(candidate["full_frame_prediction"] for candidate in candidates),
        "full_frame_candidate_rate": sum(candidate["full_frame_prediction"] for candidate in candidates) / candidate_count if candidate_count else None,
        "full_frame_acceptance_count": sum(candidate["full_frame_prediction"] and candidate["accepted"] for candidate in candidates),
    }


def _per_target(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["target"]].append(sample)
    return [{"target": target, "sample_count": len(rows), **aggregate_grounding_evaluation(rows)} for target, rows in sorted(grouped.items())]


def _examples(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "reliable_localization": [],
        "low_score_rejection": [],
        "near_full_frame_rejection": [],
        "false_accepted_localization": [],
        "missed_target": [],
    }
    for sample in samples:
        for candidate in sample["candidates"]:
            if candidate["outcome"] == "accepted_true_positive":
                categories["reliable_localization"].append(sample["sample_id"])
            if "alignment_score_below_minimum" in candidate["rejection_reasons"]:
                categories["low_score_rejection"].append(sample["sample_id"])
            if candidate["full_frame_prediction"] and not candidate["accepted"]:
                categories["near_full_frame_rejection"].append(sample["sample_id"])
            if candidate["outcome"] in {"accepted_false_localization", "accepted_false_positive"}:
                categories["false_accepted_localization"].append(sample["sample_id"])
        if sample["missed_target"]:
            categories["missed_target"].append(sample["sample_id"])
    return {name: list(dict.fromkeys(values))[:5] for name, values in categories.items()}


def evaluate_grounding_dataset(
    dataset: GroundingEvaluationDataset,
    provider: CandidateProvider,
    minimum_score: float = DEFAULT_RELIABILITY_POLICY.minimum_alignment_score,
    maximum_area_ratio: float = DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio,
    limit: Optional[int] = None,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    candidate_floor = float(getattr(provider, "candidate_floor", min(SCORE_GRID)))
    policy = GroundingReliabilityPolicy(
        minimum_alignment_score=minimum_score,
        maximum_localized_area_ratio=maximum_area_ratio,
        localized_targets=DEFAULT_RELIABILITY_POLICY.localized_targets,
        calibration_status=DEFAULT_RELIABILITY_POLICY.calibration_status,
    )
    selected = [sample for sample in dataset.samples if target is None or sample.target == target.strip().lower()]
    if limit is not None:
        if limit < 1:
            raise GroundingEvaluationError("--limit must be at least 1.")
        selected = selected[:limit]
    inferences: Dict[str, CandidateInference] = {}
    sample_results: List[Dict[str, Any]] = []
    for sample in selected:
        inference = provider(sample)
        inferences[sample.sample_id] = inference
        sample_results.append(_evaluate_sample(sample, inference, policy))
    devices = sorted({item.device for item in inferences.values() if not item.error_code})
    checkpoints = sorted({item.checkpoint for item in inferences.values() if not item.error_code})
    summary = aggregate_grounding_evaluation(sample_results)
    category_counts = dict(sorted(Counter(sample.target for sample in selected).items()))
    small_dataset = len(selected) < 50 or any(count < 5 for count in category_counts.values())
    return {
        "framework": "GeoVision local Grounding DINO reliability evaluation",
        "benchmark_claim": False,
        "dataset": {
            "sample_count": len(selected),
            "category_counts": category_counts,
            "positive_sample_count": sum(sample.target_present for sample in selected),
            "negative_sample_count": sum(not sample.target_present for sample in selected),
            "small_dataset_warning": "Results are exploratory and not statistically significant for a small local dataset." if small_dataset else None,
        },
        "model": {"checkpoint": checkpoints[0] if len(checkpoints) == 1 else None, "devices": devices},
        "production_gate_reference": {
            "model_candidate_box_threshold": DEFAULT_CONFIG.box_threshold,
            "model_text_threshold": DEFAULT_CONFIG.text_threshold,
            "minimum_alignment_score": DEFAULT_RELIABILITY_POLICY.minimum_alignment_score,
            "maximum_localized_area_ratio": DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio,
            "calibration_status": DEFAULT_RELIABILITY_POLICY.calibration_status,
        },
        "evaluation_gate": {
            "candidate_floor": candidate_floor,
            "minimum_alignment_score": minimum_score,
            "maximum_localized_area_ratio": maximum_area_ratio,
            "analysis_only": True,
            "uses_production_reliability_values": minimum_score == DEFAULT_RELIABILITY_POLICY.minimum_alignment_score and maximum_area_ratio == DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio,
        },
        "iou_definitions": {"iou_25": "IoU >= 0.25", "iou_50": "IoU >= 0.50", "matching": "Greedy one-to-one matching by descending IoU."},
        "metrics": summary,
        "per_target": _per_target(sample_results),
        "examples": _examples(sample_results),
        "samples": sample_results,
        "_inferences": inferences,
        "_selected_samples": tuple(selected),
    }


def evaluate_threshold_grid(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples: Sequence[GroundingEvaluationSample] = report["_selected_samples"]
    inferences: Dict[str, CandidateInference] = report["_inferences"]
    rows: List[Dict[str, Any]] = []
    for minimum_score in SCORE_GRID:
        for maximum_area_ratio in AREA_RATIO_GRID:
            policy = GroundingReliabilityPolicy(
                minimum_alignment_score=minimum_score,
                maximum_localized_area_ratio=maximum_area_ratio,
                localized_targets=DEFAULT_RELIABILITY_POLICY.localized_targets,
                calibration_status=DEFAULT_RELIABILITY_POLICY.calibration_status,
            )
            evaluated = [_evaluate_sample(sample, inferences[sample.sample_id], policy) for sample in samples]
            metrics = aggregate_grounding_evaluation(evaluated)
            rows.append({
                "minimum_score": minimum_score,
                "maximum_area_ratio": maximum_area_ratio,
                "accepted_precision": metrics["accepted_precision_iou_25"],
                "accepted_recall": metrics["accepted_recall_iou_25"],
                "false_localization_rate": metrics["false_localization_rate"],
                "missed_target_rate": metrics["missed_target_rate"],
                "full_frame_acceptance_count": metrics["full_frame_acceptance_count"],
                "accepted_count": metrics["accepted_count"],
                "rejected_count": metrics["rejected_count"],
            })
    return rows


def _public_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _format_metric(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_report(report: Dict[str, Any], threshold_rows: Sequence[Dict[str, Any]]) -> str:
    dataset = report["dataset"]
    metrics = report["metrics"]
    lines = [
        "# GeoVision Local Grounding Reliability Evaluation",
        "",
        "> Exploratory local validation only. This is not a statistically significant benchmark or scientific performance claim.",
        "",
        "## Dataset",
        "",
        f"- Samples: {dataset['sample_count']}",
        f"- Positive samples: {dataset['positive_sample_count']}",
        f"- Negative samples: {dataset['negative_sample_count']}",
        f"- Categories: {json.dumps(dataset['category_counts'], sort_keys=True)}",
        f"- Warning: {dataset['small_dataset_warning'] or 'No small-dataset warning triggered.'}",
        "",
        "## Model and operational gate",
        "",
        f"- Checkpoint: {report['model']['checkpoint'] or 'Unavailable'}",
        f"- Device(s): {', '.join(report['model']['devices']) or 'Unavailable'}",
        f"- Production minimum alignment score: {report['production_gate_reference']['minimum_alignment_score']}",
        f"- Production maximum localized area ratio: {report['production_gate_reference']['maximum_localized_area_ratio']}",
        f"- Production processor candidate threshold: {report['production_gate_reference']['model_candidate_box_threshold']}",
        f"- Evaluation candidate floor: {report['evaluation_gate']['candidate_floor']}",
        f"- Disclaimer: {report['production_gate_reference']['calibration_status']}",
        "- Alignment scores are text-region alignment scores, not probabilities.",
        "",
        "## Evaluation-gate metrics",
        "",
    ]
    for name in (
        "accepted_precision_iou_25", "accepted_recall_iou_25", "accepted_precision_iou_50",
        "accepted_recall_iou_50", "false_positive_count", "false_localization_count",
        "missed_target_count", "no_prediction_count", "mean_accepted_best_iou",
        "median_accepted_best_iou", "acceptance_rate", "rejection_rate", "full_frame_candidate_rate",
    ):
        lines.append(f"- {name.replace('_', ' ').title()}: {_format_metric(metrics.get(name))}")
    lines.extend([
        f"- Score distribution: {json.dumps(metrics['score_distribution'], sort_keys=True)}",
        f"- Box area ratio distribution: {json.dumps(metrics['box_area_ratio_distribution'], sort_keys=True)}",
        "",
        "These fields distinguish raw model-candidate distributions, operational gate acceptance/rejection, and reference-box localization accuracy.",
        "",
        "## IoU definitions",
        "",
        "- IoU 0.25: a matched prediction/reference pair has IoU >= 0.25.",
        "- IoU 0.50: a matched prediction/reference pair has IoU >= 0.50.",
        "- Aggregate precision and recall use greedy one-to-one matching by descending IoU.",
        "",
        "## Metrics by category",
        "",
    ])
    if report["per_target"]:
        lines.append("| Target | Samples | Precision @ 0.25 | Recall @ 0.25 | Missed targets |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in report["per_target"]:
            lines.append(f"| {row['target']} | {row['sample_count']} | {_format_metric(row['accepted_precision_iou_25'])} | {_format_metric(row['accepted_recall_iou_25'])} | {row['missed_target_count']} |")
    else:
        lines.append("Unavailable: no evaluated samples.")
    lines.extend(["", "## Auditable examples", ""])
    for name, sample_ids in report["examples"].items():
        lines.append(f"- {name.replace('_', ' ').title()}: {', '.join(sample_ids) if sample_ids else 'Unavailable'}")
    lines.extend([
        "",
        "## Threshold grid",
        "",
        f"- Evaluated combinations: {len(threshold_rows)}",
        "- Grid values are analysis-only. No production threshold was selected or changed.",
        "- Candidate values should not be recommended until enough representative labelled samples are available.",
        "",
        "## Limitations",
        "",
        "- Grounding DINO is a general-domain zero-shot detector and is not calibrated for remote-sensing imagery.",
        "- Small local datasets do not support statistically significant performance claims.",
        "- Reference-box quality, target scale, sensor characteristics, and scene diversity directly affect these metrics.",
        "- IoU measures box overlap; it does not establish scientific or semantic ground truth beyond the supplied annotations.",
        "",
    ])
    return "\n".join(lines)


def write_grounding_evaluation_outputs(report: Dict[str, Any], output_dir: Path) -> List[str]:
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    public = _public_report(report)
    threshold_rows = evaluate_threshold_grid(report)
    artifact_names = [
        "summary.json", "per_sample.csv", "per_target.csv", "rejected_candidates.csv",
        "false_localizations.csv", "evaluation_report.md", "threshold_grid.csv",
    ]
    (output / "summary.json").write_text(json.dumps(public, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    per_sample_rows = [{key: value for key, value in sample.items() if key != "candidates"} for sample in public["samples"]]
    _write_csv(output / "per_sample.csv", per_sample_rows, (
        "sample_id", "target", "annotation_source", "target_present", "reference_box_count",
        "candidate_count", "accepted_count", "rejected_count", "no_model_candidate",
        "correct_rejection", "missed_target", "inference_error_code",
    ))
    _write_csv(output / "per_target.csv", public["per_target"], (
        "target", "sample_count", "positive_sample_count", "negative_sample_count", "candidate_count",
        "accepted_count", "rejected_count", "accepted_precision_iou_25", "accepted_recall_iou_25",
        "accepted_precision_iou_50", "accepted_recall_iou_50", "false_positive_count",
        "false_localization_count", "missed_target_count", "no_prediction_count",
        "mean_accepted_best_iou", "median_accepted_best_iou", "score_distribution",
        "box_area_ratio_distribution", "acceptance_rate", "rejection_rate", "full_frame_candidate_rate",
    ))

    rejected_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    for sample in public["samples"]:
        for index, candidate in enumerate(sample["candidates"], start=1):
            row = {"sample_id": sample["sample_id"], "target": sample["target"], "candidate_index": index, **candidate}
            if not candidate["accepted"]:
                rejected_rows.append(row)
            if candidate["outcome"] in {"accepted_false_localization", "accepted_false_positive"}:
                false_rows.append(row)
    candidate_fields = (
        "sample_id", "target", "candidate_index", "prediction_label", "alignment_score", "box_source_xyxy",
        "box_area_ratio", "accepted", "rejection_reasons", "best_iou", "matched_reference_index",
        "matched_reference_box", "center_point_inclusion", "full_frame_prediction", "outcome",
    )
    _write_csv(output / "rejected_candidates.csv", rejected_rows, candidate_fields)
    _write_csv(output / "false_localizations.csv", false_rows, candidate_fields)
    _write_csv(output / "threshold_grid.csv", threshold_rows, (
        "minimum_score", "maximum_area_ratio", "accepted_precision", "accepted_recall",
        "false_localization_rate", "missed_target_rate", "full_frame_acceptance_count",
        "accepted_count", "rejected_count",
    ))
    (output / "evaluation_report.md").write_text(_markdown_report(public, threshold_rows), encoding="utf-8")
    return artifact_names


class LocalGroundingDinoCandidateProvider:
    """Evaluation-only candidate provider using the unchanged local checkpoint."""

    def __init__(self, device: str = "auto", reuse_model: bool = True, candidate_floor: float = 0.30) -> None:
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise GroundingEvaluationError("Device must be one of auto, cpu, mps, or cuda.")
        self.device = device
        self.reuse_model = reuse_model
        self.candidate_floor = candidate_floor
        self._grounder: Optional[RemoteSensingGrounder] = None

    def _loaded_grounder(self) -> RemoteSensingGrounder:
        import torch

        grounder = self._grounder if self.reuse_model else None
        if grounder is None:
            grounder = RemoteSensingGrounder()
            grounder.load()
            if self.device != "auto":
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise GroundingEvaluationError("CUDA was requested but is unavailable.")
                if self.device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                    raise GroundingEvaluationError("MPS was requested but is unavailable.")
                grounder._model.to(self.device)
                grounder._device = self.device
            if self.reuse_model:
                self._grounder = grounder
        return grounder

    def __call__(self, sample: GroundingEvaluationSample) -> CandidateInference:
        import torch

        grounder = self._loaded_grounder()
        try:
            with Image.open(sample.image_path) as source:
                prepared = source.convert("RGB")
            if max(prepared.size) > grounder.config.maximum_source_dimension:
                prepared.thumbnail(
                    (grounder.config.maximum_source_dimension, grounder.config.maximum_source_dimension),
                    Image.Resampling.LANCZOS,
                )
            prompt = f"{sample.target.lower().rstrip('.')}."
            inputs = grounder._processor(images=prepared, text=prompt, return_tensors="pt")
            device_inputs = {name: value.to(grounder._device) for name, value in inputs.items()}
            with grounder._inference_lock, torch.inference_mode():
                outputs = grounder._model(**device_inputs)
            processed = grounder._processor.post_process_grounded_object_detection(
                outputs,
                device_inputs["input_ids"],
                box_threshold=self.candidate_floor,
                text_threshold=grounder.config.text_threshold,
                target_sizes=_grounding_target_sizes(sample.width, sample.height),
            )[0]
            boxes = _values(processed.get("boxes", []))
            scores = _values(processed.get("scores", []))
            labels = _values(processed.get("labels", [sample.target] * len(scores)))
            candidates = tuple(
                ModelCandidate(
                    score=float(score),
                    box=tuple(float(value) for value in box),
                    label=str(label).strip(" .") or sample.target,
                )
                for box, score, label in zip(boxes, scores, labels)
            )
            prepared.close()
            return CandidateInference(candidates=candidates, device=str(grounder._device), checkpoint=grounder.checkpoint)
        except GroundingEvaluationError:
            raise
        except Exception as error:
            raise GroundingEvaluationError(f"Grounding evaluation inference failed for sample '{sample.sample_id}'.") from error
