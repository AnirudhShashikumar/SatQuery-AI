"""Lazy local open-vocabulary grounding with the official Grounding DINO checkpoint."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..entity_registry import canonical_grounding_prompt
from ..image_ingestion import save_preview
from ..models import (
    Confidence,
    ConfidenceLevel,
    GroundingCandidateQuality,
    GroundingDetection,
    GroundingInputDetails,
    GroundingQualityPolicy,
    GroundingResult,
    ImageMetadata,
    Modality,
    ModelProvenance,
    RejectedGroundingCandidate,
    SpecialistHealth,
)


logger = logging.getLogger(__name__)

CHECKPOINT = "IDEA-Research/grounding-dino-tiny"
BASE_ARCHITECTURE = "Grounding DINO (Swin-T)"
MODEL_LICENSE = "Apache-2.0"
MODEL_SOURCE = "https://huggingface.co/IDEA-Research/grounding-dino-tiny"
ADAPTATION_DATASET = "Not applicable (general-domain zero-shot detector)"
SUPPORTED_MODALITIES = {Modality.OPTICAL, Modality.MULTISPECTRAL}
LIMITATIONS = [
    "Grounding DINO is an open-vocabulary detector and is not fine-tuned or calibrated for remote-sensing imagery.",
    "Small, densely packed, low-contrast, or unusual-scale objects may be missed or localized imprecisely.",
    "Clouds, shadows, roofs, terrain, seasonal appearance, and sensor differences can cause false or missed detections.",
    "Detection scores are text-region alignment scores, not calibrated scientific confidence or ground-truth probability.",
    "Predicted boxes are model-produced evidence and must be reviewed against the source imagery.",
    "Grounding currently returns model-produced bounding boxes; SAM/SAM2 mask refinement is not connected and remains a planned enhancement.",
]


@dataclass(frozen=True)
class GroundingConfig:
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    maximum_detections: int = 20
    maximum_source_dimension: int = 1024


@dataclass(frozen=True)
class GroundingReliabilityPolicy:
    minimum_alignment_score: float = 0.45
    maximum_localized_area_ratio: float = 0.85
    localized_targets: Tuple[str, ...] = (
        "water body",
        "river",
        "lake",
        "building",
        "buildings",
        "road",
        "bridge",
        "aircraft",
        "airplane",
        "stadium",
        "ship",
    )
    calibration_status: str = "Operational reliability gates pending benchmark calibration."


DEFAULT_CONFIG = GroundingConfig()
DEFAULT_RELIABILITY_POLICY = GroundingReliabilityPolicy()
RAW_BOX_FORMAT = "normalized_cxcywh"
POSTPROCESSED_BOX_FORMAT = "source_pixel_xyxy"
_DIAGNOSTIC_BOX_LIMIT = 8


class GrounderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_INTENT_PREFIXES = (
    r"highlight(?:\s+all)?",
    r"locate(?:\s+all)?",
    r"show\s+me\s+where(?:\s+(?:the|a|an))?",
    r"show\s+where(?:\s+(?:the|a|an))?",
    r"show\s+me(?:\s+(?:the|a|an))?",
    r"show(?:\s+(?:the|a|an))?",
    r"mark(?:\s+all)?",
    r"find(?:\s+(?:the|a|an))?",
    r"where\s+(?:is|are)(?:\s+(?:the|a|an))?",
    r"identify\s+the\s+region\s+containing(?:\s+(?:the|a|an))?",
)
_REJECTED_QUERY = re.compile(
    r"\b(why|cause|caused|latitude|longitude|coordinates?|who owns|ownership|whose|how many|count|number of|identity)\b"
)


def grounder_enabled() -> bool:
    return os.getenv("SATQUERY_GROUNDER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def grounder_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "transformers", "safetensors"))


def grounder_configured() -> bool:
    return grounder_enabled() and bool(os.getenv("SATQUERY_GROUNDER_CHECKPOINT", CHECKPOINT).strip()) and grounder_dependencies_available()


def safe_grounding_parameters(
    config: GroundingConfig = DEFAULT_CONFIG,
    policy: GroundingReliabilityPolicy = DEFAULT_RELIABILITY_POLICY,
) -> Dict[str, Any]:
    return {
        **asdict(config),
        "minimum_reliable_alignment_score": policy.minimum_alignment_score,
        "maximum_localized_area_ratio": policy.maximum_localized_area_ratio,
        "quality_policy_label": policy.calibration_status,
    }


def normalize_grounding_target(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    normalized = normalized.strip(" .!?;:\"'")
    if not normalized:
        raise GrounderError("EMPTY_GROUNDING_TARGET", "A grounding target phrase is required.")
    if _REJECTED_QUERY.search(normalized):
        raise GrounderError(
            "UNSUPPORTED_GROUNDING_QUERY",
            "Grounding supports concrete visual targets, not counting, ownership, causality, identity, or exact geographic-coordinate requests.",
        )
    target: Optional[str] = None
    for prefix in _INTENT_PREFIXES:
        match = re.fullmatch(rf"{prefix}\s+(.+)", normalized)
        if match:
            target = match.group(1)
            break
    if target is None:
        raise GrounderError("EMPTY_GROUNDING_TARGET", "Use a grounding request such as highlight, locate, mark, show where, or find a concrete object.")
    target = re.sub(r"^(?:the|a|an)\s+", "", target)
    target = re.sub(r"\s+(?:in|within|on)\s+(?:this|the)\s+(?:image|scene)$", "", target)
    target = re.sub(r"\s+(?:is|are)$", "", target)
    target = target.strip(" .!?;:\"'")
    canonical = canonical_grounding_prompt(target)
    if canonical is None:
        raise GrounderError(
            "UNSUPPORTED_GROUNDING_TARGET",
            "The target is not present in the controlled remote-sensing grounding registry.",
        )
    return canonical


def _duration(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _values(value: Any) -> List[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _grounding_target_sizes(source_width: int, source_height: int) -> Any:
    """Build the explicit Hugging Face target-size tensor in (height, width) order."""
    import torch

    return torch.tensor([[int(source_height), int(source_width)]], dtype=torch.int64)


def _audit_normalized_cxcywh_to_source_xyxy(
    box: Sequence[float], source_width: int, source_height: int
) -> List[float]:
    """Independent audit oracle for the processor contract; not used for inference scaling."""
    if len(box) != 4:
        raise ValueError("A Grounding DINO box must contain four coordinates.")
    center_x, center_y, width, height = (float(value) for value in box)
    return [
        (center_x - width / 2.0) * source_width,
        (center_y - height / 2.0) * source_height,
        (center_x + width / 2.0) * source_width,
        (center_y + height / 2.0) * source_height,
    ]


def _raw_box_is_full_image(box: Sequence[float], tolerance: float = 1e-3) -> bool:
    """Identify an effectively full-frame raw normalized cxcywh model prediction."""
    try:
        left, top, right, bottom = _audit_normalized_cxcywh_to_source_xyxy(box, 1, 1)
    except (TypeError, ValueError):
        return False
    return left <= tolerance and top <= tolerance and right >= 1.0 - tolerance and bottom >= 1.0 - tolerance


def _clip_source_xyxy(
    box: Sequence[float], source_width: int, source_height: int
) -> Tuple[List[int], bool, Optional[str]]:
    """Round source-pixel xyxy once, then clip once while retaining an audit reason."""
    rounded = [int(round(float(value))) for value in box]
    clipped = [
        max(0, min(source_width, rounded[0])),
        max(0, min(source_height, rounded[1])),
        max(0, min(source_width, rounded[2])),
        max(0, min(source_height, rounded[3])),
    ]
    clipping_occurred = clipped != rounded
    reason = "postprocessed_box_outside_source_bounds" if clipping_occurred else None
    return clipped, clipping_occurred, reason


def _raw_candidate_diagnostics(outputs: Any, box_threshold: float) -> List[Dict[str, Any]]:
    """Return a bounded, scalar-only view of raw model candidates for DEBUG logging."""
    try:
        raw_boxes = outputs.pred_boxes.detach()
        raw_scores = outputs.logits.detach().sigmoid().amax(dim=-1)
        candidates: List[Dict[str, Any]] = []
        for raw_box, raw_score in zip(raw_boxes[0], raw_scores[0]):
            score = float(raw_score.cpu())
            if score <= box_threshold:
                continue
            values = [round(float(value), 8) for value in raw_box.cpu().tolist()]
            candidates.append(
                {
                    "box": values,
                    "score": round(score, 8),
                    "model_output_full_image": _raw_box_is_full_image(values),
                }
            )
            if len(candidates) >= _DIAGNOSTIC_BOX_LIMIT:
                break
        return candidates
    except (AttributeError, IndexError, TypeError, ValueError):
        return []


def _log_box_diagnostic(
    *,
    outputs: Any,
    source_width: int,
    source_height: int,
    model_input_width: int,
    model_input_height: int,
    target_sizes: Any,
    processed_boxes: Sequence[Sequence[float]],
    final_box_audits: Sequence[Dict[str, Any]],
    box_threshold: float,
) -> None:
    """Log only bounded box metadata when development/test DEBUG logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Grounding DINO box audit: %s",
        {
            "source_width": source_width,
            "source_height": source_height,
            "processor_input_dimensions": {"height": model_input_height, "width": model_input_width},
            "target_sizes": target_sizes.detach().cpu().tolist(),
            "raw_box_tensor_values": _raw_candidate_diagnostics(outputs, box_threshold),
            "raw_box_format": RAW_BOX_FORMAT,
            "postprocessed_box_values": [
                [round(float(value), 6) for value in box]
                for box in list(processed_boxes)[:_DIAGNOSTIC_BOX_LIMIT]
            ],
            "postprocessed_box_format": POSTPROCESSED_BOX_FORMAT,
            "final_box_values": list(final_box_audits)[:_DIAGNOSTIC_BOX_LIMIT],
            "manual_scaling_occurred": False,
        },
    )


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _pixel_to_world(transform: Sequence[float], x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def _bbox_world(metadata: ImageMetadata, bbox: Sequence[int]) -> Optional[List[float]]:
    if not metadata.crs or not metadata.transform or len(metadata.transform) != 6:
        return None
    left, top, right, bottom = bbox
    corners = [_pixel_to_world(metadata.transform, x, y) for x, y in ((left, top), (right, top), (left, bottom), (right, bottom))]
    xs, ys = zip(*corners)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _quality_metrics(
    raw_box: Sequence[float],
    raw_score: float,
    metadata: ImageMetadata,
    target_phrase: str,
    policy: GroundingReliabilityPolicy,
) -> Tuple[List[Optional[float]], Optional[List[int]], GroundingCandidateQuality]:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = math.nan
    finite_score = math.isfinite(score)
    coordinates: List[Optional[float]] = []
    for value in list(raw_box)[:4]:
        try:
            coordinate = float(value)
        except (TypeError, ValueError):
            coordinate = math.nan
        coordinates.append(coordinate if math.isfinite(coordinate) else None)
    while len(coordinates) < 4:
        coordinates.append(None)
    finite_coordinates = all(value is not None for value in coordinates)
    box_width: Optional[float] = None
    box_height: Optional[float] = None
    box_area: Optional[float] = None
    area_ratio: Optional[float] = None
    clipped_box: Optional[List[int]] = None
    positive_area = False
    in_bounds = False
    if finite_coordinates:
        left, top, right, bottom = (float(value) for value in coordinates)
        box_width = right - left
        box_height = bottom - top
        box_area = box_width * box_height
        positive_area = box_width > 0 and box_height > 0
        area_ratio = box_area / (metadata.width * metadata.height)
        in_bounds = 0 <= left <= metadata.width and 0 <= right <= metadata.width and 0 <= top <= metadata.height and 0 <= bottom <= metadata.height
        clipped_box, _, _ = _clip_source_xyxy([left, top, right, bottom], metadata.width, metadata.height)

    rejection_reasons: List[str] = []
    if not finite_score:
        rejection_reasons.append("non_finite_alignment_score")
    elif score < policy.minimum_alignment_score:
        rejection_reasons.append("alignment_score_below_minimum")
    if not finite_coordinates:
        rejection_reasons.append("non_finite_coordinates")
    elif not positive_area:
        rejection_reasons.append("non_positive_box_area")
    if (
        target_phrase in policy.localized_targets
        and area_ratio is not None
        and area_ratio > policy.maximum_localized_area_ratio
    ):
        rejection_reasons.append("localized_box_area_ratio_above_maximum")

    quality = GroundingCandidateQuality(
        source_width=metadata.width,
        source_height=metadata.height,
        box_width=round(box_width, 6) if box_width is not None else None,
        box_height=round(box_height, 6) if box_height is not None else None,
        box_area=round(box_area, 6) if box_area is not None else None,
        image_area=metadata.width * metadata.height,
        box_area_ratio=round(area_ratio, 8) if area_ratio is not None else None,
        alignment_score=round(score, 8) if finite_score else None,
        finite_score=finite_score,
        finite_coordinates=finite_coordinates,
        positive_area=positive_area,
        in_bounds=in_bounds,
        rejection_reasons=rejection_reasons,
    )
    return coordinates, clipped_box, quality


def _rejected_candidate(
    label: str,
    coordinates: List[Optional[float]],
    clipped_box: Optional[List[int]],
    quality: GroundingCandidateQuality,
) -> RejectedGroundingCandidate:
    score = quality.alignment_score
    return RejectedGroundingCandidate(
        label=label,
        score=min(1.0, max(0.0, score)) if score is not None else None,
        bbox_source_xyxy=[round(value, 6) if value is not None else None for value in coordinates],
        bbox_pixels=clipped_box,
        box_area_ratio=quality.box_area_ratio,
        rejection_reasons=list(quality.rejection_reasons),
        quality=quality,
    )


def evaluate_detection_quality(
    boxes: Iterable[Sequence[float]],
    scores: Iterable[float],
    labels: Iterable[str],
    metadata: ImageMetadata,
    target_phrase: str,
    config: GroundingConfig = DEFAULT_CONFIG,
    policy: GroundingReliabilityPolicy = DEFAULT_RELIABILITY_POLICY,
    audit_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[GroundingDetection], List[RejectedGroundingCandidate]]:
    candidates: List[Tuple[List[int], float, str, GroundingCandidateQuality, List[Optional[float]]]] = []
    rejected: List[RejectedGroundingCandidate] = []
    for raw_box, raw_score, raw_label in zip(boxes, scores, labels):
        label = str(raw_label).strip(" .") or target_phrase
        coordinates, clipped_box, quality = _quality_metrics(raw_box, raw_score, metadata, target_phrase, policy)
        clipping_occurred = False
        clipping_reason = None
        if quality.finite_coordinates and clipped_box is not None:
            _, clipping_occurred, clipping_reason = _clip_source_xyxy(
                [float(value) for value in coordinates], metadata.width, metadata.height
            )
        if audit_records is not None:
            audit_records.append(
                {
                    "postprocessed_box": [round(value, 6) if value is not None else None for value in coordinates],
                    "final_clipped_box": clipped_box,
                    "score": quality.alignment_score,
                    "label": label,
                    "manual_scaling_occurred": False,
                    "clipping_occurred": clipping_occurred,
                    "clipping_reason": clipping_reason,
                    "box_area_ratio": quality.box_area_ratio,
                    "rejection_reasons": list(quality.rejection_reasons),
                }
            )
        if quality.rejection_reasons or clipped_box is None:
            rejected.append(_rejected_candidate(label, coordinates, clipped_box, quality))
            continue
        score = float(quality.alignment_score)
        candidates.append((clipped_box, min(1.0, max(0.0, score)), label, quality, coordinates))
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected: List[Tuple[List[int], float, str, GroundingCandidateQuality, List[Optional[float]]]] = []
    for candidate in candidates:
        if any(_iou(candidate[0], kept[0]) > config.nms_iou_threshold for kept in selected):
            quality = candidate[3].model_copy(update={"rejection_reasons": ["nms_overlap_suppressed"]})
            rejected.append(_rejected_candidate(candidate[2], candidate[4], candidate[0], quality))
            continue
        if len(selected) >= config.maximum_detections:
            quality = candidate[3].model_copy(update={"rejection_reasons": ["maximum_detection_limit"]})
            rejected.append(_rejected_candidate(candidate[2], candidate[4], candidate[0], quality))
            continue
        selected.append(candidate)
    detections: List[GroundingDetection] = []
    for bbox, score, label, quality, _coordinates in selected:
        normalized = [
            bbox[0] / metadata.width,
            bbox[1] / metadata.height,
            bbox[2] / metadata.width,
            bbox[3] / metadata.height,
        ]
        world = _bbox_world(metadata, bbox)
        detections.append(
            GroundingDetection(
                label=label,
                score=round(score, 6),
                bbox_pixels=bbox,
                bbox_normalized=[round(value, 6) for value in normalized],
                bbox_world=world,
                crs=metadata.crs if world is not None else None,
                mask_url=None,
                quality=quality,
            )
        )
    return detections, rejected


def sanitize_detections(
    boxes: Iterable[Sequence[float]],
    scores: Iterable[float],
    labels: Iterable[str],
    metadata: ImageMetadata,
    target_phrase: str,
    config: GroundingConfig = DEFAULT_CONFIG,
    audit_records: Optional[List[Dict[str, Any]]] = None,
) -> List[GroundingDetection]:
    """Backward-compatible accepted-detection view of the reliability evaluation."""
    detections, _ = evaluate_detection_quality(
        boxes,
        scores,
        labels,
        metadata,
        target_phrase,
        config,
        DEFAULT_RELIABILITY_POLICY,
        audit_records,
    )
    return detections


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotated_preview(image: Image.Image, detections: Sequence[GroundingDetection], metadata: ImageMetadata) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    scale_x = output.width / metadata.width
    scale_y = output.height / metadata.height
    line_width = max(2, round(min(output.size) / 220))
    font = _font(max(12, round(min(output.size) / 32)))
    colors = ((39, 196, 255), (255, 165, 55), (82, 224, 139), (238, 92, 216), (255, 222, 73))
    legend_title = "Grounding DINO boxes"
    legend_detail = "label · alignment score"
    legend_boxes = [draw.textbbox((0, 0), text, font=font) for text in (legend_title, legend_detail)]
    legend_width = min(output.width, max(box[2] - box[0] for box in legend_boxes) + 18)
    legend_line_height = max(box[3] - box[1] for box in legend_boxes) + 5
    draw.rectangle((0, 0, legend_width, legend_line_height * 2 + 9), fill=(8, 14, 24))
    draw.text((9, 5), legend_title, fill=(225, 235, 245), font=font)
    draw.text((9, legend_line_height + 4), legend_detail, fill=(139, 205, 236), font=font)
    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        left, top, right, bottom = detection.bbox_pixels
        box = [round(left * scale_x), round(top * scale_y), round(right * scale_x), round(bottom * scale_y)]
        draw.rectangle(box, outline=color, width=line_width)
        text = f"{detection.label} {detection.score:.3f}"
        text_box = draw.textbbox((0, 0), text, font=font)
        label_width = text_box[2] - text_box[0] + 12
        label_height = text_box[3] - text_box[1] + 10
        label_top = max(0, box[1] - label_height)
        draw.rectangle((box[0], label_top, min(output.width, box[0] + label_width), box[1]), fill=(8, 14, 24))
        draw.text((box[0] + 6, label_top + 4), text, fill=color, font=font)
    return output


class RemoteSensingGrounder:
    def __init__(
        self,
        config: GroundingConfig = DEFAULT_CONFIG,
        reliability_policy: GroundingReliabilityPolicy = DEFAULT_RELIABILITY_POLICY,
    ) -> None:
        self.config = config
        self.reliability_policy = reliability_policy
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._state = "unloaded"
        self._device: Optional[str] = None
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._safe_error: Optional[str] = None
        self._load_count = 0

    @property
    def checkpoint(self) -> str:
        return os.getenv("SATQUERY_GROUNDER_CHECKPOINT", CHECKPOINT).strip()

    @property
    def cache_dir(self) -> str:
        configured = os.getenv("SATQUERY_MODEL_CACHE", "").strip()
        return configured or str(Path(tempfile.gettempdir()) / "geovision-satquery-hf")

    @property
    def limitations(self) -> List[str]:
        return list(LIMITATIONS)

    def is_available(self) -> bool:
        return grounder_configured() and self._state != "failed"

    def health(self) -> SpecialistHealth:
        if not grounder_configured():
            return SpecialistHealth(status="disabled", device=None, error=None)
        return SpecialistHealth(status=self._state, device=self._device, error=self._safe_error)

    def _select_device(self, torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load_components(self, local_only: bool) -> Tuple[Any, Any]:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint, cache_dir=self.cache_dir, local_files_only=local_only)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.checkpoint,
            cache_dir=self.cache_dir,
            local_files_only=local_only,
            use_safetensors=True,
        )
        return processor, model

    def load(self) -> None:
        if self._state == "ready":
            return
        if not grounder_configured():
            raise GrounderError("GROUNDER_UNAVAILABLE", "The local Grounding DINO specialist is not configured.")
        with self._load_lock:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise GrounderError("GROUNDER_LOAD_FAILED", "The local Grounding DINO checkpoint could not be loaded.")
            self._state = "loading"
            self._safe_error = None
            try:
                import torch

                local_only = os.getenv("SATQUERY_GROUNDER_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
                processor, model = self._load_components(local_only)
                device = self._select_device(torch)
                model.eval()
                model.to(device)
                self._processor = processor
                self._model = model
                self._device = device
                self._load_count += 1
                self._state = "ready"
            except Exception as error:
                self._processor = None
                self._model = None
                self._device = None
                self._state = "failed"
                self._safe_error = "Checkpoint loading failed."
                raise GrounderError("GROUNDER_LOAD_FAILED", "The local Grounding DINO checkpoint could not be loaded.") from error

    def ground(
        self,
        image: Image.Image,
        text_query: str,
        metadata: ImageMetadata,
        modality: Modality,
        bands_used: List[str],
        image_representation: str,
    ) -> GroundingResult:
        total_started = time.perf_counter()
        target_started = time.perf_counter()
        target_phrase = normalize_grounding_target(text_query)
        stage_durations = {"target_phrase_extraction": _duration(target_started)}
        if modality not in SUPPORTED_MODALITIES:
            raise GrounderError("UNSUPPORTED_GROUNDING_MODALITY", "Grounding supports optical and RGB-like multispectral imagery only; SAR grounding is not implemented.")
        if metadata.band_count < 3:
            raise GrounderError("UNSUPPORTED_GROUNDING_BANDS", "Grounding requires an RGB or documented RGB-like representation with at least three bands.")

        was_ready = self._state == "ready"
        load_started = time.perf_counter()
        self.load()
        model_load_ms = 0 if was_ready else _duration(load_started)
        stage_durations["grounder_model_load"] = model_load_ms
        if self._processor is None or self._model is None or self._device is None:
            raise GrounderError("GROUNDER_UNAVAILABLE", "The local Grounding DINO specialist is unavailable.")

        import torch

        preparation_started = time.perf_counter()
        prepared = image.convert("RGB")
        if max(prepared.size) > self.config.maximum_source_dimension:
            prepared.thumbnail((self.config.maximum_source_dimension, self.config.maximum_source_dimension), Image.Resampling.LANCZOS)
        prompt = f"{target_phrase.lower().rstrip('.')}."
        inputs = self._processor(images=prepared, text=prompt, return_tensors="pt")
        model_input_height, model_input_width = (int(value) for value in inputs["pixel_values"].shape[-2:])
        stage_durations["grounding_image_preparation"] = _duration(preparation_started)
        warnings = list(metadata.warnings)

        inference_started = time.perf_counter()
        with self._inference_lock:
            try:
                device_inputs = {name: value.to(self._device) for name, value in inputs.items()}
                with torch.inference_mode():
                    outputs = self._model(**device_inputs)
            except Exception as error:
                if self._device != "mps":
                    raise GrounderError("GROUNDING_INFERENCE_FAILED", "Grounding DINO inference failed safely.") from error
                self._model.to("cpu")
                self._device = "cpu"
                device_inputs = {name: value.to("cpu") for name, value in inputs.items()}
                with torch.inference_mode():
                    outputs = self._model(**device_inputs)
                warnings.append("Grounding DINO fell back to CPU after an incompatible MPS operation.")
        stage_durations["grounding_inference"] = _duration(inference_started)

        post_started = time.perf_counter()
        target_sizes = _grounding_target_sizes(metadata.width, metadata.height)
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            device_inputs["input_ids"],
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = _values(processed.get("boxes", []))
        scores = _values(processed.get("scores", []))
        labels = _values(processed.get("labels", [target_phrase] * len(scores)))
        stage_durations["grounding_postprocessing"] = _duration(post_started)

        quality_started = time.perf_counter()
        final_box_audits: Optional[List[Dict[str, Any]]] = [] if logger.isEnabledFor(logging.DEBUG) else None
        detections, rejected_candidates = evaluate_detection_quality(
            boxes,
            scores,
            labels,
            metadata,
            target_phrase,
            self.config,
            self.reliability_policy,
            audit_records=final_box_audits,
        )
        if final_box_audits is not None:
            _log_box_diagnostic(
                outputs=outputs,
                source_width=metadata.width,
                source_height=metadata.height,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                target_sizes=target_sizes,
                processed_boxes=boxes,
                final_box_audits=final_box_audits,
                box_threshold=self.config.box_threshold,
            )
        stage_durations["grounding_quality_filter"] = _duration(quality_started)

        preview_started = time.perf_counter()
        annotated_preview_url: Optional[str] = None
        if detections:
            annotated = _annotated_preview(prepared, detections, metadata)
            try:
                annotated_preview_url = save_preview(annotated)
            finally:
                annotated.close()
        stage_durations["grounding_preview_generation"] = _duration(preview_started)

        if detections:
            top_score = detections[0].score
            confidence = Confidence(
                level=ConfidenceLevel.MODERATE if top_score >= 0.6 else ConfidenceLevel.LOW,
                score=top_score,
                reason="The score is Grounding DINO text-region alignment, not calibrated confidence for remote-sensing imagery.",
            )
        else:
            confidence = Confidence(
                level=ConfidenceLevel.UNAVAILABLE,
                score=None,
                reason="No model candidate met the operational reliability gates for precise localization; no box was invented.",
            )
            warnings.append(f"No reliable localized region was found for '{target_phrase}'.")

        empty_result_explanation: Optional[str] = None
        if not detections:
            empty_result_explanation = (
                f"No reliable localized region was found for '{target_phrase}'. "
                "The model produced a weak scene-level match rather than a precise object region."
                if rejected_candidates
                else f"No confident region found for '{target_phrase}'."
            )

        quality_policy = GroundingQualityPolicy(
            minimum_alignment_score=self.reliability_policy.minimum_alignment_score,
            maximum_localized_area_ratio=self.reliability_policy.maximum_localized_area_ratio,
            localized_targets=list(self.reliability_policy.localized_targets),
            calibration_status=self.reliability_policy.calibration_status,
        )

        return GroundingResult(
            original_query=text_query,
            target_phrase=target_phrase,
            detections=detections,
            accepted_detections=detections,
            rejected_candidates=rejected_candidates,
            accepted_detection_count=len(detections),
            rejected_candidate_count=len(rejected_candidates),
            quality_policy=quality_policy,
            empty_result_explanation=empty_result_explanation,
            operational_threshold_disclaimer=self.reliability_policy.calibration_status,
            annotated_preview_url=annotated_preview_url,
            confidence=confidence,
            model=ModelProvenance(
                tool_id="rs_grounder",
                checkpoint=self.checkpoint,
                base_architecture=BASE_ARCHITECTURE,
                adaptation_dataset=ADAPTATION_DATASET,
                remote_sensing_adapted=False,
                license=MODEL_LICENSE,
                source=MODEL_SOURCE,
            ),
            input=GroundingInputDetails(
                modality=modality,
                bands_used=bands_used,
                representation=image_representation,
                original_width=metadata.width,
                original_height=metadata.height,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                normalization_method="Official Grounding DINO processor resize and ImageNet normalization",
            ),
            device=self._device,
            warnings=list(dict.fromkeys(warnings)),
            limitations=self.limitations,
            runtime_ms=_duration(total_started),
            model_load_ms=model_load_ms,
            model_reused=was_ready,
            stage_durations_ms=stage_durations,
        )

    def reset_for_tests(self) -> None:
        with self._load_lock:
            self._state = "unloaded"
            self._device = None
            self._processor = None
            self._model = None
            self._safe_error = None
            self._load_count = 0


_GROUNDER = RemoteSensingGrounder()


def get_grounder() -> RemoteSensingGrounder:
    return _GROUNDER
