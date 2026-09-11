"""Grounding Specialist v1.1 training primitives.

This package owns every component needed by the trainer. It has no imports
from the SatQuery application, serving layer, or benchmark evaluators.
The head definition remains state-dict compatible with Grounding Specialist v1.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"
TRAINING_RECIPE_VERSION = "v1.1"
DEFAULT_AREA_REGULARIZATION_WEIGHT = 0.05
DEFAULT_QUERY_ENTROPY_WEIGHT = 0.01
CLASSIFICATION_WEIGHT = 1.0
L1_WEIGHT = 5.0
GIOU_WEIGHT = 2.0


class SatQueryGroundingHead(nn.Module):
    """Exact Grounding Specialist v1 exported trainable architecture."""

    def __init__(
        self,
        hidden_dim: int = 256,
        intermediate_dim: int = 256,
        max_box_delta: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_box_delta = float(max_box_delta)
        self.query_scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1),
        )
        self.box_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 4),
            nn.Tanh(),
        )

    def forward(self, hidden_state: Tensor, base_boxes_cxcywh: Tensor) -> tuple[Tensor, Tensor]:
        query_logits = self.query_scorer(hidden_state).squeeze(-1)
        refined = base_boxes_cxcywh + self.box_refiner(hidden_state) * self.max_box_delta
        centers = refined[..., :2].clamp(0.0, 1.0)
        sizes = refined[..., 2:].clamp(1e-4, 1.0)
        return query_logits, torch.cat((centers, sizes), dim=-1)


def load_frozen_base(
    device: torch.device,
    *,
    local_files_only: bool = False,
    cache_dir: Optional[Path] = None,
) -> tuple[Any, nn.Module]:
    """Load one frozen Grounding DINO base directly through Transformers."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    common = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    processor = AutoProcessor.from_pretrained(BASE_CHECKPOINT, **common)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        BASE_CHECKPOINT,
        use_safetensors=True,
        **common,
    )
    model.to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Grounding DINO base model was not fully frozen")
    return processor, model


def inverse_frequency_sample_weights(labels: Sequence[str]) -> Tensor:
    """Return one double-precision inverse-frequency weight per sample."""
    if not labels:
        raise ValueError("At least one training label is required")
    normalized = [str(label) for label in labels]
    counts = Counter(normalized)
    return torch.tensor([1.0 / counts[label] for label in normalized], dtype=torch.double)


def build_weighted_sampler(
    labels: Sequence[str], *, generator: Optional[torch.Generator] = None
) -> WeightedRandomSampler:
    """Build the training-only class-balanced sampler.

    The total sampling mass is identical for every class because each sample is
    weighted by the inverse of its class frequency.  Sampling is with
    replacement and retains the original epoch length.
    """
    weights = inverse_frequency_sample_weights(labels)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_data_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    training: bool,
    collate_fn: Optional[Callable[[list[Any]], Any]] = None,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
) -> DataLoader[Any]:
    """Build a balanced training loader or unchanged sequential validation loader."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if training:
        labels = getattr(dataset, "class_labels", None)
        if labels is None:
            raise ValueError("Training dataset must expose class_labels")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=build_weighted_sampler(labels, generator=generator),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _normalized_xyxy(values: Sequence[Any]) -> list[float]:
    if len(values) != 4:
        raise ValueError("ground_truth_bbox must contain four coordinates")
    box = [float(value) for value in values]
    if not all(math.isfinite(value) for value in box):
        raise ValueError("ground_truth_bbox contains a non-finite coordinate")
    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"ground_truth_bbox must be normalized xyxy, received {box}")
    return box


def load_manifest_records(manifest: Path, image_root: Path) -> list[dict[str, Any]]:
    """Load a JSONL grounding split without imposing Smoke-100 cardinality rules."""
    records: list[dict[str, Any]] = []
    required = {"object_class", "referring_sentence", "ground_truth_bbox"}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {manifest}:{line_number}") from error
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"Manifest line {line_number} lacks required grounding fields")
            image_name = record.get("image") or Path(str(record.get("local_image_path", ""))).name
            if not image_name:
                raise ValueError(f"Manifest line {line_number} lacks an image path")
            image_path = image_root / Path(str(image_name)).name
            if not image_path.is_file():
                local_path = Path(str(record.get("local_image_path", "")))
                image_path = local_path if local_path.is_file() else image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing training image: {image_path}")
            records.append(
                {
                    **record,
                    "resolved_image_path": str(image_path),
                    "ground_truth_bbox": _normalized_xyxy(record["ground_truth_bbox"]),
                }
            )
    if not records:
        raise ValueError(f"No records found in {manifest}")
    return records


class GroundingManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(record) for record in records]
        self.class_labels = [str(record["object_class"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(str(record["resolved_image_path"])) as source:
            image = source.convert("RGB")
        return {
            "image": image,
            "text": str(record["referring_sentence"]),
            "ground_truth_box": torch.tensor(record["ground_truth_bbox"], dtype=torch.float32),
            "object_class": str(record["object_class"]),
            "sample_id": str(record.get("smoke_id", record.get("id", index))),
        }


class GroundingBatchCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]:
        encoded = self.processor(
            images=[sample["image"] for sample in samples],
            text=[sample["text"] for sample in samples],
            return_tensors="pt",
            padding=True,
        )
        return {
            "model_inputs": dict(encoded),
            "ground_truth_boxes": torch.stack([sample["ground_truth_box"] for sample in samples]),
            "object_classes": [sample["object_class"] for sample in samples],
            "sample_ids": [sample["sample_id"] for sample in samples],
        }


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    centers = boxes[..., :2].clamp(0.0, 1.0)
    sizes = boxes[..., 2:].clamp(1e-4, 1.0)
    half = sizes / 2.0
    return torch.cat((centers - half, centers + half), dim=-1).clamp(0.0, 1.0)


def box_areas_xyxy(boxes: Tensor) -> Tensor:
    return (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0).prod(dim=-1)


def pairwise_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    """IoU for boxes [B,Q,4] against one target [B,4] per sample."""
    if boxes.ndim != 3 or targets.ndim != 2 or boxes.shape[0] != targets.shape[0]:
        raise ValueError("Expected boxes [batch, queries, 4] and targets [batch, 4]")
    target = targets[:, None, :]
    intersection_corners = torch.minimum(boxes[..., 2:], target[..., 2:]) - torch.maximum(
        boxes[..., :2], target[..., :2]
    )
    intersection = intersection_corners.clamp(min=0.0).prod(dim=-1)
    union = box_areas_xyxy(boxes) + box_areas_xyxy(target) - intersection
    return torch.where(union > 0.0, intersection / union, torch.zeros_like(union))


def paired_generalized_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    """Generalized IoU for aligned normalized xyxy box tensors."""
    intersection_corners = torch.minimum(boxes[..., 2:], targets[..., 2:]) - torch.maximum(
        boxes[..., :2], targets[..., :2]
    )
    intersection = intersection_corners.clamp(min=0.0).prod(dim=-1)
    union = box_areas_xyxy(boxes) + box_areas_xyxy(targets) - intersection
    iou = torch.where(union > 0.0, intersection / union, torch.zeros_like(union))
    enclosure = (
        torch.maximum(boxes[..., 2:], targets[..., 2:])
        - torch.minimum(boxes[..., :2], targets[..., :2])
    ).clamp(min=0.0).prod(dim=-1)
    return iou - torch.where(enclosure > 0.0, (enclosure - union) / enclosure, torch.zeros_like(enclosure))


def matched_query_indices(base_boxes_cxcywh: Tensor, target_boxes_xyxy: Tensor) -> Tensor:
    """Match each target to the frozen base query with greatest IoU."""
    return pairwise_iou(cxcywh_to_xyxy(base_boxes_cxcywh), target_boxes_xyxy).argmax(dim=1)


def box_area_regularization_loss(predicted_boxes_cxcywh: Tensor, target_boxes_xyxy: Tensor) -> Tensor:
    """L1 area penalty for aligned boxes, kept unweighted for transparent logging."""
    predicted_areas = box_areas_xyxy(cxcywh_to_xyxy(predicted_boxes_cxcywh))
    target_areas = box_areas_xyxy(target_boxes_xyxy)
    return F.l1_loss(predicted_areas, target_areas)


def query_diversity_loss(query_logits: Tensor, epsilon: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Penalize low entropy in the batch-marginal query selection distribution.

    Returns ``(loss, entropy)``.  Entropy is normalized to [0, 1], so the loss
    is also bounded to [0, 1] and its small coefficient cannot dominate the
    detection objectives.
    """
    if query_logits.ndim != 2 or query_logits.shape[1] < 2:
        raise ValueError("query_logits must have shape [batch, queries] with at least two queries")
    marginal = query_logits.softmax(dim=1).mean(dim=0)
    entropy = -(marginal * marginal.clamp_min(epsilon).log()).sum()
    normalized_entropy = entropy / math.log(query_logits.shape[1])
    return (1.0 - normalized_entropy).clamp(min=0.0, max=1.0), normalized_entropy


def grounding_specialist_losses(
    query_logits: Tensor,
    refined_boxes_cxcywh: Tensor,
    base_boxes_cxcywh: Tensor,
    target_boxes_xyxy: Tensor,
    *,
    area_regularization_weight: float = DEFAULT_AREA_REGULARIZATION_WEIGHT,
    query_entropy_weight: float = DEFAULT_QUERY_ENTROPY_WEIGHT,
) -> dict[str, Tensor]:
    """Compute the unchanged pilot losses plus v1.1 regularizers."""
    if area_regularization_weight < 0.0 or query_entropy_weight < 0.0:
        raise ValueError("Regularization weights must be non-negative")
    indices = matched_query_indices(base_boxes_cxcywh.detach(), target_boxes_xyxy)
    batch_indices = torch.arange(query_logits.shape[0], device=query_logits.device)
    selected_boxes = refined_boxes_cxcywh[batch_indices, indices]
    selected_xyxy = cxcywh_to_xyxy(selected_boxes)

    query_targets = torch.zeros_like(query_logits)
    query_targets[batch_indices, indices] = 1.0
    classification_loss = F.binary_cross_entropy_with_logits(query_logits, query_targets)
    l1_loss = F.l1_loss(selected_xyxy, target_boxes_xyxy)
    giou_loss = (1.0 - paired_generalized_iou(selected_xyxy, target_boxes_xyxy)).mean()
    area_loss = box_area_regularization_loss(selected_boxes, target_boxes_xyxy)
    diversity_loss, query_entropy = query_diversity_loss(query_logits)
    total = (
        CLASSIFICATION_WEIGHT * classification_loss
        + L1_WEIGHT * l1_loss
        + GIOU_WEIGHT * giou_loss
        + float(area_regularization_weight) * area_loss
        + float(query_entropy_weight) * diversity_loss
    )
    return {
        "loss": total,
        "classification_loss": classification_loss,
        "l1_loss": l1_loss,
        "giou_loss": giou_loss,
        "area_loss": area_loss,
        "diversity_loss": diversity_loss,
        "query_entropy": query_entropy,
        "matched_query_indices": indices,
    }


@dataclass
class ValidationAccumulator:
    ious: list[float]
    predicted_areas: list[float]
    ground_truth_areas: list[float]
    query_indices: list[int]
    confidences: list[float]

    @classmethod
    def empty(cls) -> "ValidationAccumulator":
        return cls([], [], [], [], [])

    def update(
        self,
        query_logits: Tensor,
        refined_boxes_cxcywh: Tensor,
        target_boxes_xyxy: Tensor,
    ) -> None:
        indices = query_logits.argmax(dim=1)
        batch_indices = torch.arange(query_logits.shape[0], device=query_logits.device)
        selected = refined_boxes_cxcywh[batch_indices, indices]
        selected_xyxy = cxcywh_to_xyxy(selected)
        ious = pairwise_iou(selected_xyxy[:, None, :], target_boxes_xyxy).squeeze(1)
        self.ious.extend(float(value) for value in ious.detach().cpu())
        self.predicted_areas.extend(float(value) for value in box_areas_xyxy(selected_xyxy).detach().cpu())
        self.ground_truth_areas.extend(float(value) for value in box_areas_xyxy(target_boxes_xyxy).detach().cpu())
        self.query_indices.extend(int(value) for value in indices.detach().cpu())
        selected_logits = query_logits[batch_indices, indices]
        self.confidences.extend(float(value) for value in selected_logits.sigmoid().detach().cpu())

    def metrics(self) -> dict[str, float | int]:
        if not self.ious:
            raise ValueError("Validation produced no samples")
        counts = Counter(self.query_indices)
        total = len(self.query_indices)
        probabilities = [count / total for count in counts.values()]
        selection_entropy = -sum(probability * math.log(probability) for probability in probabilities)
        ranked_counts = sorted(counts.values(), reverse=True)
        ordered_ious = sorted(self.ious)
        middle = len(ordered_ious) // 2
        median = (
            ordered_ious[middle]
            if len(ordered_ious) % 2
            else (ordered_ious[middle - 1] + ordered_ious[middle]) / 2.0
        )
        return {
            "samples": len(self.ious),
            "mean_iou": sum(self.ious) / len(self.ious),
            "median_iou": median,
            "accuracy_at_025": sum(value >= 0.25 for value in self.ious) / len(self.ious),
            "accuracy_at_050": sum(value >= 0.50 for value in self.ious) / len(self.ious),
            "accuracy_at_075": sum(value >= 0.75 for value in self.ious) / len(self.ious),
            "average_predicted_box_area": sum(self.predicted_areas) / len(self.predicted_areas),
            "average_ground_truth_box_area": sum(self.ground_truth_areas) / len(self.ground_truth_areas),
            "top_1_query_usage": ranked_counts[0] / total,
            "top_5_query_coverage": sum(ranked_counts[:5]) / total,
            "query_entropy": selection_entropy,
            "unique_queries_used": len(counts),
            "average_confidence": sum(self.confidences) / len(self.confidences),
        }


@dataclass
class EarlyStopping:
    patience: int = 5
    best_accuracy50: float = -math.inf
    validations_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def update(self, accuracy50: float) -> tuple[bool, bool]:
        if not math.isfinite(accuracy50):
            raise ValueError("accuracy50 must be finite")
        improved = accuracy50 > self.best_accuracy50
        if improved:
            self.best_accuracy50 = accuracy50
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1
        return improved, self.validations_without_improvement >= self.patience


@dataclass(frozen=True)
class LoadedCheckpoint:
    step: int
    history: list[dict[str, Any]]
    validation: dict[str, Any]
    configuration: dict[str, Any]
    metadata: dict[str, Any]


def checkpoint_payload(
    *,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    history: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    area_regularization_weight: float,
    query_entropy_weight: float,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create a v1.1 checkpoint while preserving all v1 top-level fields."""
    payload = {
        "model_name": "SatQuery Grounding Specialist v1.1",
        "step": int(step),
        "base_model": BASE_CHECKPOINT,
        "specialist_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": [dict(row) for row in history],
        "validation": dict(validation),
        "configuration": dict(configuration),
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "balanced_sampling": True,
        "area_regularization_weight": float(area_regularization_weight),
        "query_entropy_weight": float(query_entropy_weight),
    }
    if run_metadata:
        payload.update(dict(run_metadata))
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path: Path,
    head: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    require_training_state: bool = False,
) -> LoadedCheckpoint:
    """Strictly load v1/v1.1 head weights; absent v1.1 metadata gets safe defaults."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("specialist_state_dict"), dict):
        raise ValueError("Checkpoint lacks specialist_state_dict")
    if payload.get("base_model") != BASE_CHECKPOINT:
        raise ValueError(f"Checkpoint base_model must be {BASE_CHECKPOINT}")
    head.load_state_dict(payload["specialist_state_dict"], strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    if require_training_state and not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint lacks optimizer_state_dict required by --resume-optimizer")
    if require_training_state and not isinstance(scheduler_state, dict):
        raise ValueError("Checkpoint lacks scheduler_state_dict required by --resume-optimizer")
    if optimizer is not None and isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    metadata = {
        "training_recipe_version": payload.get("training_recipe_version", "v1.0"),
        "balanced_sampling": bool(payload.get("balanced_sampling", False)),
        "area_regularization_weight": float(payload.get("area_regularization_weight", 0.0)),
        "query_entropy_weight": float(payload.get("query_entropy_weight", 0.0)),
    }
    for key in ("resume_mode", "source_checkpoint", "source_checkpoint_step", "starting_global_step"):
        if key in payload:
            metadata[key] = payload[key]
    return LoadedCheckpoint(
        step=int(payload.get("step", 0)),
        history=[dict(row) for row in payload.get("history", [])],
        validation=dict(payload.get("validation", {})),
        configuration=dict(payload.get("configuration", {})),
        metadata=metadata,
    )


def move_model_inputs(model_inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def validate(
    base_model: nn.Module,
    head: SatQueryGroundingHead,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | int]:
    base_model.eval()
    head.eval()
    accumulator = ValidationAccumulator.empty()
    with torch.no_grad():
        for batch in loader:
            inputs = move_model_inputs(batch["model_inputs"], device)
            targets = batch["ground_truth_boxes"].to(device)
            outputs = base_model(**inputs)
            query_logits, refined_boxes = head(outputs.last_hidden_state, outputs.pred_boxes)
            accumulator.update(query_logits, refined_boxes, targets)
    return accumulator.metrics()


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    minimum_learning_rate_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= minimum_learning_rate_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")

    def multiplier(step: int) -> float:
        progress = min(max(step / total_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_learning_rate_ratio + (1.0 - minimum_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


__all__ = [
    "BASE_CHECKPOINT",
    "DEFAULT_AREA_REGULARIZATION_WEIGHT",
    "DEFAULT_QUERY_ENTROPY_WEIGHT",
    "EarlyStopping",
    "GroundingBatchCollator",
    "GroundingManifestDataset",
    "LoadedCheckpoint",
    "SatQueryGroundingHead",
    "TRAINING_RECIPE_VERSION",
    "ValidationAccumulator",
    "box_area_regularization_loss",
    "build_data_loader",
    "build_weighted_sampler",
    "checkpoint_payload",
    "grounding_specialist_losses",
    "inverse_frequency_sample_weights",
    "load_manifest_records",
    "load_frozen_base",
    "load_training_checkpoint",
    "make_cosine_scheduler",
    "move_model_inputs",
    "save_checkpoint",
    "validate",
]
