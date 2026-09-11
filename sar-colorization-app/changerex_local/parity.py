"""Strict comparison helpers for official-reference parity artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ParityThresholds:
    binary_mask_agreement_minimum: float = 0.999
    mask_iou_minimum: float = 0.998
    changed_percentage_difference_maximum: float = 0.1
    probability_max_absolute_difference_maximum: float = 1e-4
    probability_mean_absolute_difference_maximum: float = 1e-5


@dataclass(frozen=True)
class ParityResult:
    probability_shape_match: bool
    mask_shape_match: bool
    probability_max_absolute_difference: float
    probability_mean_absolute_difference: float
    binary_mask_agreement: float
    mask_iou: float
    changed_percentage_difference: float
    floating_probability_pass: bool
    mask_pass: bool
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_outputs(
    extracted_probability: np.ndarray,
    extracted_mask: np.ndarray,
    reference_probability: np.ndarray,
    reference_mask: np.ndarray,
    *,
    thresholds: ParityThresholds = ParityThresholds(),
) -> ParityResult:
    probability_shape_match = extracted_probability.shape == reference_probability.shape
    mask_shape_match = extracted_mask.shape == reference_mask.shape
    if not probability_shape_match or not mask_shape_match:
        return ParityResult(
            probability_shape_match,
            mask_shape_match,
            float("inf"),
            float("inf"),
            0.0,
            0.0,
            float("inf"),
            False,
            False,
            False,
        )
    extracted_probability = np.asarray(extracted_probability, dtype=np.float64)
    reference_probability = np.asarray(reference_probability, dtype=np.float64)
    extracted_mask = np.asarray(extracted_mask).astype(bool)
    reference_mask = np.asarray(reference_mask).astype(bool)
    difference = np.abs(extracted_probability - reference_probability)
    agreement = float(np.mean(extracted_mask == reference_mask))
    intersection = int(np.logical_and(extracted_mask, reference_mask).sum())
    union = int(np.logical_or(extracted_mask, reference_mask).sum())
    iou = 1.0 if union == 0 else intersection / union
    extracted_changed = 100.0 * float(extracted_mask.mean())
    reference_changed = 100.0 * float(reference_mask.mean())
    changed_difference = abs(extracted_changed - reference_changed)
    maximum = float(difference.max()) if difference.size else 0.0
    mean = float(difference.mean()) if difference.size else 0.0
    floating_pass = (
        maximum <= thresholds.probability_max_absolute_difference_maximum
        and mean <= thresholds.probability_mean_absolute_difference_maximum
    )
    mask_pass = (
        agreement >= thresholds.binary_mask_agreement_minimum
        and iou >= thresholds.mask_iou_minimum
        and changed_difference <= thresholds.changed_percentage_difference_maximum
    )
    return ParityResult(
        probability_shape_match=True,
        mask_shape_match=True,
        probability_max_absolute_difference=maximum,
        probability_mean_absolute_difference=mean,
        binary_mask_agreement=agreement,
        mask_iou=iou,
        changed_percentage_difference=changed_difference,
        floating_probability_pass=floating_pass,
        mask_pass=mask_pass,
        accepted=floating_pass and mask_pass,
    )


def save_parity_report(
    path: str | Path,
    result: ParityResult,
    *,
    thresholds: ParityThresholds = ParityThresholds(),
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "result": result.to_dict(),
        "thresholds": asdict(thresholds),
        "metadata": metadata or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
