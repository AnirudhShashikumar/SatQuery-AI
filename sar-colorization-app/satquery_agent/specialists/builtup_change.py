"""Conservative region-level built-up evidence over real change geometry.

The change mask supplies geometry only. Direction is assigned only when before/
after optical structural evidence changes consistently inside that geometry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..image_ingestion import save_preview
from ..models import BuiltUpChangeAssessment, BuiltUpRegionEvidence, ChangeStatistics


STRUCTURAL_DIRECTION_DELTA = 0.06
MIN_BUILT_UP_SUPPORT = 0.20
MEANINGFUL_REGION_PERCENT = 0.02


def _location(left: int, top: int, right: int, bottom: int, width: int, height: int) -> str:
    x = (left + right) / (2 * max(width, 1))
    y = (top + bottom) / (2 * max(height, 1))
    horizontal = "west" if x < 1 / 3 else "east" if x > 2 / 3 else "center"
    vertical = "north" if y < 1 / 3 else "south" if y > 2 / 3 else "center"
    if horizontal == vertical == "center":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "center":
        return horizontal
    return f"{vertical}{horizontal}"


def _scores(rgb: np.ndarray, selection: np.ndarray) -> tuple[float, float]:
    values = rgb.astype(np.float32) / 255.0
    gray = values[..., :3].mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gradient = np.sqrt(gx * gx + gy * gy)
    selected_gradient = gradient[selection]
    if selected_gradient.size == 0:
        return 0.0, 0.0
    threshold = max(0.08, float(np.median(selected_gradient) + 0.5 * np.std(selected_gradient)))
    edge_density = float(np.mean(selected_gradient >= threshold))
    mean_gradient = float(np.mean(selected_gradient))
    texture = min(1.0, float(np.std(gray[selection])) / 0.25)
    structural = min(1.0, 0.50 * min(edge_density / 0.25, 1.0) + 0.30 * min(mean_gradient / 0.18, 1.0) + 0.20 * texture)
    red, green, blue = (values[..., index][selection] for index in range(3))
    vegetation_fraction = float(np.mean((green > red * 1.08) & (green > blue * 1.05)))
    built_up_like = structural * (1.0 - 0.5 * vegetation_fraction)
    return round(structural, 6), round(min(1.0, max(0.0, built_up_like)), 6)


def _direction(before: float, after: float) -> tuple[str, str]:
    delta = after - before
    if delta >= STRUCTURAL_DIRECTION_DELTA and after >= MIN_BUILT_UP_SUPPORT:
        return "INCREASE_SUPPORTED", "supported"
    if delta <= -STRUCTURAL_DIRECTION_DELTA and before >= MIN_BUILT_UP_SUPPORT:
        return "DECREASE_SUPPORTED", "supported"
    if abs(delta) >= STRUCTURAL_DIRECTION_DELTA / 2:
        return "INSUFFICIENT_EVIDENCE", "uncertain"
    return "NO_DIRECTIONAL_EVIDENCE", "limited"


def assess_built_up_change(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    primary_mask: np.ndarray,
    statistics: ChangeStatistics,
    deterministic_mask: Optional[np.ndarray] = None,
) -> BuiltUpChangeAssessment:
    """Compare structural support inside up to three largest real components."""
    before = np.asarray(Image.fromarray(before_rgb).resize((primary_mask.shape[1], primary_mask.shape[0]), Image.Resampling.BILINEAR))
    after = np.asarray(Image.fromarray(after_rgb).resize((primary_mask.shape[1], primary_mask.shape[0]), Image.Resampling.BILINEAR))
    mask = np.asarray(primary_mask, dtype=bool)
    deterministic = np.asarray(deterministic_mask, dtype=bool) if deterministic_mask is not None and deterministic_mask.shape == mask.shape else None
    region_evidence: list[BuiltUpRegionEvidence] = []
    for region in sorted(statistics.regions, key=lambda item: item.area_pixels, reverse=True):
        if region.percentage_of_image < MEANINGFUL_REGION_PERCENT:
            continue
        box = region.bounding_box
        selection = np.zeros(mask.shape, dtype=bool)
        selection[box.top:box.bottom, box.left:box.right] = mask[box.top:box.bottom, box.left:box.right]
        if not selection.any():
            continue
        before_structural, before_built = _scores(before, selection)
        after_structural, after_built = _scores(after, selection)
        direction, support = _direction(before_built, after_built)
        bbox_area = max(1, (box.right - box.left) * (box.bottom - box.top))
        overlap = round(float(np.mean(deterministic[selection])) * 100.0, 6) if deterministic is not None else None
        strength = "strong" if region.percentage_of_image >= 1.0 or (overlap or 0) >= 70 else "moderate" if region.percentage_of_image >= 0.2 or (overlap or 0) >= 40 else "small"
        region_evidence.append(BuiltUpRegionEvidence(
            region_id=region.region_id,
            relative_location=_location(box.left, box.top, box.right, box.bottom, statistics.analysis_width, statistics.analysis_height),
            bbox=[box.left, box.top, box.right, box.bottom],
            pixel_area=region.area_pixels,
            relative_area_percent=region.percentage_of_image,
            change_strength=strength,
            compactness=round(min(1.0, region.area_pixels / bbox_area), 6),
            before_structural_evidence=before_structural,
            after_structural_evidence=after_structural,
            before_built_up_evidence=before_built,
            after_built_up_evidence=after_built,
            directional_state=direction,
            semantic_support_level=support,
            deterministic_overlap_percent=overlap,
        ))
        if len(region_evidence) == 3:
            break

    states = {item.directional_state for item in region_evidence}
    if "INCREASE_SUPPORTED" in states and "DECREASE_SUPPORTED" in states:
        state = "MIXED_CHANGE"
    elif "INCREASE_SUPPORTED" in states:
        state = "INCREASE_SUPPORTED"
    elif "DECREASE_SUPPORTED" in states:
        state = "DECREASE_SUPPORTED"
    elif not region_evidence or statistics.percentage_changed < 0.5:
        state = "NO_MEANINGFUL_EVIDENCE"
    else:
        state = "INSUFFICIENT_EVIDENCE"

    magnitude = "minimal" if statistics.percentage_changed < 0.5 else "localized" if statistics.percentage_changed < 2 else "moderate" if statistics.percentage_changed < 10 else "substantial" if statistics.percentage_changed < 30 else "widespread"
    factors = ["ChangerEx connected-region geometry is the primary evidence."]
    overlaps = [item.deterministic_overlap_percent for item in region_evidence if item.deterministic_overlap_percent is not None]
    if overlaps:
        factors.append("Deterministic change evidence agrees strongly in the analyzed regions." if sum(overlaps) / len(overlaps) >= 60 else "Learned and deterministic regional evidence has limited agreement.")
    supported = [item for item in region_evidence if item.semantic_support_level == "supported"]
    if supported:
        factors.append("Before/after structural evidence supplies a consistent direction in at least one region.")
    coherent = bool(region_evidence and sum(item.compactness for item in region_evidence) / len(region_evidence) >= 0.35)
    if coherent:
        factors.append("The leading connected regions are spatially coherent.")
    confidence = "high" if supported and coherent and overlaps and sum(overlaps) / len(overlaps) >= 60 else "moderate" if supported else "low"
    return BuiltUpChangeAssessment(
        state=state,
        magnitude=magnitude,
        confidence=confidence,
        confidence_factors=factors,
        regions=region_evidence,
        limitations=[
            "Built-up-like and structural scores are heuristic support measures, not calibrated probabilities, ground-truth, or land-cover labels.",
            "The binary change mask supplies geometry only and cannot establish urban growth by itself.",
        ],
    )


def save_top_region_overlay(after_rgb: np.ndarray, assessment: BuiltUpChangeAssessment) -> Optional[str]:
    if not assessment.regions:
        return None
    image = Image.fromarray(after_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ((255, 190, 48), (60, 210, 255), (245, 95, 190))
    for index, region in enumerate(assessment.regions):
        color = colors[index % len(colors)]
        left, top, right, bottom = region.bbox
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, min(image.size) // 180))
        draw.rectangle((left, top, min(right, left + 24), min(bottom, top + 16)), fill=(8, 14, 24))
        draw.text((left + 4, top + 2), str(index + 1), fill=color, font=font)
    try:
        return save_preview(image)
    finally:
        image.close()
