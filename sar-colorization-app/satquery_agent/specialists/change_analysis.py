"""Deterministic, non-generative bi-temporal change analysis.

The engine deliberately does not register, reproject, resample, caption, or
interpret imagery. It compares only pixel-compatible arrays and reports the
visual evidence it computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

from ..image_ingestion import remove_preview_url, save_preview
from ..models import (
    AlignmentLevel,
    ChangePreviewUrls,
    ChangeRegion,
    ChangeStatistics,
    ImageMetadata,
    MaskComparison,
    PairCompatibility,
    PixelBoundingBox,
)


MIN_NORMALIZED_THRESHOLD = 0.05
MIN_REGION_PIXELS = 4
MIN_REGION_FRACTION = 0.00005


class ChangeAnalysisError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ChangeAnalysisResult:
    statistics: ChangeStatistics
    previews: ChangePreviewUrls
    durations_ms: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    before_rgb: Optional[np.ndarray] = field(default=None, repr=False)
    after_rgb: Optional[np.ndarray] = field(default=None, repr=False)


def requires_alignment(compatibility: PairCompatibility) -> bool:
    """Return true when deterministic pixel comparison must not proceed."""
    if compatibility.alignment_level == AlignmentLevel.EXACT:
        return False
    return not (
        compatibility.alignment_level == AlignmentLevel.VISUAL_ONLY
        and compatibility.same_dimensions
        and not compatibility.resampling_required
    )


def _duration(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _canonical_pair(before: np.ndarray, after: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if before.ndim != 3 or after.ndim != 3 or before.shape[2] < 1 or after.shape[2] < 1:
        raise ChangeAnalysisError("INVALID_RASTER_SHAPE", "Both inputs must contain at least one raster band.")
    if before.shape[:2] != after.shape[:2]:
        raise ChangeAnalysisError("ALIGNMENT_REQUIRED", "Image dimensions differ; explicit alignment is required.")
    if before.shape[2] != after.shape[2]:
        before = np.mean(before.astype(np.float64), axis=2, keepdims=True)
        after = np.mean(after.astype(np.float64), axis=2, keepdims=True)
        warnings.append("Band counts differ; the comparison used a single mean-intensity channel for each image.")
    else:
        channels = min(before.shape[2], 3)
        before = before[:, :, :channels].astype(np.float64, copy=False)
        after = after[:, :, :channels].astype(np.float64, copy=False)
    return before, after


def _joint_normalize(
    before: np.ndarray,
    after: np.ndarray,
    before_nodata: Optional[float],
    after_nodata: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_before = np.all(np.isfinite(before), axis=2)
    valid_after = np.all(np.isfinite(after), axis=2)
    if before_nodata is not None:
        valid_before &= ~np.any(np.isclose(before, before_nodata), axis=2)
    if after_nodata is not None:
        valid_after &= ~np.any(np.isclose(after, after_nodata), axis=2)
    comparable = valid_before & valid_after
    if not np.any(comparable):
        raise ChangeAnalysisError("NO_COMPARABLE_PIXELS", "The pair has no finite, non-nodata pixels in common.")

    normalized_before = np.zeros(before.shape, dtype=np.float32)
    normalized_after = np.zeros(after.shape, dtype=np.float32)
    for channel in range(before.shape[2]):
        left = before[:, :, channel][comparable]
        right = after[:, :, channel][comparable]
        joint = np.concatenate((left, right))
        low, high = np.percentile(joint, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ChangeAnalysisError("NON_FINITE_RASTER", "The comparable raster values have invalid numeric ranges.")
        if high <= low:
            low, high = float(np.min(joint)), float(np.max(joint))
        if high <= low:
            continue
        normalized_before[:, :, channel] = np.clip((before[:, :, channel] - low) / (high - low), 0.0, 1.0)
        normalized_after[:, :, channel] = np.clip((after[:, :, channel] - low) / (high - low), 0.0, 1.0)
    normalized_before[~comparable] = 0.0
    normalized_after[~comparable] = 0.0
    return normalized_before, normalized_after, comparable


def _threshold_difference(difference: np.ndarray, comparable: np.ndarray) -> Tuple[np.ndarray, float]:
    values = difference[comparable]
    if values.size == 0 or float(np.max(values)) < MIN_NORMALIZED_THRESHOLD:
        return np.zeros(difference.shape, dtype=bool), MIN_NORMALIZED_THRESHOLD
    threshold = max(MIN_NORMALIZED_THRESHOLD, float(threshold_otsu(values)))
    return (difference >= threshold) & comparable, min(threshold, 1.0)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    footprint = disk(1)
    cleaned = binary_closing(mask, footprint)
    cleaned = binary_opening(cleaned, footprint)
    minimum_area = max(MIN_REGION_PIXELS, int(np.ceil(mask.size * MIN_REGION_FRACTION)))
    return remove_small_objects(cleaned, min_size=minimum_area, connectivity=2)


def _statistics(mask: np.ndarray, threshold: float, source_width: int, source_height: int) -> ChangeStatistics:
    components = label(mask, connectivity=2)
    extracted = sorted(regionprops(components), key=lambda region: int(region.area), reverse=True)
    total_pixels = int(mask.size)
    changed_pixels = int(np.count_nonzero(mask))
    regions: List[ChangeRegion] = []
    boxes: List[PixelBoundingBox] = []
    for index, region in enumerate(extracted, start=1):
        top, left, bottom, right = (int(value) for value in region.bbox)
        area = int(region.area)
        box = PixelBoundingBox(left=left, top=top, right=right, bottom=bottom, area_pixels=area)
        boxes.append(box)
        regions.append(
            ChangeRegion(
                region_id=index,
                area_pixels=area,
                percentage_of_image=round(area * 100.0 / total_pixels, 6),
                bounding_box=box,
            )
        )
    return ChangeStatistics(
        analysis_width=int(mask.shape[1]),
        analysis_height=int(mask.shape[0]),
        source_width=source_width,
        source_height=source_height,
        total_pixels=total_pixels,
        changed_pixels=changed_pixels,
        percentage_changed=round(changed_pixels * 100.0 / total_pixels, 6),
        largest_connected_region=regions[0].area_pixels if regions else 0,
        number_of_regions=len(regions),
        bounding_boxes=boxes,
        regions=regions,
        normalized_threshold=round(threshold, 6),
    )


def statistics_from_binary_mask(mask: np.ndarray, source_width: int, source_height: int) -> ChangeStatistics:
    """Compute the shared region contract without altering a learned mask."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ChangeAnalysisError("INVALID_MASK_DIMENSIONS", "The learned mask does not contain a valid analysis grid.")
    if values.shape[0] > source_height or values.shape[1] > source_width:
        raise ChangeAnalysisError("INVALID_MASK_DIMENSIONS", "The learned mask exceeds the source image dimensions.")
    if not np.all(np.isfinite(values)) or not set(np.unique(values).tolist()).issubset({0, 1, False, True}):
        raise ChangeAnalysisError("INVALID_BINARY_MASK", "The learned mask is not a finite binary array.")
    return _statistics(values.astype(bool), 0.0, source_width, source_height)


def compare_binary_masks(primary: np.ndarray, supporting: np.ndarray) -> MaskComparison:
    if primary.shape != supporting.shape:
        raise ChangeAnalysisError("MASK_DIMENSION_MISMATCH", "Learned and deterministic masks use different grids.")
    learned = primary.astype(bool, copy=False)
    deterministic = supporting.astype(bool, copy=False)
    intersection = learned & deterministic
    union = learned | deterministic
    agreement = learned == deterministic
    total = learned.size
    union_count = int(union.sum())
    learned_changed = int(learned.sum())
    learned_background = total - learned_changed
    return MaskComparison(
        intersection_pixels=int(intersection.sum()),
        union_pixels=union_count,
        iou=round(float(intersection.sum()) / union_count, 6) if union_count else 1.0,
        agreement_percentage=round(float(agreement.sum()) * 100.0 / total, 6),
        disagreement_percentage=round(float((~agreement).sum()) * 100.0 / total, 6),
        changed_class_agreement=round(float(intersection.sum()) * 100.0 / learned_changed, 6) if learned_changed else None,
        background_agreement=round(float((~learned & ~deterministic).sum()) * 100.0 / learned_background, 6) if learned_background else None,
    )


def _to_rgb(values: np.ndarray) -> np.ndarray:
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    elif values.shape[2] == 2:
        values = np.dstack((values, np.mean(values, axis=2)))
    return np.round(np.clip(values[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def _heatmap(difference: np.ndarray) -> np.ndarray:
    value = np.clip(difference, 0.0, 1.0)
    # Fixed blue -> cyan -> yellow -> red scale; no data-dependent exaggeration.
    red = np.clip(4.0 * value - 1.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - 4.0 * value, 0.0, 1.0)
    return np.round(np.dstack((red, green, blue)) * 255.0).astype(np.uint8)


def _save_previews(before: np.ndarray, after: np.ndarray, difference: np.ndarray, mask: np.ndarray) -> ChangePreviewUrls:
    saved: List[str] = []
    try:
        before_rgb = _to_rgb(before)
        after_rgb = _to_rgb(after)
        overlay = after_rgb.astype(np.float32)
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([255.0, 32.0, 32.0], dtype=np.float32)
        images = (
            Image.fromarray(before_rgb),
            Image.fromarray(after_rgb),
            Image.fromarray(_heatmap(difference)),
            Image.fromarray(mask.astype(np.uint8) * 255),
            Image.fromarray(np.round(overlay).astype(np.uint8)),
        )
        for image in images:
            saved.append(save_preview(image))
            image.close()
        return ChangePreviewUrls(before=saved[0], after=saved[1], difference=saved[2], mask=saved[3], overlay=saved[4])
    except Exception:
        for preview_url in saved:
            remove_preview_url(preview_url)
        raise


def save_hybrid_previews(
    learned_mask: np.ndarray,
    deterministic_mask: np.ndarray,
    after_rgb: np.ndarray,
    deterministic_previews: ChangePreviewUrls,
) -> ChangePreviewUrls:
    """Create source-labelled learned/comparison products; neither mask is fused."""
    learned = learned_mask.astype(bool, copy=False)
    deterministic = deterministic_mask.astype(bool, copy=False)
    if learned.shape != deterministic.shape or after_rgb.shape[:2] != learned.shape:
        raise ChangeAnalysisError("MASK_DIMENSION_MISMATCH", "Hybrid evidence products do not share one analysis grid.")
    intersection = learned & deterministic
    union = learned | deterministic
    disagreement = learned ^ deterministic
    agreement = learned == deterministic
    learned_display = np.zeros((*learned.shape, 3), dtype=np.uint8)
    learned_display[learned] = (255, 72, 72)
    agreement_display = np.zeros((*learned.shape, 3), dtype=np.uint8)
    agreement_display[agreement] = (42, 190, 120)
    agreement_display[~agreement] = (190, 62, 190)
    disagreement_display = np.zeros((*learned.shape, 3), dtype=np.uint8)
    disagreement_display[learned & ~deterministic] = (255, 150, 40)
    disagreement_display[~learned & deterministic] = (76, 160, 255)
    overlay = after_rgb.astype(np.float32)
    overlay[learned] = 0.45 * overlay[learned] + 0.55 * np.array([255.0, 32.0, 32.0])
    saved: List[str] = []
    try:
        images = [
            Image.fromarray(learned.astype(np.uint8) * 255),
            Image.fromarray(learned_display),
            Image.fromarray(np.round(overlay).astype(np.uint8)),
            Image.fromarray(agreement_display),
            Image.fromarray(disagreement_display),
            Image.fromarray(intersection.astype(np.uint8) * 255),
            Image.fromarray(union.astype(np.uint8) * 255),
        ]
        for image in images:
            saved.append(save_preview(image))
            image.close()
        return deterministic_previews.model_copy(update={
            "mask": saved[0],
            "overlay": saved[2],
            "ttp_raw_mask": saved[0],
            "ttp_mask": saved[1],
            "ttp_overlay": saved[2],
            "deterministic_mask": deterministic_previews.mask,
            "deterministic_overlay": deterministic_previews.overlay,
            "agreement": saved[3],
            "disagreement": saved[4],
            "intersection": saved[5],
            "union": saved[6],
        })
    except Exception:
        for preview_url in saved:
            remove_preview_url(preview_url)
        raise


def analyze_change(
    before_raster: np.ndarray,
    after_raster: np.ndarray,
    before_metadata: ImageMetadata,
    after_metadata: ImageMetadata,
    compatibility: PairCompatibility,
) -> ChangeAnalysisResult:
    """Compute a reusable visual change product for an already-aligned pair."""
    if requires_alignment(compatibility):
        raise ChangeAnalysisError("ALIGNMENT_REQUIRED", "The pair is not pixel-aligned; no change products were computed.")

    warnings: List[str] = []
    durations: Dict[str, int] = {}
    started = time.perf_counter()
    before, after = _canonical_pair(before_raster, after_raster, warnings)
    normalized_before, normalized_after, comparable = _joint_normalize(
        before, after, before_metadata.nodata, after_metadata.nodata
    )
    absolute_difference = np.abs(normalized_after - normalized_before)
    normalized_difference = np.mean(absolute_difference, axis=2)
    normalized_difference[~comparable] = 0.0
    durations["difference_computation"] = _duration(started)

    started = time.perf_counter()
    raw_mask, threshold = _threshold_difference(normalized_difference, comparable)
    durations["thresholding"] = _duration(started)

    started = time.perf_counter()
    change_mask = _clean_mask(raw_mask)
    change_mask &= comparable
    durations["morphology"] = _duration(started)

    started = time.perf_counter()
    statistics = _statistics(change_mask, threshold, before_metadata.width, before_metadata.height)
    durations["connected_components"] = _duration(started)

    started = time.perf_counter()
    previews = _save_previews(normalized_before, normalized_after, normalized_difference, change_mask)
    durations["preview_generation"] = _duration(started)

    excluded = int(comparable.size - np.count_nonzero(comparable))
    if excluded:
        warnings.append(f"{excluded} pixel locations were excluded because one or both inputs contained nodata or non-finite values.")
    if compatibility.alignment_level == AlignmentLevel.VISUAL_ONLY:
        warnings.append("No georeferencing was available; pixel alignment was assumed from equal dimensions and was not independently verified.")
    if (statistics.analysis_width, statistics.analysis_height) != (before_metadata.width, before_metadata.height):
        warnings.append(
            "The change mask and pixel statistics use a "
            f"{statistics.analysis_width} × {statistics.analysis_height} reduced analysis grid from the "
            f"{before_metadata.width} × {before_metadata.height} source grid."
        )
    warnings.append("Change products measure normalized visual differences only; no semantic interpretation was generated.")
    return ChangeAnalysisResult(
        statistics=statistics,
        previews=previews,
        durations_ms=durations,
        warnings=list(dict.fromkeys(warnings)),
        mask=change_mask,
        before_rgb=_to_rgb(normalized_before),
        after_rgb=_to_rgb(normalized_after),
    )
