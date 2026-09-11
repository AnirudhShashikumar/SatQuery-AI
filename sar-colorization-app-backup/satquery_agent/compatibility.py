"""Truthful pair-compatibility reporting without alignment or reprojection."""

from __future__ import annotations

import math
from datetime import date
from typing import List, Optional, Sequence, Tuple

from .models import AlignmentLevel, ImageMetadata, InputMode, Modality, PairCompatibility, RasterBounds


def _nearly_equal(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-7) -> bool:
    return len(left) == len(right) and all(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(left, right))


def _bounds_equal(left: RasterBounds, right: RasterBounds) -> bool:
    return _nearly_equal(
        [left.left, left.bottom, left.right, left.top],
        [right.left, right.bottom, right.right, right.top],
    )


def _overlap(left: RasterBounds, right: RasterBounds) -> Tuple[bool, float]:
    intersection_width = max(0.0, min(left.right, right.right) - max(left.left, right.left))
    intersection_height = max(0.0, min(left.top, right.top) - max(left.bottom, right.bottom))
    intersection_area = intersection_width * intersection_height
    left_area = max(0.0, left.right - left.left) * max(0.0, left.top - left.bottom)
    right_area = max(0.0, right.right - right.left) * max(0.0, right.top - right.bottom)
    denominator = min(left_area, right_area)
    ratio = intersection_area / denominator if denominator > 0 else 0.0
    return intersection_area > 0, round(ratio, 6)


def _date_errors(primary_date: Optional[str], secondary_date: Optional[str]) -> List[str]:
    if not primary_date or not secondary_date:
        return ["Two distinct temporal date labels are required."]
    try:
        earlier = date.fromisoformat(primary_date)
        later = date.fromisoformat(secondary_date)
    except ValueError:
        return ["Temporal dates must use YYYY-MM-DD format."]
    if earlier >= later:
        return ["The earlier-date image must have a date before the later-date image."]
    return []


def validate_pair_compatibility(
    input_mode: InputMode,
    primary_modality: Modality,
    secondary_modality: Optional[Modality],
    primary: ImageMetadata,
    secondary: ImageMetadata,
    primary_date: Optional[str] = None,
    secondary_date: Optional[str] = None,
) -> PairCompatibility:
    errors: List[str] = []
    warnings: List[str] = []
    same_dimensions = primary.width == secondary.width and primary.height == secondary.height

    if input_mode == InputMode.CROSS_MODAL:
        modalities = {primary_modality, secondary_modality}
        optical_family = bool(modalities.intersection({Modality.OPTICAL, Modality.MULTISPECTRAL}))
        if not (optical_family and Modality.SAR in modalities):
            errors.append("Cross-modal analysis requires one optical or multispectral image and one SAR image.")
    elif input_mode == InputMode.BI_TEMPORAL:
        optical_family = {Modality.OPTICAL, Modality.MULTISPECTRAL}
        modalities_compatible = (
            secondary_modality is not None
            and (primary_modality == secondary_modality or {primary_modality, secondary_modality}.issubset(optical_family))
        )
        if not modalities_compatible:
            errors.append("Bi-temporal comparison requires matching modalities or an optical/multispectral pair.")
        errors.extend(_date_errors(primary_date, secondary_date))

    same_crs = None if not primary.crs or not secondary.crs else primary.crs == secondary.crs
    same_transform = None if primary.transform is None or secondary.transform is None else _nearly_equal(primary.transform, secondary.transform)
    bounds_overlap: Optional[bool] = None
    overlap_ratio: Optional[float] = None
    if primary.bounds is not None and secondary.bounds is not None and same_crs is True:
        bounds_overlap, overlap_ratio = _overlap(primary.bounds, secondary.bounds)

    if errors:
        return PairCompatibility(
            compatible=False,
            alignment_level=AlignmentLevel.INCOMPATIBLE,
            same_dimensions=same_dimensions,
            same_crs=same_crs,
            same_transform=same_transform,
            bounds_overlap=bounds_overlap,
            overlap_ratio=overlap_ratio,
            resampling_required=not same_dimensions,
            warnings=warnings,
            errors=list(dict.fromkeys(errors)),
        )

    if primary.crs and secondary.crs and same_crs is False:
        errors.append("The raster CRS values differ; overlap cannot be established without reprojection.")
    elif same_crs is True and primary.bounds is not None and secondary.bounds is not None:
        if not bounds_overlap:
            errors.append("The raster bounds do not overlap in their shared CRS.")
        else:
            bounds_match = _bounds_equal(primary.bounds, secondary.bounds)
            exact = same_dimensions and same_transform is True and bounds_match
            if exact:
                return PairCompatibility(
                    compatible=True,
                    alignment_level=AlignmentLevel.EXACT,
                    same_dimensions=True,
                    same_crs=True,
                    same_transform=True,
                    bounds_overlap=True,
                    overlap_ratio=overlap_ratio,
                    resampling_required=False,
                    warnings=[],
                    errors=[],
                )
            warnings.append("The images overlap geographically but are not pixel-aligned.")
            return PairCompatibility(
                compatible=True,
                alignment_level=AlignmentLevel.GEOSPATIAL_OVERLAP,
                same_dimensions=same_dimensions,
                same_crs=True,
                same_transform=same_transform,
                bounds_overlap=True,
                overlap_ratio=overlap_ratio,
                resampling_required=True,
                warnings=warnings,
                errors=[],
            )

    if errors:
        return PairCompatibility(
            compatible=False,
            alignment_level=AlignmentLevel.INCOMPATIBLE,
            same_dimensions=same_dimensions,
            same_crs=same_crs,
            same_transform=same_transform,
            bounds_overlap=bounds_overlap,
            overlap_ratio=overlap_ratio,
            resampling_required=True,
            warnings=warnings,
            errors=errors,
        )

    aspect_primary = primary.width / primary.height
    aspect_secondary = secondary.width / secondary.height
    aspect_difference = abs(aspect_primary - aspect_secondary) / max(aspect_primary, aspect_secondary)
    if aspect_difference > 0.05:
        errors.append("Without georeferencing, the image aspect ratios differ too much for a reliable visual comparison.")
        return PairCompatibility(
            compatible=False,
            alignment_level=AlignmentLevel.INCOMPATIBLE,
            same_dimensions=same_dimensions,
            same_crs=same_crs,
            same_transform=same_transform,
            bounds_overlap=None,
            overlap_ratio=None,
            resampling_required=True,
            warnings=warnings,
            errors=errors,
        )

    warnings.append("Georeferencing is unavailable; this pair cannot be treated as co-registered.")
    if not same_dimensions:
        warnings.append("Image dimensions differ; later visual comparison would require explicit resampling.")
    return PairCompatibility(
        compatible=True,
        alignment_level=AlignmentLevel.VISUAL_ONLY,
        same_dimensions=same_dimensions,
        same_crs=same_crs,
        same_transform=same_transform,
        bounds_overlap=None,
        overlap_ratio=None,
        resampling_required=not same_dimensions,
        warnings=warnings,
        errors=[],
    )
