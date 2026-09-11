"""Transparent deterministic optical-SAR evidence fusion.

This module intentionally implements a heuristic evidence baseline, not a
semantic classifier. It never invokes reconstruction models or hosted AI and
never performs registration, reprojection, or resampling.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

from ..image_ingestion import remove_preview_url, save_preview
from ..models import (
    AlignmentLevel,
    Confidence,
    ConfidenceLevel,
    CrossModalMethod,
    CrossModalPreparation,
    CrossModalPreviewUrls,
    CrossModalRegion,
    CrossModalRegionSupport,
    CrossModalResult,
    CrossModalStatistics,
    CrossModalStatus,
    CrossModalSummary,
    ImageMetadata,
    Modality,
    PairCompatibility,
)


METHOD_NAME = "Deterministic optical-SAR evidence fusion"
METHOD_VERSION = "1.0"
METHOD_ASSUMPTIONS = [
    "Full pixel fusion requires matching CRS, affine transform, bounds, and source dimensions.",
    "Optical inputs are user-declared RGB or RGB-like imagery; the first three prepared bands are treated as red, green, and blue when explicit names are unavailable.",
    "SAR values are treated as relative uncalibrated intensity evidence, not physical calibrated backscatter.",
    "Thresholds are relative to each uploaded scene and indicate candidate support rather than semantic probability.",
]
METHOD_LIMITATIONS = [
    "This deterministic baseline is not a trained or calibrated land-cover classifier.",
    "Dark smooth terrain, cloud shadow, terrain shadow, and radar shadow can resemble water-like evidence.",
    "Bright terrain, speckle, vegetation structure, and isolated strong scatterers can resemble built-up or structural evidence.",
    "Visible-spectrum green dominance is only vegetation support; it is not NDVI and does not use NIR unless explicitly identified and implemented.",
    "Analysis may use a reduced grid capped at 1024 pixels on the longest side, so small features can be omitted.",
]

MIN_REGION_PIXELS = 6
MIN_REGION_FRACTION = 0.0001
MAX_RETURNED_REGIONS = 200


class CrossModalAnalysisError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PreparedOptical:
    rgb: np.ndarray
    valid: np.ndarray
    preparation: CrossModalPreparation
    rgb_like: bool
    dynamic_range_ok: bool


@dataclass
class PreparedSar:
    channels: np.ndarray
    intensity: np.ndarray
    valid: np.ndarray
    preparation: CrossModalPreparation
    dynamic_range_ok: bool


def _duration(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _valid_pixels(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.all(np.isfinite(array), axis=2)
    if nodata is not None:
        valid &= ~np.any(np.isclose(array, nodata), axis=2)
    return valid


def _normalize_channels(array: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, List[float], List[Tuple[float, float]]]:
    normalized = np.zeros(array.shape, dtype=np.float32)
    spans: List[float] = []
    limits: List[Tuple[float, float]] = []
    for channel in range(array.shape[2]):
        values = array[:, :, channel][valid]
        if values.size == 0:
            spans.append(0.0)
            limits.append((0.0, 0.0))
            continue
        low, high = (float(value) for value in np.percentile(values, [2.0, 98.0]))
        if not math.isfinite(low) or not math.isfinite(high):
            raise CrossModalAnalysisError("NON_FINITE_RASTER", "A prepared raster channel has an invalid numeric range.")
        if high <= low:
            low, high = float(np.min(values)), float(np.max(values))
        span = high - low
        spans.append(span)
        limits.append((low, high))
        if span > np.finfo(np.float32).eps:
            normalized[:, :, channel] = np.clip((array[:, :, channel] - low) / span, 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized, spans, limits


def _resize_status(metadata: ImageMetadata, shape: Tuple[int, int]) -> str:
    if (metadata.height, metadata.width) == shape:
        return "No resize; native source grid retained."
    return f"Common reduced analysis grid {shape[1]} × {shape[0]} from source {metadata.width} × {metadata.height}; no registration performed."


def _prepare_optical(array: np.ndarray, metadata: ImageMetadata) -> PreparedOptical:
    if array.ndim != 3 or array.shape[2] < 1:
        raise CrossModalAnalysisError("INVALID_OPTICAL_RASTER", "The optical input has no prepared raster bands.")
    raw = array[:, :, : min(3, array.shape[2])].astype(np.float64, copy=False)
    valid = _valid_pixels(raw, metadata.nodata)
    if not np.any(valid):
        raise CrossModalAnalysisError("NO_VALID_OPTICAL_PIXELS", "The optical input contains no finite, non-nodata pixels.")
    rgb_like = raw.shape[2] >= 3
    if raw.shape[2] == 1:
        raw = np.repeat(raw, 3, axis=2)
    elif raw.shape[2] == 2:
        raw = np.dstack((raw, np.mean(raw, axis=2)))
    normalized, spans, limits = _normalize_channels(raw, valid)
    bands = (metadata.color_interpretation[:3] or [f"band_{index + 1}" for index in range(min(3, metadata.band_count))])
    if len(bands) < 3:
        bands = list(bands) + ["derived_mean"] * (3 - len(bands))
    limits_text = ", ".join(f"{low:.6g}–{high:.6g}" for low, high in limits)
    return PreparedOptical(
        rgb=normalized,
        valid=valid,
        rgb_like=rgb_like,
        dynamic_range_ok=any(span > np.finfo(np.float32).eps for span in spans),
        preparation=CrossModalPreparation(
            modality=Modality.OPTICAL,
            band_count=metadata.band_count,
            bands_used=list(bands[:3]),
            channel_interpretation=list(metadata.color_interpretation),
            stretch_method=f"Per-channel finite-pixel 2nd/98th percentile clip ({limits_text}).",
            normalization="Each selected visible channel scaled independently to [0, 1].",
            invalid_pixel_handling="Non-finite and declared NoData pixels excluded and rendered as zero.",
            resize_status=_resize_status(metadata, valid.shape),
            log_transform="None.",
        ),
    )


def _prepare_sar(array: np.ndarray, metadata: ImageMetadata) -> PreparedSar:
    if array.ndim != 3 or array.shape[2] < 1:
        raise CrossModalAnalysisError("INVALID_SAR_RASTER", "The SAR input has no prepared raster bands.")
    raw = array[:, :, : min(2, array.shape[2])].astype(np.float64, copy=False)
    valid = _valid_pixels(raw, metadata.nodata)
    if not np.any(valid):
        raise CrossModalAnalysisError("NO_VALID_SAR_PIXELS", "The SAR input contains no finite, non-nodata pixels.")
    normalized, spans, limits = _normalize_channels(raw, valid)
    intensity = np.mean(normalized, axis=2).astype(np.float32)
    intensity[~valid] = 0.0
    bands = metadata.color_interpretation[: raw.shape[2]] or [f"band_{index + 1}" for index in range(raw.shape[2])]
    limits_text = ", ".join(f"{low:.6g}–{high:.6g}" for low, high in limits)
    return PreparedSar(
        channels=normalized,
        intensity=intensity,
        valid=valid,
        dynamic_range_ok=any(span > np.finfo(np.float32).eps for span in spans),
        preparation=CrossModalPreparation(
            modality=Modality.SAR,
            band_count=metadata.band_count,
            bands_used=list(bands),
            channel_interpretation=list(metadata.color_interpretation),
            stretch_method=f"Per-channel finite-pixel 2nd/98th percentile clip ({limits_text}).",
            normalization="Each available SAR channel scaled independently to [0, 1], then averaged for neutral intensity evidence.",
            invalid_pixel_handling="Non-finite and declared NoData pixels excluded and rendered as zero.",
            resize_status=_resize_status(metadata, valid.shape),
            log_transform="None; source calibration and linear/dB encoding were not established.",
        ),
    )


def _local_standard_deviation(values: np.ndarray, size: int = 5) -> np.ndarray:
    mean = uniform_filter(values.astype(np.float32), size=size, mode="reflect")
    mean_square = uniform_filter(np.square(values, dtype=np.float32), size=size, mode="reflect")
    return np.sqrt(np.maximum(mean_square - np.square(mean), 0.0)).astype(np.float32)


def _quantile(values: np.ndarray, valid: np.ndarray, fraction: float, fallback: float) -> float:
    selected = values[valid]
    return float(np.quantile(selected, fraction)) if selected.size else fallback


def _clean(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    footprint = disk(1)
    cleaned = binary_closing(mask, footprint)
    cleaned = binary_opening(cleaned, footprint)
    minimum = max(MIN_REGION_PIXELS, int(math.ceil(mask.size * MIN_REGION_FRACTION)))
    return remove_small_objects(cleaned, min_size=minimum, connectivity=2)


def _optical_evidence(prepared: PreparedOptical) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rgb, valid = prepared.rgb, prepared.valid
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    brightness = np.mean(rgb, axis=2)
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    texture = _local_standard_deviation(brightness)
    edge = np.hypot(sobel(brightness, axis=0, mode="reflect"), sobel(brightness, axis=1, mode="reflect"))
    dark_threshold = min(0.38, max(0.18, _quantile(brightness, valid, 0.35, 0.3)))
    smooth_threshold = max(0.035, _quantile(texture, valid, 0.55, 0.06))
    vegetation = np.zeros(valid.shape, dtype=bool)
    water = np.zeros(valid.shape, dtype=bool)
    if prepared.rgb_like:
        vegetation = valid & (green >= red + 0.06) & (green >= blue + 0.04) & (saturation >= 0.10) & (brightness >= 0.12)
        water = valid & (brightness <= dark_threshold) & (texture <= smooth_threshold) & (blue >= red - 0.03) & (blue >= 0.72 * green) & ~vegetation
    edge_threshold = max(0.055, _quantile(edge, valid, 0.75, 0.08))
    texture_threshold = max(0.045, _quantile(texture, valid, 0.72, 0.06))
    structural = valid & (brightness >= 0.18) & ((edge >= edge_threshold) | (texture >= texture_threshold)) & ~water & ~vegetation
    return _clean(water), _clean(structural), _clean(vegetation), {
        "darkness_threshold": dark_threshold,
        "smoothness_threshold": smooth_threshold,
        "edge_threshold": edge_threshold,
        "texture_threshold": texture_threshold,
    }


def _sar_evidence(prepared: PreparedSar) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    intensity, valid = prepared.intensity, prepared.valid
    texture = _local_standard_deviation(intensity)
    median_intensity = _quantile(intensity, valid, 0.50, 0.50)
    low_threshold = min(_quantile(intensity, valid, 0.30, 0.25), median_intensity - 0.10)
    high_threshold = max(_quantile(intensity, valid, 0.70, 0.75), median_intensity + 0.10)
    smooth_threshold = max(0.025, _quantile(texture, valid, 0.45, 0.05))
    heterogeneous_threshold = max(0.045, _quantile(texture, valid, 0.72, 0.08))
    if not prepared.dynamic_range_ok:
        empty = np.zeros(valid.shape, dtype=bool)
        return empty, empty.copy(), {
            "low_backscatter_threshold": low_threshold,
            "high_backscatter_threshold": high_threshold,
            "smoothness_threshold": smooth_threshold,
            "heterogeneity_threshold": heterogeneous_threshold,
        }
    low = valid & (intensity <= low_threshold) & (texture <= smooth_threshold)
    structural = valid & ((intensity >= high_threshold) | (texture >= heterogeneous_threshold))
    if prepared.channels.shape[2] == 2:
        contrast = np.abs(prepared.channels[:, :, 0] - prepared.channels[:, :, 1])
        contrast_threshold = max(0.08, _quantile(contrast, valid, 0.78, 0.12))
        structural |= valid & (contrast >= contrast_threshold) & (texture >= smooth_threshold)
    return _clean(low), _clean(structural), {
        "low_backscatter_threshold": low_threshold,
        "high_backscatter_threshold": high_threshold,
        "smoothness_threshold": smooth_threshold,
        "heterogeneity_threshold": heterogeneous_threshold,
    }


def _percent(mask: np.ndarray, denominator: int) -> float:
    return round(float(np.count_nonzero(mask)) * 100.0 / denominator, 6) if denominator else 0.0


def _pixel_to_world(transform: List[float], x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def _world_geometry(
    metadata: ImageMetadata,
    bbox: Tuple[int, int, int, int],
    centroid: Tuple[float, float],
    analysis_shape: Tuple[int, int],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if not metadata.crs or not metadata.transform or len(metadata.transform) != 6:
        return None, None
    left, top, right, bottom = bbox
    scale_x = metadata.width / analysis_shape[1]
    scale_y = metadata.height / analysis_shape[0]
    corners = [
        _pixel_to_world(metadata.transform, left * scale_x, top * scale_y),
        _pixel_to_world(metadata.transform, right * scale_x, top * scale_y),
        _pixel_to_world(metadata.transform, left * scale_x, bottom * scale_y),
        _pixel_to_world(metadata.transform, right * scale_x, bottom * scale_y),
    ]
    xs, ys = zip(*corners)
    cx, cy = _pixel_to_world(metadata.transform, centroid[0] * scale_x, centroid[1] * scale_y)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], [float(cx), float(cy)]


def _regions_for_mask(
    mask: np.ndarray,
    region_type: str,
    valid_count: int,
    metadata: ImageMetadata,
    optical_support: np.ndarray,
    sar_support: np.ndarray,
) -> List[CrossModalRegion]:
    components = label(mask, connectivity=2)
    properties = sorted(regionprops(components), key=lambda item: int(item.area), reverse=True)
    output: List[CrossModalRegion] = []
    for index, region in enumerate(properties[:MAX_RETURNED_REGIONS], start=1):
        top, left, bottom, right = (int(value) for value in region.bbox)
        row, column = (float(value) for value in region.centroid)
        region_mask = components == region.label
        bbox = (left, top, right, bottom)
        centroid = (column, row)
        bbox_world, centroid_world = _world_geometry(metadata, bbox, centroid, mask.shape)
        output.append(
            CrossModalRegion(
                region_id=f"{region_type}-{index}",
                type=region_type,
                area_pixels=int(region.area),
                area_percent=_percent(region_mask, valid_count),
                bbox_pixels=list(bbox),
                centroid_pixels=[round(column, 3), round(row, 3)],
                bbox_world=bbox_world,
                centroid_world=centroid_world,
                support=CrossModalRegionSupport(
                    optical=bool(np.any(optical_support & region_mask)),
                    sar=bool(np.any(sar_support & region_mask)),
                ),
            )
        )
    return output


def _mask_image(mask: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = color
    return Image.fromarray(image)


def _evidence_composite(shape: Tuple[int, int], water: np.ndarray, structural: np.ndarray, vegetation: Optional[np.ndarray] = None) -> Image.Image:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[water] = (32, 145, 255)
    image[structural] = (255, 150, 35)
    if vegetation is not None:
        image[vegetation] = (45, 200, 105)
    return Image.fromarray(image)


def _save_products(
    optical: PreparedOptical,
    sar: PreparedSar,
    optical_water: np.ndarray,
    optical_structural: np.ndarray,
    vegetation: np.ndarray,
    sar_water: np.ndarray,
    sar_structural: np.ndarray,
    joint_water: np.ndarray,
    joint_structural: np.ndarray,
    agreement: np.ndarray,
    disagreement: np.ndarray,
) -> CrossModalPreviewUrls:
    saved: List[str] = []
    optical_rgb = np.round(np.clip(optical.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    sar_gray = np.round(np.clip(sar.intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
    overlay = optical_rgb.astype(np.float32)
    overlay[vegetation] = 0.65 * overlay[vegetation] + 0.35 * np.array([45.0, 200.0, 105.0])
    overlay[joint_water] = 0.35 * overlay[joint_water] + 0.65 * np.array([32.0, 145.0, 255.0])
    overlay[joint_structural] = 0.35 * overlay[joint_structural] + 0.65 * np.array([255.0, 150.0, 35.0])
    overlay[disagreement] = 0.55 * overlay[disagreement] + 0.45 * np.array([225.0, 65.0, 235.0])
    images = [
        Image.fromarray(optical_rgb),
        Image.fromarray(sar_gray),
        _evidence_composite(optical_water.shape, optical_water, optical_structural, vegetation),
        _evidence_composite(sar_water.shape, sar_water, sar_structural),
        _evidence_composite(joint_water.shape, joint_water, joint_structural),
        _mask_image(joint_water, (32, 145, 255)),
        _mask_image(joint_structural, (255, 150, 35)),
        _mask_image(vegetation, (45, 200, 105)),
        _mask_image(agreement, (35, 210, 210)),
        _mask_image(disagreement, (225, 65, 235)),
        Image.fromarray(np.round(np.clip(overlay, 0.0, 255.0)).astype(np.uint8)),
    ]
    try:
        for image in images:
            saved.append(save_preview(image))
            image.close()
        return CrossModalPreviewUrls(
            optical=saved[0],
            sar=saved[1],
            optical_evidence=saved[2],
            sar_evidence=saved[3],
            joint_evidence=saved[4],
            water_likelihood=saved[5],
            built_up_likelihood=saved[6],
            vegetation_support=saved[7],
            agreement=saved[8],
            disagreement=saved[9],
            joint_overlay=saved[10],
        )
    except Exception:
        for preview_url in saved:
            remove_preview_url(preview_url)
        raise


def _method() -> CrossModalMethod:
    return CrossModalMethod(
        name=METHOD_NAME,
        version=METHOD_VERSION,
        uses_trained_model=False,
        assumptions=METHOD_ASSUMPTIONS,
        limitations=METHOD_LIMITATIONS,
    )


def _guarded_result(
    compatibility: PairCompatibility,
    optical_metadata: ImageMetadata,
    sar_metadata: ImageMetadata,
    runtime_ms: int,
) -> CrossModalResult:
    previews = CrossModalPreviewUrls(optical=optical_metadata.preview_url, sar=sar_metadata.preview_url)
    if compatibility.alignment_level == AlignmentLevel.VISUAL_ONLY:
        return CrossModalResult(
            status=CrossModalStatus.PARTIAL,
            summary=CrossModalSummary(
                optical_observations=[f"Optical input has {optical_metadata.band_count} band(s) at {optical_metadata.width} × {optical_metadata.height} pixels."],
                sar_observations=[f"SAR input has {sar_metadata.band_count} band(s) at {sar_metadata.width} × {sar_metadata.height} pixels."],
                joint_observations=["Only side-by-side visual inspection is available because geospatial co-registration cannot be established."],
            ),
            statistics=None,
            regions=[],
            previews=previews,
            confidence=Confidence(level=ConfidenceLevel.UNAVAILABLE, score=None, reason="Pixel-level confidence is unavailable because the pair cannot be treated as co-registered."),
            method=_method(),
            warnings=_unique(compatibility.warnings + ["No pixel-level fusion, evidence masks, or joint statistics were computed."]),
            runtime_ms=runtime_ms,
        )
    if compatibility.alignment_level == AlignmentLevel.GEOSPATIAL_OVERLAP:
        return CrossModalResult(
            status=CrossModalStatus.ALIGNMENT_REQUIRED,
            summary=CrossModalSummary(joint_observations=["The pair requires explicit alignment or reprojection before pixel-level evidence fusion."]),
            statistics=None,
            regions=[],
            previews=previews,
            confidence=Confidence(level=ConfidenceLevel.UNAVAILABLE, score=None, reason="No fusion confidence is available until an explicit alignment workflow is applied."),
            method=_method(),
            warnings=_unique(compatibility.warnings + compatibility.errors + ["The service did not resize, register, reproject, or resample either image."]),
            runtime_ms=runtime_ms,
        )
    return CrossModalResult(
        status=CrossModalStatus.FAILED,
        summary=CrossModalSummary(joint_observations=["The optical-SAR pair is incompatible and was not analyzed."]),
        statistics=None,
        regions=[],
        previews=previews,
        confidence=Confidence(level=ConfidenceLevel.UNAVAILABLE, score=None, reason="Input compatibility validation failed."),
        method=_method(),
        warnings=_unique(compatibility.errors + compatibility.warnings),
        runtime_ms=runtime_ms,
    )


class CrossModalOpticalSarAnalyzer:
    """Local deterministic specialist for exactly aligned optical-SAR pairs."""

    def analyze(
        self,
        optical_image: np.ndarray,
        sar_image: np.ndarray,
        optical_metadata: ImageMetadata,
        sar_metadata: ImageMetadata,
        compatibility: PairCompatibility,
    ) -> CrossModalResult:
        started_total = time.perf_counter()
        if compatibility.alignment_level != AlignmentLevel.EXACT:
            return _guarded_result(compatibility, optical_metadata, sar_metadata, _duration(started_total))
        if optical_image.shape[:2] != sar_image.shape[:2]:
            raise CrossModalAnalysisError("ALIGNMENT_REQUIRED", "Prepared analysis grids differ; no hidden resizing was applied.")

        warnings: List[str] = []
        durations: Dict[str, int] = {}
        started = time.perf_counter()
        optical = _prepare_optical(optical_image, optical_metadata)
        durations["optical_preparation"] = _duration(started)
        started = time.perf_counter()
        sar = _prepare_sar(sar_image, sar_metadata)
        durations["sar_preparation"] = _duration(started)

        valid = optical.valid & sar.valid
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            raise CrossModalAnalysisError("NO_COMMON_VALID_PIXELS", "The exactly aligned pair has no finite, non-nodata pixels in common.")

        started = time.perf_counter()
        optical_water, optical_structural, vegetation, _ = _optical_evidence(optical)
        optical_water &= valid
        optical_structural &= valid
        vegetation &= valid
        durations["optical_evidence_extraction"] = _duration(started)

        started = time.perf_counter()
        sar_water, sar_structural, _ = _sar_evidence(sar)
        sar_water &= valid
        sar_structural &= valid
        durations["sar_evidence_extraction"] = _duration(started)

        started = time.perf_counter()
        joint_water = _clean(optical_water & sar_water) & valid
        joint_structural = _clean(optical_structural & sar_structural) & valid
        agreement = joint_water | joint_structural
        candidate_union = optical_water | sar_water | optical_structural | sar_structural
        disagreement = _clean(candidate_union & ~agreement) & valid
        evaluated_count = int(np.count_nonzero(candidate_union))
        durations["joint_evidence_fusion"] = _duration(started)

        statistics = CrossModalStatistics(
            analysis_width=int(valid.shape[1]),
            analysis_height=int(valid.shape[0]),
            source_width=optical_metadata.width,
            source_height=optical_metadata.height,
            water_likelihood_percent=_percent(joint_water, valid_count),
            built_up_likelihood_percent=_percent(joint_structural, valid_count),
            vegetation_support_percent=_percent(vegetation, valid_count) if optical.rgb_like else None,
            agreement_percent=_percent(agreement, evaluated_count) if evaluated_count else None,
            disagreement_percent=_percent(disagreement, evaluated_count) if evaluated_count else None,
            valid_pixel_percent=_percent(valid, valid.size),
            evaluated_candidate_pixels=evaluated_count,
        )

        started = time.perf_counter()
        regions = []
        regions.extend(_regions_for_mask(joint_water, "water_likelihood", valid_count, optical_metadata, optical_water, sar_water))
        regions.extend(_regions_for_mask(joint_structural, "built_up_likelihood", valid_count, optical_metadata, optical_structural, sar_structural))
        regions.extend(_regions_for_mask(disagreement, "disagreement", valid_count, optical_metadata, optical_water | optical_structural, sar_water | sar_structural))
        durations["region_extraction"] = _duration(started)

        started = time.perf_counter()
        previews = _save_products(
            optical,
            sar,
            optical_water,
            optical_structural,
            vegetation,
            sar_water,
            sar_structural,
            joint_water,
            joint_structural,
            agreement,
            disagreement,
        )
        durations["preview_generation"] = _duration(started)

        if not optical.rgb_like:
            warnings.append("The optical input was not RGB-like; visible-spectrum water and vegetation evidence were omitted.")
        if not optical.dynamic_range_ok:
            warnings.append("The optical input has negligible dynamic range; optical evidence is low-information.")
        if not sar.dynamic_range_ok:
            warnings.append("The SAR input has negligible dynamic range; SAR evidence masks are empty.")
        if (valid.shape[1], valid.shape[0]) != (optical_metadata.width, optical_metadata.height):
            warnings.append(
                f"Evidence masks and statistics use a {valid.shape[1]} × {valid.shape[0]} reduced analysis grid from the "
                f"{optical_metadata.width} × {optical_metadata.height} exact source grid."
            )
        warnings.extend([
            "Low-backscatter candidates can include smooth surfaces, radar shadow, and terrain effects.",
            "Structural-likelihood candidates can include bright terrain, vegetation structure, speckle, or isolated strong scatterers.",
            "Percentages describe deterministic candidate evidence, not calibrated semantic probabilities.",
        ])
        if len(regions) >= MAX_RETURNED_REGIONS:
            warnings.append(f"Region output is limited to the {MAX_RETURNED_REGIONS} largest connected components per evidence type.")

        low_information = not optical.dynamic_range_ok or not sar.dynamic_range_ok or not optical.rgb_like
        confidence = Confidence(
            level=ConfidenceLevel.LOW if low_information or statistics.valid_pixel_percent < 70.0 else ConfidenceLevel.MODERATE,
            score=None,
            reason=(
                "The pair is exactly aligned, but one or both inputs have limited usable spectral or intensity information; the output remains heuristic."
                if low_information
                else "The pair is exactly aligned and both modalities contribute valid candidate evidence, but this deterministic baseline is not a calibrated semantic classifier."
            ),
        )
        status = CrossModalStatus.PARTIAL if low_information else CrossModalStatus.SUCCESS
        water_text = f"Joint optical and SAR water-like support covers {statistics.water_likelihood_percent:.3f}% of valid pixels."
        structural_text = f"Joint optical and SAR structural-likelihood support covers {statistics.built_up_likelihood_percent:.3f}% of valid pixels."
        agreement_text = (
            f"The modalities positively agree over {statistics.agreement_percent:.3f}% of evaluated candidate-evidence pixels; disagreement covers {statistics.disagreement_percent:.3f}%."
            if statistics.agreement_percent is not None and statistics.disagreement_percent is not None
            else "No candidate-evidence pixels survived the deterministic extraction rules, so agreement percentages are unavailable."
        )
        return CrossModalResult(
            status=status,
            summary=CrossModalSummary(
                optical_observations=[
                    f"Optical evidence used {', '.join(optical.preparation.bands_used)} with a finite-pixel percentile stretch.",
                    f"Visible-spectrum vegetation support covers {statistics.vegetation_support_percent:.3f}% of valid pixels."
                    if statistics.vegetation_support_percent is not None
                    else "Visible-spectrum vegetation support is unavailable because an RGB-like optical representation was not established.",
                ],
                sar_observations=[
                    f"SAR evidence used {sar.preparation.band_count} available band(s) as relative, uncalibrated intensity.",
                    "Low relative intensity and spatial smoothness support water-likelihood; high relative intensity or heterogeneity support structural likelihood.",
                ],
                joint_observations=[water_text, structural_text, agreement_text],
            ),
            statistics=statistics,
            regions=regions,
            previews=previews,
            confidence=confidence,
            method=_method(),
            optical_preparation=optical.preparation,
            sar_preparation=sar.preparation,
            warnings=_unique(optical_metadata.warnings + sar_metadata.warnings + compatibility.warnings + warnings),
            runtime_ms=_duration(started_total),
            stage_durations_ms=durations,
        )


def template_summary(result: CrossModalResult) -> Optional[str]:
    """Produce statistics-only controlled text for the existing agent answer field."""
    if result.statistics is None:
        return result.summary.joint_observations[0] if result.summary.joint_observations else None
    statistics = result.statistics
    agreement = (
        f" Optical and SAR candidate evidence agrees over {statistics.agreement_percent:.1f}% of evaluated candidate pixels."
        if statistics.agreement_percent is not None
        else " Agreement is unavailable because no candidate pixels were evaluated."
    )
    joint_regions = sum(region.type != "disagreement" for region in result.regions)
    return (
        f"Both modalities support water-like conditions over {statistics.water_likelihood_percent:.1f}% of valid pixels "
        f"and structural-likelihood conditions over {statistics.built_up_likelihood_percent:.1f}%."
        f"{agreement} {joint_regions} joint evidence region(s) survived morphological cleanup."
    )


_ANALYZER = CrossModalOpticalSarAnalyzer()


def get_cross_modal_analyzer() -> CrossModalOpticalSarAnalyzer:
    return _ANALYZER
