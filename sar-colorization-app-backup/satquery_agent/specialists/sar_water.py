"""Conservative single-image SAR water-candidate analysis."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import measure, morphology
from skimage.filters import threshold_otsu

try:
    from rasterio.crs import CRS
except ImportError:  # pragma: no cover - rasterio is present in the production backend
    CRS = None  # type: ignore[assignment]

from ..image_ingestion import save_preview
from ..models import (
    ImageMetadata,
    RepresentationType,
    SarEvidenceProduct,
    SarRegion,
    SarWaterResult,
)
from .sar_preprocessing import PreparedSar, SarPreprocessingError, preprocess_sar


SPECIALIST_ID = "sar_water_segmenter"
SPECIALIST_VERSION = "heuristic-sar-water-1.0"
METHOD_NAME = "heuristic_candidate_detector"


class SarWaterAnalysisError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _elapsed(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _position(centroid: Tuple[float, float], width: int, height: int) -> str:
    x, y = centroid
    horizontal = "left" if x < width / 3 else "right" if x > width * 2 / 3 else "middle"
    vertical = "upper" if y < height / 3 else "lower" if y > height * 2 / 3 else "central"
    return f"{vertical}-{horizontal}"


def _projected_area_square_meters(metadata: ImageMetadata, image_area_percent: float) -> Tuple[Optional[float], Optional[str]]:
    """Estimate area only when affine units can be safely converted to metres."""
    if not metadata.is_georeferenced or not metadata.crs or not metadata.transform or CRS is None:
        return None, None
    try:
        crs = CRS.from_string(metadata.crs)
        if not crs.is_projected:
            return None, None
        _unit_name, metres_per_unit = crs.linear_units_factor
        a, b, _c, d, e, _f = metadata.transform
        source_pixel_area = abs(a * e - b * d) * float(metres_per_unit) ** 2
        if not np.isfinite(source_pixel_area) or source_pixel_area <= 0:
            return None, None
        total_area = source_pixel_area * metadata.width * metadata.height
        return (
            round(float(total_area * image_area_percent / 100.0), 3),
            "projected affine pixel area multiplied by segmented image fraction",
        )
    except (TypeError, ValueError, AttributeError):
        return None, None


def _make_evidence(prepared: PreparedSar, mask: np.ndarray, metadata: ImageMetadata) -> Tuple[List[SarEvidenceProduct], Dict[str, int]]:
    started = time.perf_counter()
    gray = np.round(np.clip(prepared.normalized, 0, 1) * 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    base = np.stack([gray, gray, gray], axis=-1)
    overlay = base.astype(np.float32)
    color = np.array([25, 180, 255], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.48 + color * 0.52
    boundary = morphology.binary_dilation(mask, morphology.disk(1)) ^ morphology.binary_erosion(mask, morphology.disk(1))
    overlay[boundary] = np.array([255, 225, 70], dtype=np.float32)
    normalized_url = save_preview(Image.fromarray(gray))
    mask_url = save_preview(Image.fromarray(mask_u8))
    overlay_url = save_preview(Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)))
    width, height = int(mask.shape[1]), int(mask.shape[0])
    common = {"width": width, "height": height, "source": SPECIALIST_ID, "authoritative": True}
    products = [
        SarEvidenceProduct(
            id=uuid.uuid4().hex, type="source_preview", label="Original source preview",
            description="Backend-generated display preview of the uploaded source; original scientific values remain separate.",
            reference=metadata.preview_url or normalized_url, generation_method="ingestion display preview", **common,
        ),
        SarEvidenceProduct(
            id=uuid.uuid4().hex, type="normalized_sar_preview", label="Normalized SAR preview",
            description="Finite-pixel percentile normalization used by the heuristic; no logarithmic transform was applied.",
            reference=normalized_url, generation_method=prepared.details.normalization, **common,
        ),
        SarEvidenceProduct(
            id=uuid.uuid4().hex, type="binary_candidate_mask", label="Binary water-candidate mask",
            description="Connected low-backscatter candidate pixels retained after deterministic morphology.",
            reference=mask_url, generation_method=SPECIALIST_VERSION, **common,
        ),
        SarEvidenceProduct(
            id=uuid.uuid4().hex, type="candidate_overlay", label="Water-candidate overlay",
            description="Blue candidate regions and yellow component outlines over the normalized SAR display.",
            reference=overlay_url, generation_method=SPECIALIST_VERSION, **common,
        ),
    ]
    return products, {"evidence_generation": _elapsed(started)}


def analyze_sar_water(raster: np.ndarray, metadata: ImageMetadata) -> Tuple[SarWaterResult, str]:
    overall_started = time.perf_counter()
    try:
        prepared = preprocess_sar(raster, metadata)
    except SarPreprocessingError as error:
        raise SarWaterAnalysisError(error.code, error.message) from error

    segmentation_started = time.perf_counter()
    valid_values = prepared.denoised[prepared.valid_mask]
    if prepared.low_information:
        threshold = None
        mask = np.zeros(prepared.valid_mask.shape, dtype=bool)
    else:
        otsu = float(threshold_otsu(valid_values))
        conservative_cap = float(np.percentile(valid_values, 35.0))
        threshold = max(0.0, min(1.0, min(otsu, conservative_cap)))
        mask = prepared.valid_mask & (prepared.denoised <= threshold)
    segmentation_ms = _elapsed(segmentation_started)

    morphology_started = time.perf_counter()
    minimum_region = max(12, int(mask.size * 0.0005))
    if mask.any():
        mask = morphology.binary_opening(mask, morphology.disk(1))
        mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = morphology.remove_small_objects(mask, min_size=minimum_region)
        mask = morphology.remove_small_holes(mask, area_threshold=max(8, minimum_region // 2))
        mask &= prepared.valid_mask
    morphology_ms = _elapsed(morphology_started)

    components_started = time.perf_counter()
    labels = measure.label(mask, connectivity=2)
    regions: List[SarRegion] = []
    for region in sorted(measure.regionprops(labels), key=lambda item: item.area, reverse=True):
        min_row, min_col, max_row, max_col = region.bbox
        regions.append(SarRegion(
            region_id=len(regions) + 1,
            area_pixels=int(region.area),
            image_area_percent=round(float(region.area) * 100.0 / mask.size, 4),
            bounding_box=[int(min_col), int(min_row), int(max_col), int(max_row)],
            centroid=[round(float(region.centroid[1]), 2), round(float(region.centroid[0]), 2)],
        ))
    component_ms = _elapsed(components_started)
    candidate_pixels = int(mask.sum())
    valid_pixels = int(prepared.valid_mask.sum())
    area_percent = round(candidate_pixels * 100.0 / mask.size, 4)

    if candidate_pixels and valid_values.size:
        candidate_mean = float(prepared.denoised[mask].mean())
        background = prepared.denoised[prepared.valid_mask & ~mask]
        contrast = max(0.0, float(background.mean()) - candidate_mean) if background.size else 0.0
        contrast_score = min(1.0, contrast / 0.35)
        coherence = regions[0].area_pixels / candidate_pixels if regions else 0.0
        fragmentation = 1.0 / (1.0 + max(0, len(regions) - 1) * 0.12)
        reliability = round(float(np.clip(0.15 + 0.45 * contrast_score + 0.25 * coherence + 0.15 * fragmentation, 0, 1)), 3)
    else:
        reliability = 0.2 if prepared.low_information else 0.35
    input_quality = round(float(np.clip(1.0 - (prepared.details.invalid_pixel_count + prepared.details.nodata_pixel_count) / max(mask.size, 1), 0, 1)), 3)
    geographic_area, geographic_area_method = _projected_area_square_meters(metadata, area_percent)

    evidence, evidence_durations = _make_evidence(prepared, mask, metadata)
    limitations = [
        "This is an explicit low-backscatter heuristic, not a trained water-segmentation model.",
        "Radar shadow, smooth man-made surfaces, and other low-return areas may resemble water.",
    ]
    if metadata.representation == RepresentationType.DISPLAY_PREVIEW:
        limitations.append("The input is a display preview without calibrated backscatter or polarization metadata; analysis is qualitative.")
    else:
        limitations.append("The SAR value domain (linear amplitude, power, or decibels) was not verified; no logarithmic transform was applied.")
    if not metadata.is_georeferenced:
        limitations.append("The image has no usable georeference, so only image-area percentage—not geographic area—is reported.")
    elif geographic_area is None:
        limitations.append("The georeference is not a safely convertible projected CRS, so geographic area is not reported.")
    else:
        limitations.append("Geographic area is an image-derived estimate from projected pixel area, not a surveyed measurement.")
    if prepared.low_information:
        limitations.append("The image has insufficient intensity variation for dependable low-backscatter separation.")
    water_detected = candidate_pixels > 0
    if water_detected and regions:
        location = _position((regions[0].centroid[0], regions[0].centroid[1]), mask.shape[1], mask.shape[0])
        answer = f"Probable low-backscatter water candidates were identified, led by a connected region in the {location} portion of the image. The candidates cover approximately {area_percent:.2f}% of image pixels."
    else:
        answer = "No dependable low-backscatter water candidate was retained by the conservative heuristic for this image."
    durations = {
        **prepared.durations_ms,
        "heuristic_segmentation": segmentation_ms,
        "morphological_postprocessing": morphology_ms,
        "connected_components": component_ms,
        **evidence_durations,
    }
    result = SarWaterResult(
        execution_status="completed_with_limitations",
        method=METHOD_NAME,
        method_version=SPECIALIST_VERSION,
        water_detected=water_detected,
        image_area_percent=area_percent,
        geographic_area_square_meters=geographic_area,
        geographic_area_method=geographic_area_method,
        model_confidence=None,
        heuristic_reliability=reliability,
        input_quality_score=input_quality,
        threshold=round(threshold, 6) if threshold is not None else None,
        candidate_pixels=candidate_pixels,
        valid_pixels=valid_pixels,
        regions=regions,
        evidence_products=evidence,
        preprocessing=prepared.details,
        rationale=[
            "Water can appear as spatially coherent low-backscatter regions in SAR imagery.",
            "The threshold is derived from this image's finite normalized values and capped conservatively.",
            "Only connected candidates surviving deterministic morphology are included.",
        ],
        limitations=limitations,
        warnings=["Low-backscatter candidates are evidence for review, not ground truth."],
        runtime_ms=_elapsed(overall_started),
        stage_durations_ms=durations,
    )
    return result, answer
