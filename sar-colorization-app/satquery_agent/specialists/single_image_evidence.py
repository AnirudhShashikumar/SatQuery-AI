"""Deterministic visible-spectrum evidence for controlled single-image VQA.

The outputs are scene-relative heuristic support maps. They are not trained
semantic segmentation, calibrated probabilities, or ground truth.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

from ..image_ingestion import remove_preview_url, save_preview
from ..models import (
    ImageMetadata,
    SingleImageEvidenceMethod,
    SingleImageEvidencePreviews,
    SingleImageEvidenceRegion,
    SingleImageEvidenceResult,
    SingleImageEvidenceStatistics,
)


METHOD_NAME = "deterministic single-image evidence extraction"
METHOD_VERSION = "1.0"
METHOD_ASSUMPTIONS = [
    "The uploaded optical input is RGB or RGB-like; the first three prepared bands are treated as red, green, and blue.",
    "All thresholds are relative to the uploaded scene and indicate candidate support rather than semantic probability.",
    "Visible-spectrum colour, brightness, edges, and local texture are usable at the reduced analysis resolution.",
]
METHOD_LIMITATIONS = [
    "This method is not a trained or calibrated land-cover or building classifier.",
    "Dark smooth terrain, cloud shadow, terrain shadow, and dark roofs can resemble water support.",
    "Bright soil, roads, rooftops, speckled texture, and vegetation structure can resemble built-up support.",
    "Visible green dominance is not NDVI and does not establish vegetation from near-infrared reflectance.",
    "Agricultural support is a field-like texture heuristic and cannot identify crop type or confirm land use.",
    "Analysis may use a grid reduced to 1024 pixels on its longest side, so small features can be omitted.",
]


@dataclass(frozen=True)
class EvidenceThresholdConfig:
    weak_evidence_percent: float = 2.0
    moderate_evidence_percent: float = 8.0
    strong_evidence_percent: float = 20.0
    dominant_class_margin_percent: float = 7.0
    minimum_region_pixels: int = 6
    minimum_region_fraction: float = 0.0001
    morphology_radius: int = 1
    low_valid_pixel_percent: float = 70.0


DEFAULT_THRESHOLDS = EvidenceThresholdConfig()


@dataclass
class EvidenceExtraction:
    result: SingleImageEvidenceResult
    stage_durations_ms: Dict[str, int]
    threshold_parameters: Dict[str, float]


class SingleImageEvidenceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def safe_threshold_parameters(config: EvidenceThresholdConfig = DEFAULT_THRESHOLDS) -> Dict[str, float]:
    return {name: float(value) for name, value in asdict(config).items()}


def _duration(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _percent(mask: np.ndarray, denominator: int) -> float:
    value = float(np.count_nonzero(mask)) * 100.0 / denominator if denominator else 0.0
    return round(min(100.0, max(0.0, value)), 6)


def _valid_pixels(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.all(np.isfinite(array), axis=2)
    if nodata is not None:
        valid &= ~np.any(np.isclose(array, nodata), axis=2)
    return valid


def _normalize_rgb(array: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]], bool]:
    normalized = np.zeros(array.shape, dtype=np.float32)
    limits: List[Tuple[float, float]] = []
    dynamic = False
    for channel in range(3):
        selected = array[:, :, channel][valid]
        low, high = (float(value) for value in np.percentile(selected, [2.0, 98.0]))
        if not math.isfinite(low) or not math.isfinite(high):
            raise SingleImageEvidenceError("NON_FINITE_RASTER", "The optical raster has a non-finite channel range.")
        if high <= low:
            low, high = float(np.min(selected)), float(np.max(selected))
        limits.append((low, high))
        if high > low:
            dynamic = True
            normalized[:, :, channel] = np.clip((array[:, :, channel] - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized, limits, dynamic


def _local_std(values: np.ndarray, size: int) -> np.ndarray:
    mean = uniform_filter(values.astype(np.float32), size=size, mode="reflect")
    mean_square = uniform_filter(np.square(values, dtype=np.float32), size=size, mode="reflect")
    return np.sqrt(np.maximum(mean_square - np.square(mean), 0.0)).astype(np.float32)


def _quantile(values: np.ndarray, valid: np.ndarray, fraction: float, fallback: float) -> float:
    selected = values[valid]
    return float(np.quantile(selected, fraction)) if selected.size else fallback


def _clean(mask: np.ndarray, config: EvidenceThresholdConfig) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    footprint = disk(config.morphology_radius)
    cleaned = binary_closing(mask, footprint)
    cleaned = binary_opening(cleaned, footprint)
    minimum = max(config.minimum_region_pixels, int(math.ceil(mask.size * config.minimum_region_fraction)))
    return remove_small_objects(cleaned, min_size=minimum, connectivity=2)


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
    scale_x, scale_y = metadata.width / analysis_shape[1], metadata.height / analysis_shape[0]
    corners = [
        _pixel_to_world(metadata.transform, x * scale_x, y * scale_y)
        for x, y in ((left, top), (right, top), (left, bottom), (right, bottom))
    ]
    xs, ys = zip(*corners)
    cx, cy = _pixel_to_world(metadata.transform, centroid[0] * scale_x, centroid[1] * scale_y)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], [float(cx), float(cy)]


def _regions(mask: np.ndarray, kind: str, valid_count: int, metadata: ImageMetadata) -> List[SingleImageEvidenceRegion]:
    components = label(mask, connectivity=2)
    properties = sorted(regionprops(components), key=lambda item: int(item.area), reverse=True)[:100]
    output: List[SingleImageEvidenceRegion] = []
    for index, region in enumerate(properties, start=1):
        top, left, bottom, right = (int(value) for value in region.bbox)
        row, column = (float(value) for value in region.centroid)
        bbox = (left, top, right, bottom)
        centroid = (column, row)
        bbox_world, centroid_world = _world_geometry(metadata, bbox, centroid, mask.shape)
        output.append(
            SingleImageEvidenceRegion(
                region_id=f"{kind}-{index}",
                type=kind,
                area_pixels=int(region.area),
                area_percent=_percent(components == region.label, valid_count),
                bbox_pixels=list(bbox),
                centroid_pixels=[round(column, 3), round(row, 3)],
                bbox_world=bbox_world,
                centroid_world=centroid_world,
            )
        )
    return output


def _mask_image(mask: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask] = color
    return Image.fromarray(output)


def _save_previews(
    rgb: np.ndarray,
    water: np.ndarray,
    vegetation: np.ndarray,
    structural: np.ndarray,
    agriculture: np.ndarray,
) -> SingleImageEvidencePreviews:
    saved: List[str] = []
    try:
        base = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        overlay = base.astype(np.float32)
        for mask, color in (
            (water, np.array([30.0, 145.0, 255.0])),
            (vegetation, np.array([45.0, 205.0, 105.0])),
            (structural, np.array([255.0, 150.0, 35.0])),
            (agriculture, np.array([215.0, 215.0, 55.0])),
        ):
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
        images = (
            _mask_image(water, (30, 145, 255)),
            _mask_image(vegetation, (45, 205, 105)),
            _mask_image(structural, (255, 150, 35)),
            _mask_image(agriculture, (215, 215, 55)),
            Image.fromarray(np.round(overlay).astype(np.uint8)),
        )
        for image in images:
            saved.append(save_preview(image))
            image.close()
        return SingleImageEvidencePreviews(
            water_support=saved[0],
            vegetation_support=saved[1],
            built_up_support=saved[2],
            agriculture_support=saved[3],
            combined_overlay=saved[4],
        )
    except Exception:
        for preview in saved:
            remove_preview_url(preview)
        raise


def _dominant_scene(statistics: SingleImageEvidenceStatistics, config: EvidenceThresholdConfig) -> str:
    scores = {
        "urban": statistics.built_up_support_percent,
        "agricultural": statistics.agriculture_support_percent + 0.35 * statistics.vegetation_support_percent,
        "vegetated": max(0.0, statistics.vegetation_support_percent - 0.5 * statistics.agriculture_support_percent),
        "water_dominant": statistics.water_support_percent,
        "barren": statistics.barren_support_percent,
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_value = ordered[0]
    runner_up = ordered[1][1]
    if top_value < config.moderate_evidence_percent:
        return "uncertain"
    if top_value - runner_up < config.dominant_class_margin_percent:
        return "mixed"
    return top_name


def extract_single_image_evidence(
    raster: np.ndarray,
    metadata: ImageMetadata,
    config: EvidenceThresholdConfig = DEFAULT_THRESHOLDS,
) -> EvidenceExtraction:
    overall_started = time.perf_counter()
    durations: Dict[str, int] = {}
    warnings: List[str] = []

    started = time.perf_counter()
    if raster.ndim != 3 or raster.shape[2] < 3 or metadata.band_count < 3:
        raise SingleImageEvidenceError(
            "UNSUPPORTED_OPTICAL_BANDS",
            "Controlled VQA requires an optical RGB or RGB-like input with at least three bands.",
        )
    raw = raster[:, :, :3].astype(np.float64, copy=False)
    valid = _valid_pixels(raw, metadata.nodata)
    if not np.any(valid):
        raise SingleImageEvidenceError("NO_VALID_PIXELS", "The image contains no finite, non-nodata RGB pixels.")
    rgb, limits, dynamic = _normalize_rgb(raw, valid)
    durations["optical_image_preparation"] = _duration(started)

    started = time.perf_counter()
    brightness = np.mean(rgb, axis=2)
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    texture = _local_std(brightness, 5)
    edge = np.hypot(sobel(brightness, axis=0, mode="reflect"), sobel(brightness, axis=1, mode="reflect"))
    dark_threshold = min(0.38, max(0.18, _quantile(brightness, valid, 0.35, 0.30)))
    smooth_threshold = max(0.035, _quantile(texture, valid, 0.55, 0.06))
    edge_threshold = max(0.055, _quantile(edge, valid, 0.75, 0.08))
    texture_threshold = max(0.045, _quantile(texture, valid, 0.72, 0.06))
    low_texture = _quantile(texture, valid, 0.30, 0.025)
    high_texture = max(texture_threshold, _quantile(texture, valid, 0.82, 0.09))
    edge_binary = edge >= edge_threshold
    local_edge_density = uniform_filter(edge_binary.astype(np.float32), size=11, mode="reflect")
    texture_regularity = _local_std(texture, 11)
    regularity_threshold = max(0.012, _quantile(texture_regularity, valid, 0.62, 0.025))

    vegetation = valid & (green >= red + 0.06) & (green >= blue + 0.04) & (saturation >= 0.10) & (brightness >= 0.12)
    water = valid & (brightness <= dark_threshold) & (texture <= smooth_threshold) & (blue >= red - 0.03) & (blue >= 0.72 * green) & ~vegetation
    structural = valid & (brightness >= 0.18) & ((edge >= edge_threshold) | (texture >= texture_threshold)) & ~water & ~vegetation
    barren = valid & ~water & ~vegetation & (brightness >= _quantile(brightness, valid, 0.48, 0.45)) & (saturation <= max(0.10, _quantile(saturation, valid, 0.45, 0.15))) & (texture <= high_texture)
    agriculture = valid & vegetation & (texture >= low_texture) & (texture <= high_texture) & (local_edge_density >= 0.04) & (local_edge_density <= 0.48) & (texture_regularity <= regularity_threshold)

    if not dynamic:
        water[:] = False
        vegetation[:] = False
        structural[:] = False
        barren[:] = False
        agriculture[:] = False
        warnings.append("The optical input has negligible dynamic range; semantic support maps were left empty.")
    water, vegetation = _clean(water, config), _clean(vegetation, config)
    structural, barren = _clean(structural, config), _clean(barren, config)
    agriculture = _clean(agriculture, config)
    durations["evidence_extraction"] = _duration(started)

    started = time.perf_counter()
    valid_count = int(np.count_nonzero(valid))
    statistics = SingleImageEvidenceStatistics(
        water_support_percent=_percent(water, valid_count),
        vegetation_support_percent=_percent(vegetation, valid_count),
        built_up_support_percent=_percent(structural, valid_count),
        barren_support_percent=_percent(barren, valid_count),
        agriculture_support_percent=_percent(agriculture, valid_count),
        edge_density_percent=_percent(edge_binary & valid, valid_count),
        valid_pixel_percent=_percent(valid, valid.size),
    )
    dominant = "uncertain" if not dynamic else _dominant_scene(statistics, config)
    regions: List[SingleImageEvidenceRegion] = []
    regions.extend(_regions(water, "water_support", valid_count, metadata))
    regions.extend(_regions(structural, "built_up_support", valid_count, metadata))
    regions.extend(_regions(vegetation, "vegetation_support", valid_count, metadata))
    durations["scene_summary_computation"] = _duration(started)

    started = time.perf_counter()
    previews = _save_previews(rgb, water, vegetation, structural, agriculture)
    durations["evidence_preview_generation"] = _duration(started)

    if statistics.valid_pixel_percent < config.low_valid_pixel_percent:
        warnings.append("A large fraction of pixels is invalid or NoData; evidence coverage may be unreliable.")
    warnings.extend(
        [
            "Water support may include shadows or other dark smooth surfaces.",
            "Built-up support indicates visible structural complexity, not confirmed buildings.",
            "Agricultural support is heuristic and does not identify crop type.",
        ]
    )
    limits_text = ", ".join(f"{low:.6g}–{high:.6g}" for low, high in limits)
    warnings.append(f"RGB preparation used per-channel finite-pixel 2nd/98th percentile limits: {limits_text}.")
    result = SingleImageEvidenceResult(
        statistics=statistics,
        dominant_scene=dominant,
        regions=regions,
        previews=previews,
        warnings=list(dict.fromkeys(warnings)),
        method=SingleImageEvidenceMethod(
            name=METHOD_NAME,
            version=METHOD_VERSION,
            uses_trained_classifier=False,
            assumptions=METHOD_ASSUMPTIONS,
            limitations=METHOD_LIMITATIONS,
        ),
        low_information=not dynamic or statistics.valid_pixel_percent < config.low_valid_pixel_percent,
        runtime_ms=_duration(overall_started),
    )
    return EvidenceExtraction(
        result=result,
        stage_durations_ms=durations,
        threshold_parameters=safe_threshold_parameters(config),
    )
