"""Factual SAR intensity/quality summaries without semantic hallucination."""

from __future__ import annotations

import time
import uuid

import numpy as np
from PIL import Image

from ..image_ingestion import save_preview
from ..models import ImageMetadata, RepresentationType, SarEvidenceProduct, SarSceneResult, TaskType
from .sar_preprocessing import SarPreprocessingError, preprocess_sar


SPECIALIST_VERSION = "deterministic-sar-scene-1.0"


class SarSceneAnalysisError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def analyze_sar_scene(raster: np.ndarray, metadata: ImageMetadata, task: TaskType) -> tuple[SarSceneResult, str]:
    started = time.perf_counter()
    try:
        prepared = preprocess_sar(raster, metadata)
    except SarPreprocessingError as error:
        raise SarSceneAnalysisError(error.code, error.message) from error
    statistics_started = time.perf_counter()
    values = prepared.normalized[prepared.valid_mask]
    low = float((values <= 0.33).mean() * 100)
    high = float((values >= 0.67).mean() * 100)
    mid = max(0.0, 100.0 - low - high)
    gy, gx = np.gradient(prepared.denoised)
    gradient = np.sqrt(gx * gx + gy * gy)
    texture = float(np.clip(np.percentile(gradient[prepared.valid_mask], 75) / 0.3, 0, 1))
    valid_percent = float(prepared.valid_mask.mean() * 100)
    input_quality = float(np.clip((valid_percent / 100) * min(1.0, float(np.std(values)) / 0.12), 0, 1))
    statistics_ms = max(0, round((time.perf_counter() - statistics_started) * 1000))
    gray = np.round(prepared.normalized * 255).astype(np.uint8)
    preview_url = save_preview(Image.fromarray(gray))
    height, width = gray.shape
    evidence = [SarEvidenceProduct(
        id=uuid.uuid4().hex,
        type="normalized_sar_preview",
        label="Normalized SAR intensity preview",
        description="The normalized values used for the reported distribution and texture statistics.",
        source="sar_scene_analyzer",
        authoritative=True,
        reference=preview_url,
        width=width,
        height=height,
        generation_method=prepared.details.normalization,
    )]
    limitations = [
        "Intensity and texture statistics do not establish semantic land-cover classes or object identity.",
        "No trained SAR scene classifier was used and no model confidence is available.",
    ]
    if metadata.representation == RepresentationType.DISPLAY_PREVIEW:
        limitations.append("The display preview lacks calibrated backscatter and polarization metadata; analysis is qualitative.")
    if not metadata.is_georeferenced:
        limitations.append("No geographic measurements are reported because the input lacks a usable georeference.")
    texture_label = "texture-rich" if texture >= 0.6 else "moderately textured" if texture >= 0.3 else "relatively smooth"
    if task == TaskType.SAR_QUALITY_INSPECTION:
        answer = f"The SAR input has {valid_percent:.1f}% valid pixels and is {texture_label}, with a normalized intensity standard deviation of {float(np.std(values)):.3f}. It is suitable for qualitative intensity analysis; scientific backscatter interpretation remains unavailable without calibrated metadata."
    else:
        answer = f"The SAR intensity representation is {texture_label}. Low-, mid-, and high-normalized-return pixels account for approximately {low:.1f}%, {mid:.1f}%, and {high:.1f}% of valid pixels. These are backscatter-pattern observations, not semantic land-cover labels."
    runtime = max(0, round((time.perf_counter() - started) * 1000))
    result = SarSceneResult(
        execution_status="completed_with_limitations",
        task=task.value,
        method="deterministic_intensity_texture_summary",
        method_version=SPECIALIST_VERSION,
        valid_pixel_percent=round(valid_percent, 4),
        low_backscatter_percent=round(low, 4),
        mid_backscatter_percent=round(mid, 4),
        high_backscatter_percent=round(high, 4),
        normalized_mean=round(float(values.mean()), 6),
        normalized_standard_deviation=round(float(values.std()), 6),
        texture_index=round(texture, 4),
        input_quality_score=round(input_quality, 4),
        evidence_products=evidence,
        preprocessing=prepared.details,
        limitations=limitations,
        warnings=["Scene statistics are deterministic qualitative evidence, not a trained semantic classification."],
        runtime_ms=runtime,
        stage_durations_ms={**prepared.durations_ms, "sar_scene_statistics": statistics_ms},
    )
    return result, answer
