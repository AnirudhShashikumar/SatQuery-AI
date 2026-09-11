"""Reusable, value-preserving preprocessing for single-image SAR analysis."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from skimage.filters import median
from skimage.morphology import disk

from ..models import ImageMetadata, ImageModality, RepresentationType, SarPreprocessingDetails


PREPROCESSING_VERSION = "sar-preprocess-1.0"


class SarPreprocessingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PreparedSar:
    normalized: np.ndarray
    denoised: np.ndarray
    valid_mask: np.ndarray
    details: SarPreprocessingDetails
    durations_ms: Dict[str, int]
    low_information: bool


def _elapsed(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _value_domain(values: np.ndarray, valid: np.ndarray, requested: str) -> tuple[str, str]:
    if requested != "unknown":
        return requested, "Value domain was explicitly supplied by the caller."
    sample = values[valid]
    if sample.size == 0:
        return "unknown", "No valid samples were available for value-domain inspection."
    low, high = (float(value) for value in np.percentile(sample, [1.0, 99.0]))
    if np.issubdtype(values.dtype, np.integer):
        return "integer_unknown_scale", f"Integer SAR values span p1={low:.6g} to p99={high:.6g}; calibration is not assumed."
    if low < 0.0 and high <= 100.0:
        return "db_like_unverified", f"Floating values include negatives (p1={low:.6g}, p99={high:.6g}); dB-like encoding is possible but unverified."
    if low >= 0.0:
        return "amplitude_or_power_like_unverified", f"Floating values are non-negative (p1={low:.6g}, p99={high:.6g}); amplitude versus power is not inferred."
    return "floating_unknown_scale", f"Floating SAR values span p1={low:.6g} to p99={high:.6g}; calibration is not assumed."


def _polarizations(metadata: ImageMetadata, channel_count: int) -> list[str]:
    labels = [str(value or "").strip().upper() for value in metadata.band_descriptions[:channel_count]]
    if labels and all(label in {"VV", "VH", "HH", "HV"} for label in labels):
        return labels
    if metadata.effective_modality == ImageModality.SAR_VV:
        return ["VV"]
    if metadata.effective_modality == ImageModality.SAR_VH:
        return ["VH"]
    if metadata.effective_modality == ImageModality.SAR_VV_VH and channel_count == 2:
        return ["VV", "VH"] if metadata.user_confirmed_modality == ImageModality.SAR_VV_VH else ["unverified_channel_1", "unverified_channel_2"]
    return [f"unknown_channel_{index + 1}" for index in range(channel_count)]


def preprocess_sar(
    raster: np.ndarray,
    metadata: ImageMetadata,
    *,
    value_domain: str = "unknown",
    denoise: bool = True,
) -> PreparedSar:
    """Prepare one or two SAR channels without changing the preserved source array."""
    started = time.perf_counter()
    values = np.asarray(raster)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[2] not in {1, 2}:
        raise SarPreprocessingError(
            "UNSUPPORTED_SAR_CHANNELS",
            "Single-image SAR analysis requires one channel or a verified VV/VH pair.",
        )
    numeric = values.astype(np.float64, copy=True)
    invalid = ~np.isfinite(numeric)
    nodata_mask = np.zeros(numeric.shape, dtype=bool)
    if metadata.nodata is not None and math.isfinite(metadata.nodata):
        nodata_mask = np.isclose(numeric, metadata.nodata, equal_nan=False)
    valid_channels = ~(invalid | nodata_mask)
    valid_mask = np.all(valid_channels, axis=2)
    if not valid_mask.any():
        raise SarPreprocessingError("NO_VALID_SAR_PIXELS", "The SAR input contains no finite, non-NoData pixels.")
    validation_ms = _elapsed(started)
    inferred_domain, domain_reason = _value_domain(values, valid_channels, value_domain)

    normalize_started = time.perf_counter()
    channel_outputs = []
    lows = []
    highs = []
    for channel in range(numeric.shape[2]):
        channel_values = numeric[:, :, channel]
        channel_valid = valid_channels[:, :, channel]
        finite_values = channel_values[channel_valid]
        low, high = np.percentile(finite_values, [1.0, 99.0])
        if not math.isfinite(float(low)) or not math.isfinite(float(high)):
            raise SarPreprocessingError("INVALID_SAR_RANGE", "The SAR input has an invalid numeric range.")
        lows.append(float(low))
        highs.append(float(high))
        if high <= low:
            scaled = np.full(channel_values.shape, 0.5, dtype=np.float32)
        else:
            scaled = np.clip((channel_values - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)
        scaled[~channel_valid] = 0.0
        channel_outputs.append(scaled)
    normalized = np.mean(np.stack(channel_outputs, axis=-1), axis=-1, dtype=np.float32)
    normalized[~valid_mask] = 0.0
    normalization_ms = _elapsed(normalize_started)

    denoise_started = time.perf_counter()
    denoised = median(normalized, footprint=disk(1)).astype(np.float32) if denoise else normalized.copy()
    denoised[~valid_mask] = 0.0
    denoising_ms = _elapsed(denoise_started)
    valid_normalized = normalized[valid_mask]
    low_information = bool(valid_normalized.size == 0 or float(np.std(valid_normalized)) < 0.015 or float(np.ptp(valid_normalized)) < 0.04)
    details = SarPreprocessingDetails(
        version=PREPROCESSING_VERSION,
        input_value_domain=inferred_domain,
        log_transform_applied=False,
        normalization="finite non-NoData 1st/99th percentile per channel, then channel mean",
        percentile_low=float(np.mean(lows)),
        percentile_high=float(np.mean(highs)),
        invalid_pixel_count=int(invalid.any(axis=2).sum()),
        nodata_pixel_count=int(nodata_mask.any(axis=2).sum()),
        denoising="3x3 median filter" if denoise else "none",
        resized=bool(values.shape[1] != metadata.width or values.shape[0] != metadata.height),
        input_dtype=str(values.dtype),
        input_channel_count=int(values.shape[2]),
        polarization_labels=_polarizations(metadata, int(values.shape[2])),
        value_domain_reason=domain_reason,
        percentile_lows=lows,
        percentile_highs=highs,
    )
    return PreparedSar(
        normalized=normalized,
        denoised=denoised,
        valid_mask=valid_mask,
        details=details,
        durations_ms={
            "sar_input_validation": validation_ms,
            "sar_normalization": normalization_ms,
            "sar_denoising": denoising_ms,
        },
        low_information=low_information,
    )
