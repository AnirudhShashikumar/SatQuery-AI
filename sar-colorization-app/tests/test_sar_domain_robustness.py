"""Sensor-agnostic SAR normalization and provenance tests."""

import numpy as np
import pytest

from satquery_agent.models import ImageFormat, ImageMetadata, ImageModality, RepresentationType
from satquery_agent.specialists.sar_preprocessing import preprocess_sar
from satquery_agent.services.sar_translation_service import _sarfusionformer_eligibility


def metadata(dtype: str, channels: int, *, modality=ImageModality.SAR_VV, descriptions=None, confirmed=None):
    return ImageMetadata(
        file_id="sar",
        original_name="sar.tif",
        safe_name="sar.tif",
        format=ImageFormat.TIFF,
        mime_type="image/tiff",
        size_bytes=1,
        width=16,
        height=16,
        band_count=channels,
        dtype=dtype,
        band_descriptions=descriptions or [None] * channels,
        representation=RepresentationType.SCIENTIFIC_RASTER,
        effective_modality=modality,
        user_confirmed_modality=confirmed,
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_sar_is_normalized_without_fixed_sensor_range(dtype) -> None:
    raster = np.arange(256, dtype=dtype).reshape(16, 16, 1)
    result = preprocess_sar(raster, metadata(str(np.dtype(dtype)), 1))
    assert result.details.input_value_domain == "integer_unknown_scale"
    assert result.details.input_dtype == str(np.dtype(dtype))
    assert result.details.percentile_highs[0] > result.details.percentile_lows[0]
    assert result.normalized.min() >= 0 and result.normalized.max() <= 1


def test_float_negative_values_are_only_marked_db_like_unverified() -> None:
    raster = np.linspace(-30, 5, 256, dtype=np.float32).reshape(16, 16, 1)
    result = preprocess_sar(raster, metadata("float32", 1))
    assert result.details.input_value_domain == "db_like_unverified"
    assert "possible but unverified" in result.details.value_domain_reason


def test_float_nonnegative_values_do_not_assume_amplitude_or_power() -> None:
    raster = np.linspace(0, 500, 256, dtype=np.float32).reshape(16, 16, 1)
    result = preprocess_sar(raster, metadata("float32", 1))
    assert result.details.input_value_domain == "amplitude_or_power_like_unverified"
    assert "amplitude versus power is not inferred" in result.details.value_domain_reason


def test_single_band_is_not_duplicated_and_unknown_polarization_is_visible() -> None:
    raster = np.arange(256, dtype=np.uint16).reshape(16, 16, 1)
    result = preprocess_sar(raster, metadata("uint16", 1, modality=ImageModality.UNKNOWN))
    assert result.details.input_channel_count == 1
    assert result.details.polarization_labels == ["unknown_channel_1"]


def test_sarfusionformer_requires_explicit_vv_vh_mapping() -> None:
    raster = np.zeros((16, 16, 2), dtype=np.float32)
    ambiguous = metadata("float32", 2, modality=ImageModality.SAR_VV_VH)
    explicit = metadata(
        "float32", 2, modality=ImageModality.SAR_VV_VH,
        descriptions=["VV", "VH"], confirmed=ImageModality.SAR_VV_VH,
    )
    assert _sarfusionformer_eligibility(raster, ambiguous)[0] is False
    assert _sarfusionformer_eligibility(raster, explicit)[0] is True
