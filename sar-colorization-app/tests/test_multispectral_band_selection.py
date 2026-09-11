"""Explicit multispectral visual-band selection regression tests."""

from satquery_agent.image_ingestion import RasterDescriptor, select_visual_bands
from satquery_agent.models import ImageFormat


def descriptor(count, *, colors=None, descriptions=None):
    return RasterDescriptor(
        image_format=ImageFormat.GEOTIFF,
        mime_type="image/tiff",
        width=32,
        height=32,
        band_count=count,
        dtype="uint16",
        nodata=None,
        color_interpretation=colors or [f"band_{index + 1}" for index in range(count)],
        band_descriptions=descriptions or [None] * count,
        warnings=[],
    )


def test_explicit_rgb_color_interpretation_is_respected_in_rgb_order() -> None:
    indices, names, reason = select_visual_bands(descriptor(
        4,
        colors=["alpha", "blue", "red", "green"],
    ))
    assert indices == [2, 3, 1]
    assert names == ["red", "green", "blue"]
    assert "explicitly identifies" in reason


def test_sentinel_like_descriptions_use_b4_b3_b2_not_first_three() -> None:
    indices, names, reason = select_visual_bands(descriptor(
        6,
        descriptions=["B1", "B2", "B3", "B4", "B8", "B11"],
    ))
    assert indices == [3, 2, 1]
    assert names == ["B4", "B3", "B2"]
    assert "Sentinel-2-like" in reason


def test_unknown_multispectral_raster_gets_inspection_only_grayscale() -> None:
    indices, names, reason = select_visual_bands(descriptor(7))
    assert indices == [0]
    assert names == []
    assert "no supported RGB metadata mapping" in reason


def test_three_band_input_retains_declared_order_with_disclosure() -> None:
    indices, names, reason = select_visual_bands(descriptor(3))
    assert indices == [0, 1, 2]
    assert len(names) == 3
    assert "source order" in reason
