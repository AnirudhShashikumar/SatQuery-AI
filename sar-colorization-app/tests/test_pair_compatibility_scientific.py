"""Scientific pair-classification tests independent of endpoint behavior."""

from __future__ import annotations

from satquery_agent.compatibility import validate_pair_compatibility
from satquery_agent.models import (
    ImageFormat,
    ImageMetadata,
    InputMode,
    Modality,
    PairCompatibilityClass,
    RasterBounds,
)


def metadata(
    *,
    width=100,
    height=100,
    crs="EPSG:4326",
    transform=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
    bounds=(0.0, 0.0, 100.0, 100.0),
    nodata=None,
) -> ImageMetadata:
    return ImageMetadata(
        file_id="image",
        original_name="image.tif",
        safe_name="image.tif",
        format=ImageFormat.GEOTIFF if crs else ImageFormat.PNG,
        mime_type="image/tiff" if crs else "image/png",
        size_bytes=10,
        width=width,
        height=height,
        band_count=3,
        dtype="uint16",
        crs=crs,
        transform=list(transform) if transform else None,
        bounds=RasterBounds(left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3]) if bounds else None,
        nodata=nodata,
        is_georeferenced=bool(crs and transform and bounds),
    )


def compare(first: ImageMetadata, second: ImageMetadata):
    return validate_pair_compatibility(
        InputMode.CROSS_MODAL,
        Modality.OPTICAL,
        Modality.SAR,
        first,
        second,
    )


def test_exact_grid_match_records_resolution_orientation_and_nodata() -> None:
    result = compare(metadata(nodata=0), metadata(nodata=0))
    assert result.scientific_classification == PairCompatibilityClass.EXACT_GRID_MATCH
    assert result.primary_resolution == [1.0, 1.0]
    assert result.same_resolution is True
    assert result.same_orientation is True
    assert result.nodata_compatible is True


def test_same_area_different_grid_requires_explicit_resampling() -> None:
    first = metadata()
    second = metadata(width=50, height=50, transform=(2.0, 0.0, 0.0, 0.0, -2.0, 100.0))
    result = compare(first, second)
    assert result.scientific_classification == PairCompatibilityClass.SAME_AREA_DIFFERENT_GRID
    assert result.resampling_required is True
    assert "explicit" in result.recommended_action


def test_different_crs_requires_reprojection() -> None:
    result = compare(metadata(), metadata(crs="EPSG:3857"))
    assert result.scientific_classification == PairCompatibilityClass.REPROJECTION_REQUIRED
    assert not result.compatible


def test_shifted_transform_is_partial_overlap_even_with_same_shape() -> None:
    result = compare(
        metadata(),
        metadata(transform=(1.0, 0.0, 25.0, 0.0, -1.0, 100.0), bounds=(25.0, 0.0, 125.0, 100.0)),
    )
    assert result.same_dimensions is True
    assert result.scientific_classification == PairCompatibilityClass.PARTIAL_OVERLAP
    assert result.alignment_level.value == "geospatial_overlap"


def test_small_and_zero_overlap_are_not_silently_accepted() -> None:
    small = compare(
        metadata(),
        metadata(transform=(1.0, 0.0, 80.0, 0.0, -1.0, 100.0), bounds=(80.0, 0.0, 180.0, 100.0)),
    )
    none = compare(
        metadata(),
        metadata(transform=(1.0, 0.0, 200.0, 0.0, -1.0, 100.0), bounds=(200.0, 0.0, 300.0, 100.0)),
    )
    assert small.scientific_classification == PairCompatibilityClass.INSUFFICIENT_OVERLAP
    assert none.scientific_classification == PairCompatibilityClass.INSUFFICIENT_OVERLAP
    assert none.compatible is False


def test_non_georeferenced_benchmark_pair_is_unverifiable() -> None:
    result = compare(metadata(crs=None, transform=None, bounds=None), metadata(crs=None, transform=None, bounds=None))
    assert result.scientific_classification == PairCompatibilityClass.UNVERIFIABLE
    assert "cannot be independently verified" in " ".join(result.warnings)
    assert result.alignment_level.value == "visual_only"


def test_malformed_affine_metadata_does_not_crash() -> None:
    result = compare(metadata(transform=(1.0, 2.0), bounds=None), metadata(transform=(1.0, 2.0), bounds=None))
    assert result.primary_resolution is None
    assert result.same_orientation is None
    assert result.scientific_classification == PairCompatibilityClass.UNVERIFIABLE
