"""Safe local imagery ingestion and display-preview generation for SatQuery."""

from __future__ import annotations

import io
import hashlib
import math
import os
import re
import tempfile
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from .models import (
    DetectionConfidence,
    ImageFormat,
    ImageMetadata,
    ImageModality,
    RasterBandStatistics,
    RasterBounds,
    RepresentationType,
)

try:
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile

    RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by deployments without Rasterio
    MemoryFile = None
    Resampling = None
    RASTERIO_AVAILABLE = False


def _configured_upload_limit() -> int:
    try:
        megabytes = float(os.getenv("SATQUERY_MAX_UPLOAD_MB", "100"))
    except ValueError:
        megabytes = 100.0
    return max(1, int(megabytes * 1024 * 1024))


MAX_UPLOAD_BYTES = _configured_upload_limit()
MAX_PREVIEW_DIMENSION = 1024
MAX_PREVIEW_SOURCE_ELEMENTS = 180_000_000
READ_CHUNK_BYTES = 1024 * 1024
DISPLAY_PREVIEW_WARNING = "Display preview only — not a scientific product."
PREVIEW_DIR = Path(
    os.getenv(
        "SATQUERY_PREVIEW_DIR",
        str(Path(tempfile.gettempdir()) / "geovision-satquery-previews"),
    )
).resolve()
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
EXPECTED_MIME_TYPES = {
    ImageFormat.TIFF: {"image/tiff", "image/x-tiff"},
    ImageFormat.GEOTIFF: {"image/tiff", "image/x-tiff", "image/geotiff"},
    ImageFormat.PNG: {"image/png"},
    ImageFormat.JPEG: {"image/jpeg", "image/jpg", "image/pjpeg"},
}


class ImageIngestionError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class RasterDescriptor:
    image_format: ImageFormat
    mime_type: str
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: Optional[float]
    color_interpretation: List[str]
    band_descriptions: List[Optional[str]]
    warnings: List[str]
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds: Optional[RasterBounds] = None
    is_georeferenced: bool = False


@dataclass
class IngestedImage:
    metadata: ImageMetadata
    durations_ms: Dict[str, int]
    model_image: Image.Image
    image_representation: str
    bands_used: List[str]
    analysis_raster: np.ndarray
    content_hash: str
    source_bytes: bytes


def sanitize_filename(filename: str) -> Tuple[str, str, str]:
    normalized = unicodedata.normalize("NFKC", filename or "upload")
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(character for character in basename if character.isprintable()).strip()
    if not basename or basename in {".", ".."}:
        basename = "upload"
    extension = Path(basename).suffix.lower()
    stem = Path(basename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "upload"
    safe_name = (safe_stem[:100] + extension)[:120]
    return basename[:255], safe_name, extension


async def _read_bounded(upload: UploadFile) -> bytes:
    chunks: List[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise ImageIngestionError(
                "FILE_TOO_LARGE",
                "Each image must be {} MB or smaller.".format(MAX_UPLOAD_BYTES // (1024 * 1024)),
                413,
            )
        chunks.append(chunk)
    if size == 0:
        raise ImageIngestionError("EMPTY_FILE", "The uploaded image is empty.")
    return b"".join(chunks)


def _detect_container(data: bytes) -> ImageFormat:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ImageFormat.PNG
    if data.startswith(b"\xff\xd8\xff"):
        return ImageFormat.JPEG
    if data[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        return ImageFormat.TIFF
    raise ImageIngestionError(
        "INVALID_IMAGE_BYTES",
        "The upload is not a valid PNG, JPEG, TIFF, or GeoTIFF image.",
    )


def _validate_type(extension: str, supplied_mime: str, detected: ImageFormat) -> None:
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageIngestionError(
            "UNSUPPORTED_EXTENSION",
            "Supported file extensions are .tif, .tiff, .png, .jpg, and .jpeg.",
        )
    expected_extension_format = (
        ImageFormat.TIFF if extension in {".tif", ".tiff"}
        else ImageFormat.PNG if extension == ".png"
        else ImageFormat.JPEG
    )
    if detected != expected_extension_format:
        raise ImageIngestionError(
            "FILE_TYPE_MISMATCH",
            "The file contents do not match the filename extension.",
        )
    normalized_mime = (supplied_mime or "").split(";", 1)[0].strip().lower()
    if normalized_mime not in GENERIC_MIME_TYPES and normalized_mime not in EXPECTED_MIME_TYPES[detected]:
        raise ImageIngestionError(
            "UNSUPPORTED_MIME_TYPE",
            "The supplied MIME type does not match the detected image format.",
        )


def _series_dimensions(series: tifffile.TiffPageSeries) -> Tuple[int, int, int]:
    shape = tuple(int(value) for value in series.shape)
    axes = series.axes.upper()
    if "Y" in axes and "X" in axes:
        height = shape[axes.index("Y")]
        width = shape[axes.index("X")]
        other = [shape[index] for index, axis in enumerate(axes) if axis not in {"Y", "X"}]
        bands = int(np.prod(other)) if other else int(getattr(series.pages[0], "samplesperpixel", 1) or 1)
    elif len(shape) >= 2:
        height, width = shape[-2], shape[-1]
        bands = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    else:
        raise ImageIngestionError("MISSING_RASTER_BANDS", "The TIFF has no readable raster dimensions.")
    if width <= 0 or height <= 0 or bands <= 0:
        raise ImageIngestionError("MISSING_RASTER_BANDS", "The TIFF has no readable raster bands.")
    return width, height, bands


def _tiff_color_interpretation(page: tifffile.TiffPage, bands: int) -> List[str]:
    photometric = getattr(getattr(page, "photometric", None), "name", "unknown").lower()
    if photometric == "rgb" and bands >= 3:
        values = ["red", "green", "blue"]
        values.extend("alpha" if index == 3 else "band_{}".format(index + 1) for index in range(3, bands))
        return values
    if bands == 1:
        return ["gray"]
    return ["band_{}".format(index + 1) for index in range(bands)]


def _parse_nodata(page: tifffile.TiffPage) -> Optional[float]:
    tag = page.tags.get(42113)
    if tag is None:
        return None
    try:
        value = float(str(tag.value).strip().strip("\x00"))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _inspect_tiff(data: bytes) -> RasterDescriptor:
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as dataset:
            if not dataset.series:
                raise ImageIngestionError("CORRUPT_TIFF", "The TIFF could not be read safely.")
            series = dataset.series[0]
            width, height, bands = _series_dimensions(series)
            page = series.pages[0]
            geotiff_detected = bool(getattr(page, "is_geotiff", False) or page.tags.get(34735))
            image_format = ImageFormat.GEOTIFF if geotiff_detected else ImageFormat.TIFF
            warnings = [DISPLAY_PREVIEW_WARNING]
            if geotiff_detected and not RASTERIO_AVAILABLE:
                warnings.append(
                    "GeoTIFF tags were detected, but rasterio is unavailable; CRS, transform, and bounds were not extracted."
                )
            descriptor = RasterDescriptor(
                image_format=image_format,
                mime_type="image/tiff",
                width=width,
                height=height,
                band_count=bands,
                dtype=str(np.dtype(series.dtype)),
                nodata=_parse_nodata(page),
                color_interpretation=_tiff_color_interpretation(page, bands),
                band_descriptions=[None] * bands,
                warnings=warnings,
            )
        return _enrich_tiff_with_rasterio(data, descriptor)
    except ImageIngestionError:
        raise
    except (tifffile.TiffFileError, ValueError, TypeError, IndexError, OSError) as error:
        raise ImageIngestionError("CORRUPT_TIFF", "The TIFF could not be read safely.") from error


def _enrich_tiff_with_rasterio(data: bytes, descriptor: RasterDescriptor) -> RasterDescriptor:
    if not RASTERIO_AVAILABLE or MemoryFile is None:
        return descriptor
    try:
        with MemoryFile(data) as memory_file:
            with memory_file.open() as dataset:
                crs = dataset.crs.to_string() if dataset.crs is not None else None
                transform = dataset.transform
                meaningful_transform = bool(crs) and not bool(getattr(transform, "is_identity", False))
                transform_values = [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)] if meaningful_transform else None
                bounds = None
                if meaningful_transform:
                    bounds = RasterBounds(
                        left=float(dataset.bounds.left),
                        bottom=float(dataset.bounds.bottom),
                        right=float(dataset.bounds.right),
                        top=float(dataset.bounds.top),
                    )
                nodata = dataset.nodata
                nodata_value = float(nodata) if nodata is not None and math.isfinite(float(nodata)) else descriptor.nodata
                interpretations = [item.name.lower() for item in dataset.colorinterp]
                if len(interpretations) != descriptor.band_count or all(value in {"undefined", "gray"} for value in interpretations):
                    interpretations = descriptor.color_interpretation
                warnings = list(descriptor.warnings)
                descriptions = [value.strip() if value and value.strip() else None for value in dataset.descriptions]
                if len(descriptions) < int(dataset.count):
                    descriptions.extend([None] * (int(dataset.count) - len(descriptions)))
                if crs and not meaningful_transform:
                    warnings.append("CRS tags were detected, but no usable affine transform or geospatial bounds were available.")
                return RasterDescriptor(
                    image_format=ImageFormat.GEOTIFF if crs or descriptor.image_format == ImageFormat.GEOTIFF else ImageFormat.TIFF,
                    mime_type=descriptor.mime_type,
                    width=int(dataset.width),
                    height=int(dataset.height),
                    band_count=int(dataset.count) if int(dataset.count) >= descriptor.band_count else descriptor.band_count,
                    dtype=str(dataset.dtypes[0]) if dataset.dtypes else descriptor.dtype,
                    nodata=nodata_value,
                    color_interpretation=interpretations,
                    band_descriptions=descriptions,
                    warnings=warnings,
                    crs=crs,
                    transform=transform_values,
                    bounds=bounds,
                    is_georeferenced=bool(crs and transform_values and bounds),
                )
    except Exception:
        descriptor.warnings.append("Rasterio metadata extraction was unavailable for this TIFF; tifffile metadata was retained.")
        return descriptor


def _pillow_dtype(mode: str) -> str:
    if mode.startswith("I;16"):
        return "uint16"
    if mode == "I":
        return "int32"
    if mode == "F":
        return "float32"
    return "uint8"


def _inspect_standard_image(data: bytes, detected: ImageFormat) -> RasterDescriptor:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            bands = len(image.getbands())
            if width <= 0 or height <= 0 or bands <= 0:
                raise ImageIngestionError("MISSING_RASTER_BANDS", "The image has no readable raster bands.")
            interpretation = [value.lower() for value in image.getbands()]
            return RasterDescriptor(
                image_format=detected,
                mime_type="image/png" if detected == ImageFormat.PNG else "image/jpeg",
                width=width,
                height=height,
                band_count=bands,
                dtype=_pillow_dtype(image.mode),
                nodata=None,
                color_interpretation=interpretation,
                band_descriptions=[value.lower() for value in image.getbands()],
                warnings=[DISPLAY_PREVIEW_WARNING],
            )
    except ImageIngestionError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise ImageIngestionError("INVALID_IMAGE_BYTES", "The image could not be decoded safely.") from error


def _normalize_to_hwc(array: np.ndarray, axes: str) -> np.ndarray:
    data = np.asarray(array)
    axes = axes.upper()
    while data.ndim > len(axes):
        data = np.squeeze(data, axis=0)
    if "Y" in axes and "X" in axes and data.ndim == len(axes):
        y_index, x_index = axes.index("Y"), axes.index("X")
        order = [y_index, x_index] + [index for index in range(data.ndim) if index not in {y_index, x_index}]
        data = np.transpose(data, order)
        return data.reshape(data.shape[0], data.shape[1], -1)
    if data.ndim == 2:
        return data[:, :, None]
    if data.ndim == 3 and data.shape[-1] <= 16:
        return data
    if data.ndim == 3 and data.shape[0] <= 16:
        return np.moveaxis(data, 0, -1)
    raise ImageIngestionError("PREVIEW_FAILURE", "The raster band layout is unsupported for preview generation.")


def _choose_tiff_level(series: tifffile.TiffPageSeries) -> tifffile.TiffPageSeries:
    levels = list(getattr(series, "levels", []) or [series])
    for level in levels:
        try:
            width, height, _ = _series_dimensions(level)
        except ImageIngestionError:
            continue
        if max(width, height) <= MAX_PREVIEW_DIMENSION * 2:
            return level
    return levels[-1]


def _read_tiff_preview(data: bytes, descriptor: RasterDescriptor) -> np.ndarray:
    rasterio_preview = _read_tiff_preview_rasterio(data, descriptor)
    if rasterio_preview is not None:
        return rasterio_preview
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as dataset:
            series = _choose_tiff_level(dataset.series[0])
            pages = list(series.pages)
            if descriptor.band_count > 1 and len(pages) >= descriptor.band_count and all(page.ndim == 2 for page in pages[: min(3, descriptor.band_count)]):
                selected_pages = pages[: min(3, descriptor.band_count)]
                element_count = sum(int(np.prod(page.shape)) for page in selected_pages)
                if element_count > MAX_PREVIEW_SOURCE_ELEMENTS:
                    raise ImageIngestionError("PREVIEW_FAILURE", "The raster is too large to preview safely without reduced-resolution levels.")
                raster = np.stack([page.asarray() for page in selected_pages], axis=0)
                axes = "CYX"
            else:
                if int(np.prod(series.shape)) > MAX_PREVIEW_SOURCE_ELEMENTS:
                    raise ImageIngestionError("PREVIEW_FAILURE", "The raster is too large to preview safely without reduced-resolution levels.")
                raster = series.asarray()
                axes = series.axes
            return _normalize_to_hwc(raster, axes)
    except ImageIngestionError:
        raise
    except (tifffile.TiffFileError, ValueError, TypeError, IndexError, OSError) as error:
        raise ImageIngestionError("PREVIEW_FAILURE", "A browser preview could not be generated for this TIFF.") from error


def _read_tiff_preview_rasterio(data: bytes, descriptor: RasterDescriptor) -> Optional[np.ndarray]:
    if not RASTERIO_AVAILABLE or MemoryFile is None or Resampling is None:
        return None
    try:
        with MemoryFile(data) as memory_file:
            with memory_file.open() as dataset:
                selected_count = min(3, descriptor.band_count)
                if dataset.count < selected_count:
                    return None
                scale = min(1.0, (MAX_PREVIEW_DIMENSION * 2) / max(dataset.width, dataset.height))
                output_width = max(1, round(dataset.width * scale))
                output_height = max(1, round(dataset.height * scale))
                raster = dataset.read(
                    indexes=list(range(1, selected_count + 1)),
                    out_shape=(selected_count, output_height, output_width),
                    resampling=Resampling.bilinear,
                    masked=False,
                )
                values = np.asarray(raster)
                return _normalize_to_hwc(values, "CYX")
    except Exception:
        return None


def _read_standard_preview(data: bytes, descriptor: RasterDescriptor) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if descriptor.image_format == ImageFormat.JPEG:
                image.draft("RGB", (MAX_PREVIEW_DIMENSION, MAX_PREVIEW_DIMENSION))
            image.thumbnail((MAX_PREVIEW_DIMENSION, MAX_PREVIEW_DIMENSION), Image.Resampling.LANCZOS)
            if image.mode == "P":
                image = image.convert("RGB")
            return _normalize_to_hwc(np.asarray(image), "YXS" if len(image.getbands()) > 1 else "YX")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise ImageIngestionError("PREVIEW_FAILURE", "A browser preview could not be generated for this image.") from error


def _display_stretch(band: np.ndarray, nodata: Optional[float], warnings: List[str]) -> np.ndarray:
    values = np.asarray(band, dtype=np.float64)
    finite = np.isfinite(values)
    if nodata is not None:
        finite &= ~np.isclose(values, nodata, equal_nan=False)
    if not finite.any():
        raise ImageIngestionError("NON_FINITE_RASTER", "The selected preview bands contain no finite raster values.")
    if not finite.all():
        warnings.append("Non-finite or NoData pixels were excluded from the display stretch.")
    valid = values[finite]
    low, high = np.percentile(valid, [2.0, 98.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ImageIngestionError("NON_FINITE_RASTER", "The selected preview bands contain invalid numeric ranges.")
    if high <= low:
        low, high = float(valid.min()), float(valid.max())
    if high <= low:
        warnings.append("A constant-value band was rendered as mid-gray in the display preview.")
        output = np.full(values.shape, 127, dtype=np.uint8)
        output[~finite] = 0
        return output
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def _render_preview(raster: np.ndarray, descriptor: RasterDescriptor) -> Tuple[Image.Image, List[str], List[str], str]:
    warnings: List[str] = []
    if raster.ndim != 3 or raster.shape[2] < 1:
        raise ImageIngestionError("MISSING_RASTER_BANDS", "No previewable raster bands were found.")
    stride = max(1, int(math.ceil(max(raster.shape[:2]) / MAX_PREVIEW_DIMENSION)))
    sampled = raster[::stride, ::stride, :]
    if descriptor.band_count == 1 or sampled.shape[2] == 1:
        stretched = _display_stretch(sampled[:, :, 0], descriptor.nodata, warnings)
        preview = Image.fromarray(stretched)
        bands_used = [descriptor.color_interpretation[0] if descriptor.color_interpretation else "band_1"]
        representation = "percentile-stretched grayscale display representation"
    elif descriptor.band_count == 2 or sampled.shape[2] == 2:
        red = _display_stretch(sampled[:, :, 0], descriptor.nodata, warnings)
        green = _display_stretch(sampled[:, :, 1], descriptor.nodata, warnings)
        blue = np.round((red.astype(np.float32) + green.astype(np.float32)) / 2.0).astype(np.uint8)
        preview = Image.fromarray(np.stack([red, green, blue], axis=-1))
        warnings.append("Two-band false-color preview: R=band 1, G=band 2, B=average of bands 1 and 2.")
        bands_used = (descriptor.color_interpretation[:2] or ["band_1", "band_2"])
        representation = "two-band false-color radar display representation"
    else:
        rgb = np.stack(
            [_display_stretch(sampled[:, :, index], descriptor.nodata, warnings) for index in range(3)],
            axis=-1,
        )
        preview = Image.fromarray(rgb)
        bands_used = descriptor.color_interpretation[:3] or ["band_1", "band_2", "band_3"]
        representation = "percentile-stretched RGB display representation"
        if descriptor.band_count > 3:
            warnings.append("The display preview uses the first three bands; additional bands remain unchanged in the source file.")
    preview.thumbnail((MAX_PREVIEW_DIMENSION, MAX_PREVIEW_DIMENSION), Image.Resampling.LANCZOS)
    return preview, list(dict.fromkeys(warnings)), bands_used, representation


def _safe_number(value: float) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def _band_statistics(
    raster: np.ndarray,
    descriptor: RasterDescriptor,
) -> List[RasterBandStatistics]:
    output: List[RasterBandStatistics] = []
    for index in range(raster.shape[2]):
        values = np.asarray(raster[:, :, index], dtype=np.float64)
        nan_mask = np.isnan(values)
        inf_mask = np.isinf(values)
        nodata_mask = np.zeros(values.shape, dtype=bool)
        if descriptor.nodata is not None:
            nodata_mask = np.isclose(values, descriptor.nodata, equal_nan=False)
        valid_mask = np.isfinite(values) & ~nodata_mask
        valid = values[valid_mask]
        percentiles = np.percentile(valid, [1, 5, 50, 95, 99]) if valid.size else [math.nan] * 5
        output.append(RasterBandStatistics(
            band=index + 1,
            description=descriptor.band_descriptions[index] if index < len(descriptor.band_descriptions) else None,
            dtype=str(raster[:, :, index].dtype),
            minimum=_safe_number(valid.min()) if valid.size else None,
            maximum=_safe_number(valid.max()) if valid.size else None,
            mean=_safe_number(valid.mean()) if valid.size else None,
            standard_deviation=_safe_number(valid.std()) if valid.size else None,
            percentile_1=_safe_number(percentiles[0]),
            percentile_5=_safe_number(percentiles[1]),
            percentile_50=_safe_number(percentiles[2]),
            percentile_95=_safe_number(percentiles[3]),
            percentile_99=_safe_number(percentiles[4]),
            nan_count=int(nan_mask.sum()),
            inf_count=int(inf_mask.sum()),
            nodata_count=int(nodata_mask.sum()),
            valid_count=int(valid.size),
        ))
    return output


def _detect_representation(descriptor: RasterDescriptor) -> RepresentationType:
    if descriptor.image_format in {ImageFormat.TIFF, ImageFormat.GEOTIFF}:
        return RepresentationType.SCIENTIFIC_RASTER
    if descriptor.image_format in {ImageFormat.PNG, ImageFormat.JPEG}:
        return RepresentationType.DISPLAY_PREVIEW
    return RepresentationType.UNKNOWN_REPRESENTATION


def _detect_modality(
    descriptor: RasterDescriptor,
    representation: RepresentationType,
    original_name: str,
) -> Tuple[ImageModality, DetectionConfidence, str, List[str]]:
    descriptions = " ".join(value or "" for value in descriptor.band_descriptions).lower()
    filename = original_name.lower()
    interpretations = {value.lower() for value in descriptor.color_interpretation}
    sar_hint = any(token in descriptions or token in filename for token in ("sar", "sentinel-1", "sentinel_1", "radar"))
    has_vv = bool(re.search(r"(^|[^a-z])vv([^a-z]|$)", descriptions))
    has_vh = bool(re.search(r"(^|[^a-z])vh([^a-z]|$)", descriptions))
    limitations: List[str] = []

    if representation == RepresentationType.DISPLAY_PREVIEW and descriptor.band_count == 1:
        reason = "A one-band PNG/JPEG is a grayscale display representation; its scientific modality is not encoded."
        if sar_hint:
            reason += " The filename suggests SAR, but that hint is not authoritative."
        limitations.append("Confirm whether this grayscale display is SAR, optical grayscale, panchromatic, or another representation before analysis.")
        return ImageModality.UNKNOWN, DetectionConfidence.LOW, reason, limitations
    rgb_like = descriptor.band_count in {3, 4} and ({"r", "g", "b"}.issubset(interpretations) or {"red", "green", "blue"}.issubset(interpretations))
    if representation == RepresentationType.DISPLAY_PREVIEW and rgb_like:
        return ImageModality.OPTICAL_RGB, DetectionConfidence.HIGH, "The display image contains explicit RGB channels.", limitations
    if representation == RepresentationType.DISPLAY_PREVIEW and descriptor.band_count in {3, 4}:
        return ImageModality.OPTICAL_RGB, DetectionConfidence.MEDIUM, "The display image has three RGB-like color channels.", limitations
    if representation == RepresentationType.SCIENTIFIC_RASTER:
        if descriptor.band_count == 2 and has_vv and has_vh:
            return ImageModality.SAR_VV_VH, DetectionConfidence.HIGH, "Band descriptions identify co-polarized VV and cross-polarized VH SAR channels.", limitations
        if descriptor.band_count == 1 and has_vv:
            return ImageModality.SAR_VV, DetectionConfidence.HIGH, "The scientific raster band description identifies VV SAR data.", limitations
        if descriptor.band_count == 1 and has_vh:
            return ImageModality.SAR_VH, DetectionConfidence.HIGH, "The scientific raster band description identifies VH SAR data.", limitations
        if sar_hint and descriptor.band_count == 2:
            limitations.append("SAR channel order could not be verified from band descriptions.")
            return ImageModality.SAR_VV_VH, DetectionConfidence.LOW, "SAR filename/metadata hints and two scientific bands suggest a dual-polarization raster.", limitations
        if sar_hint and descriptor.band_count == 1:
            limitations.append("SAR polarization and value domain could not be verified.")
            return ImageModality.SAR_VV, DetectionConfidence.LOW, "SAR filename/metadata hints and one scientific band suggest a SAR raster.", limitations
        if descriptor.band_count > 3:
            limitations.append("Select a band mapping before using RGB-only specialists.")
            return ImageModality.MULTISPECTRAL, DetectionConfidence.MEDIUM, "The scientific raster contains more than three bands.", limitations
        if rgb_like:
            return ImageModality.OPTICAL_RGB, DetectionConfidence.MEDIUM, "Color interpretation identifies RGB-like scientific bands.", limitations
    return ImageModality.UNKNOWN, DetectionConfidence.LOW, "Available metadata does not establish an authoritative modality.", ["User modality confirmation is required for modality-specific analysis."]


def apply_modality_override(metadata: ImageMetadata, override: ImageModality) -> ImageMetadata:
    if override == ImageModality.AUTO:
        return metadata.model_copy(update={"user_confirmed_modality": None, "effective_modality": metadata.auto_detected_modality})
    return metadata.model_copy(update={"user_confirmed_modality": override, "effective_modality": override})


def coarse_modality(modality: ImageModality) -> Modality:
    from .models import Modality
    if modality in {ImageModality.SAR_PREVIEW, ImageModality.SAR_VV, ImageModality.SAR_VH, ImageModality.SAR_VV_VH}:
        return Modality.SAR
    if modality == ImageModality.MULTISPECTRAL:
        return Modality.MULTISPECTRAL
    if modality in {ImageModality.OPTICAL_RGB, ImageModality.OPTICAL_GRAYSCALE, ImageModality.PANCHROMATIC}:
        return Modality.OPTICAL
    return Modality.UNKNOWN


def save_preview(preview: Image.Image) -> str:
    preview_name = "{}.png".format(uuid.uuid4().hex)
    destination = PREVIEW_DIR / preview_name
    try:
        preview.save(destination, format="PNG", optimize=True)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise ImageIngestionError("PREVIEW_FAILURE", "The browser preview could not be stored safely.") from error
    return "/api/agent/previews/{}".format(preview_name)


def remove_preview_url(preview_url: Optional[str]) -> None:
    if not preview_url:
        return
    path = preview_file_path(preview_url.rsplit("/", 1)[-1])
    if path is not None:
        path.unlink(missing_ok=True)


def preview_file_path(preview_name: str) -> Optional[Path]:
    if not re.fullmatch(r"[0-9a-f]{32}\.png", preview_name):
        return None
    candidate = (PREVIEW_DIR / preview_name).resolve()
    if candidate.parent != PREVIEW_DIR or not candidate.is_file():
        return None
    return candidate


def remove_preview(metadata: Optional[ImageMetadata]) -> None:
    if metadata is None or not metadata.preview_url:
        return
    remove_preview_url(metadata.preview_url)


async def ingest_upload(upload: UploadFile) -> IngestedImage:
    original_name, safe_name, extension = sanitize_filename(upload.filename or "upload")
    durations: Dict[str, int] = {}
    preview_url: Optional[str] = None
    try:
        started = time.perf_counter()
        data = await _read_bounded(upload)
        durations["upload_received"] = max(0, round((time.perf_counter() - started) * 1000))

        started = time.perf_counter()
        if extension not in ALLOWED_EXTENSIONS:
            raise ImageIngestionError(
                "UNSUPPORTED_EXTENSION",
                "Supported file extensions are .tif, .tiff, .png, .jpg, and .jpeg.",
            )
        detected = _detect_container(data)
        _validate_type(extension, upload.content_type or "", detected)
        durations["file_type_validation"] = max(0, round((time.perf_counter() - started) * 1000))

        started = time.perf_counter()
        descriptor = _inspect_tiff(data) if detected == ImageFormat.TIFF else _inspect_standard_image(data, detected)
        durations["metadata_extraction"] = max(0, round((time.perf_counter() - started) * 1000))

        started = time.perf_counter()
        raster = _read_tiff_preview(data, descriptor) if detected == ImageFormat.TIFF else _read_standard_preview(data, descriptor)
        representation = _detect_representation(descriptor)
        detected_modality, detection_confidence, detection_reason, modality_limitations = _detect_modality(
            descriptor, representation, original_name
        )
        statistics = _band_statistics(raster, descriptor)
        preview, preview_warnings, bands_used, image_representation = _render_preview(raster, descriptor)
        analysis_stride = max(1, int(math.ceil(max(raster.shape[:2]) / MAX_PREVIEW_DIMENSION)))
        analysis_raster = np.ascontiguousarray(raster[::analysis_stride, ::analysis_stride, : min(3, raster.shape[2])])
        model_image = preview.convert("RGB").copy()
        preview_url = save_preview(preview)
        durations["preview_generation"] = max(0, round((time.perf_counter() - started) * 1000))

        metadata = ImageMetadata(
            file_id=str(uuid.uuid4()),
            original_name=original_name,
            safe_name=safe_name,
            format=descriptor.image_format,
            mime_type=descriptor.mime_type,
            size_bytes=len(data),
            width=descriptor.width,
            height=descriptor.height,
            band_count=descriptor.band_count,
            dtype=descriptor.dtype,
            crs=descriptor.crs,
            transform=descriptor.transform,
            bounds=descriptor.bounds,
            nodata=descriptor.nodata,
            is_georeferenced=descriptor.is_georeferenced,
            preview_url=preview_url,
            color_interpretation=descriptor.color_interpretation,
            band_descriptions=descriptor.band_descriptions,
            band_statistics=statistics,
            representation=representation,
            auto_detected_modality=detected_modality,
            auto_detection_confidence=detection_confidence,
            auto_detection_reason=detection_reason,
            effective_modality=detected_modality,
            modality_limitations=modality_limitations,
            warnings=list(dict.fromkeys(descriptor.warnings + preview_warnings)),
        )
        return IngestedImage(
            metadata=metadata,
            durations_ms=durations,
            model_image=model_image,
            image_representation=image_representation,
            bands_used=bands_used,
            analysis_raster=analysis_raster,
            content_hash=hashlib.sha256(data).hexdigest(),
            source_bytes=data,
        )
    except Exception:
        if preview_url:
            path = preview_file_path(preview_url.rsplit("/", 1)[-1])
            if path is not None:
                path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
