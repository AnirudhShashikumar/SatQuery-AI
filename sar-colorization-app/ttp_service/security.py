"""Bounded multipart and image validation for the isolated service."""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = int(os.getenv("TTP_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_WIDTH = int(os.getenv("TTP_MAX_WIDTH", "8192"))
MAX_HEIGHT = int(os.getenv("TTP_MAX_HEIGHT", "8192"))
MAX_PIXELS = int(os.getenv("TTP_MAX_PIXELS", str(36_000_000)))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/x-tiff", "application/octet-stream"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class ValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    extension: str
    width: int
    height: int
    rgb: Image.Image


def validate_request_id(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not REQUEST_ID_PATTERN.fullmatch(value):
        raise ValidationError("INVALID_REQUEST_ID", "request_id may contain only letters, digits, dot, underscore, and hyphen.")
    return value


def _magic_extension(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return ".tiff"
    return None


def validate_image(data: bytes, filename: str | None, content_type: str | None) -> ValidatedImage:
    if not data:
        raise ValidationError("EMPTY_UPLOAD", "The image upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError("UPLOAD_TOO_LARGE", "The image exceeds the configured upload limit.")
    raw_name = filename or "upload"
    if Path(raw_name).name != raw_name or "/" in raw_name or "\\" in raw_name or raw_name in {".", ".."}:
        raise ValidationError("UNSAFE_FILENAME", "The upload filename is invalid.")
    extension = Path(raw_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError("UNSUPPORTED_EXTENSION", "Supported image extensions are PNG, JPEG, TIFF, and GeoTIFF.")
    if (content_type or "application/octet-stream").lower() not in ALLOWED_MIME_TYPES:
        raise ValidationError("UNSUPPORTED_MIME_TYPE", "The upload MIME type is not supported.")
    magic = _magic_extension(data)
    if magic is None:
        raise ValidationError("INVALID_MAGIC_BYTES", "The upload does not contain a supported image signature.")
    if magic == ".png" and extension != ".png":
        raise ValidationError("FORMAT_MISMATCH", "The filename extension does not match the decoded image format.")
    if magic == ".jpg" and extension not in {".jpg", ".jpeg"}:
        raise ValidationError("FORMAT_MISMATCH", "The filename extension does not match the decoded image format.")
    if magic == ".tiff" and extension not in {".tif", ".tiff"}:
        raise ValidationError("FORMAT_MISMATCH", "The filename extension does not match the decoded image format.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width < 1 or height < 1 or width > MAX_WIDTH or height > MAX_HEIGHT or width * height > MAX_PIXELS:
                raise ValidationError("UNSUPPORTED_DIMENSIONS", "The decoded image dimensions exceed the configured safety limit.")
            if source.mode not in {"RGB", "RGBA", "L", "LA"}:
                raise ValidationError("UNSUPPORTED_IMAGE_MODE", "The image must be RGB, RGBA, grayscale, or grayscale with alpha.")
            rgb = source.convert("RGB").copy()
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValidationError("DECODE_FAILED", "The image could not be decoded safely.") from error
    return ValidatedImage(data=data, extension=extension, width=width, height=height, rgb=rgb)


def validate_pair(earlier: ValidatedImage, later: ValidatedImage) -> None:
    if (earlier.width, earlier.height) != (later.width, later.height):
        raise ValidationError("DIMENSION_MISMATCH", "Earlier and later images must have identical pixel dimensions; no resize was performed.")
