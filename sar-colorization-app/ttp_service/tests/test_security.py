from __future__ import annotations

import io

import pytest
from PIL import Image

from ttp_service.security import ValidationError, validate_image, validate_pair, validate_request_id


def png(width: int = 8, height: int = 8, value: int = 40) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(output, "PNG")
    return output.getvalue()


def test_validates_png_and_pair_without_resizing() -> None:
    earlier = validate_image(png(), "earlier.png", "image/png")
    later = validate_image(png(value=90), "later.png", "image/png")
    validate_pair(earlier, later)
    assert earlier.rgb.mode == "RGB"


def test_rejects_path_traversal_and_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="filename"):
        validate_image(png(), "../earlier.png", "image/png")
    earlier = validate_image(png(), "earlier.png", "image/png")
    later = validate_image(png(9, 8), "later.png", "image/png")
    with pytest.raises(ValidationError, match="identical"):
        validate_pair(earlier, later)


def test_request_id_is_bounded_and_opaque() -> None:
    assert validate_request_id("request-123") == "request-123"
    with pytest.raises(ValidationError):
        validate_request_id("../../secret")
