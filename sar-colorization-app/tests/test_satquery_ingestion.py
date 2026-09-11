"""Compact ingestion, preview, compatibility, and multipart contract tests."""

from __future__ import annotations

import io
import unittest

import numpy as np
import rasterio
import tifffile
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent import image_ingestion
from satquery_agent.compatibility import validate_pair_compatibility
from satquery_agent.models import ImageMetadata, InputMode, Modality


def pillow_bytes(image_format: str, mode: str = "RGB", size=(18, 12)) -> bytes:
    buffer = io.BytesIO()
    if mode == "RGB":
        values = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = Image.fromarray(values)
    else:
        image = Image.new(mode, size, color=80)
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def tiff_bytes(array: np.ndarray, **kwargs) -> bytes:
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, **kwargs)
    return buffer.getvalue()


def geotiff_bytes(array: np.ndarray, *, crs="EPSG:4326", transform=None, nodata=-9999) -> bytes:
    values = array if array.ndim == 3 else array[None, :, :]
    transform = transform or from_origin(70.0, 20.0, 0.01, 0.01)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            dtype=str(values.dtype),
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(values)
        return memory_file.read()


def metadata(
    width=32,
    height=24,
    crs=None,
    transform=None,
    bounds=None,
) -> ImageMetadata:
    return ImageMetadata(
        file_id="fixture",
        original_name="fixture.tif",
        safe_name="fixture.tif",
        format="geotiff" if crs else "tiff",
        mime_type="image/tiff",
        size_bytes=100,
        width=width,
        height=height,
        band_count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
        bounds=bounds,
        nodata=None,
        is_georeferenced=bool(crs and transform and bounds),
        preview_url=None,
        color_interpretation=["gray"],
        warnings=[],
    )


class SatQueryIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def submit(self, primary, *, secondary=None, **fields):
        data = {
            "query": "Process this satellite input",
            "input_mode": "single",
            "primary_modality": "optical",
        }
        data.update(fields)
        files = {"primary_image": primary}
        if secondary is not None:
            files["secondary_image"] = secondary
        return self.client.post("/api/agent/query", data=data, files=files)

    def test_valid_png(self):
        response = self.submit(("scene.png", pillow_bytes("PNG"), "image/png"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual((item["format"], item["width"], item["height"], item["band_count"]), ("png", 18, 12, 3))
        preview = self.client.get(item["preview_url"])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_valid_jpeg(self):
        response = self.submit(("scene.jpg", pillow_bytes("JPEG"), "image/jpeg"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["primary_image_metadata"]["format"], "jpeg")

    def test_valid_single_band_tiff(self):
        source = np.arange(20 * 12, dtype=np.uint16).reshape(12, 20)
        response = self.submit(("single.tif", tiff_bytes(source), "image/tiff"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual((item["band_count"], item["dtype"]), (1, "uint16"))

    def test_valid_three_band_tiff(self):
        source = np.zeros((10, 14, 3), dtype=np.uint8)
        source[:, :, 0], source[:, :, 1], source[:, :, 2] = 30, 100, 220
        response = self.submit(("rgb.tiff", tiff_bytes(source, photometric="rgb"), "image/tiff"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual(item["band_count"], 3)
        self.assertEqual(item["color_interpretation"][:3], ["red", "green", "blue"])

    def test_valid_two_band_sar_tiff(self):
        source = np.stack([
            np.arange(96, dtype=np.float32).reshape(8, 12),
            np.arange(96, dtype=np.float32).reshape(8, 12)[::-1],
        ])
        response = self.submit(("sar.tif", tiff_bytes(source, metadata={"axes": "CYX"}), "image/tiff"), primary_modality="sar")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual(item["band_count"], 2)
        self.assertIn("Two-band false-color preview", " ".join(item["warnings"]))

    def test_unresolved_geotiff_tags_are_not_fabricated(self):
        source = np.arange(64, dtype=np.uint16).reshape(8, 8)
        geokey = (1, 1, 0, 1, 1024, 0, 1, 1)
        payload = tiff_bytes(source, extratags=[(34735, "H", len(geokey), geokey, False)])
        response = self.submit(("tagged.tif", payload, "image/tiff"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual(item["format"], "geotiff")
        self.assertIsNotNone(item["crs"])
        self.assertIsNone(item["transform"])
        self.assertIsNone(item["bounds"])
        self.assertFalse(item["is_georeferenced"])

    def test_rasterio_extracts_geotiff_metadata_and_reduced_preview(self):
        source = np.arange(1024 * 2048, dtype=np.float32).reshape(1024, 2048)
        payload = geotiff_bytes(source, nodata=-9999.0)
        response = self.submit(("mapped.tif", payload, "image/tiff"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual(item["format"], "geotiff")
        self.assertEqual(item["crs"], "EPSG:4326")
        self.assertEqual(len(item["transform"]), 6)
        self.assertAlmostEqual(item["bounds"]["left"], 70.0)
        self.assertAlmostEqual(item["bounds"]["top"], 20.0)
        self.assertEqual(item["nodata"], -9999.0)
        self.assertTrue(item["is_georeferenced"])
        preview = Image.open(io.BytesIO(self.client.get(item["preview_url"]).content))
        self.assertLessEqual(max(preview.size), 1024)

    def test_actual_aligned_geotiff_pair_is_exact(self):
        payload = geotiff_bytes(np.ones((3, 16, 24), dtype=np.uint8), nodata=0)
        response = self.submit(
            ("optical.tif", payload, "image/tiff"),
            secondary=("sar.tif", geotiff_bytes(np.ones((2, 16, 24), dtype=np.uint8), nodata=0), "image/tiff"),
            query="Use optical and SAR together",
            input_mode="cross_modal",
            primary_modality="optical",
            secondary_modality="sar",
        )
        self.assertEqual(response.status_code, 200, response.text)
        pair = response.json()["pair_compatibility"]
        self.assertTrue(pair["compatible"])
        self.assertEqual(pair["alignment_level"], "exact")

    def test_corrupt_tiff(self):
        response = self.submit(("broken.tif", b"II*\x00not-a-tiff", "image/tiff"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "CORRUPT_TIFF")

    def test_empty_file(self):
        response = self.submit(("empty.png", b"", "image/png"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_FILE")

    def test_unsupported_extension(self):
        response = self.submit(("image.svg", pillow_bytes("PNG"), "image/png"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "UNSUPPORTED_EXTENSION")

    def test_oversized_file_behavior(self):
        original = image_ingestion.MAX_UPLOAD_BYTES
        image_ingestion.MAX_UPLOAD_BYTES = 16
        try:
            response = self.submit(("large.png", pillow_bytes("PNG"), "image/png"))
        finally:
            image_ingestion.MAX_UPLOAD_BYTES = original
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "FILE_TOO_LARGE")

    def test_filename_path_traversal_attempt(self):
        response = self.submit(("../../private/scene.png", pillow_bytes("PNG"), "image/png"))
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["primary_image_metadata"]
        self.assertEqual(item["original_name"], "scene.png")
        self.assertEqual(item["safe_name"], "scene.png")
        self.assertNotIn("/", item["safe_name"])

    def test_exact_matching_pair(self):
        first = metadata(crs="EPSG:4326", transform=[1, 0, 0, 0, -1, 24], bounds={"left": 0, "bottom": 0, "right": 32, "top": 24})
        result = validate_pair_compatibility(InputMode.CROSS_MODAL, Modality.OPTICAL, Modality.SAR, first, first.model_copy())
        self.assertTrue(result.compatible)
        self.assertEqual(result.alignment_level.value, "exact")
        self.assertFalse(result.resampling_required)

    def test_dimension_mismatch(self):
        result = validate_pair_compatibility(InputMode.CROSS_MODAL, Modality.OPTICAL, Modality.SAR, metadata(width=20, height=10), metadata(width=40, height=20))
        self.assertTrue(result.compatible)
        self.assertEqual(result.alignment_level.value, "visual_only")
        self.assertFalse(result.same_dimensions)
        self.assertTrue(result.resampling_required)

    def test_crs_mismatch(self):
        left = metadata(crs="EPSG:4326", transform=[1, 0, 0, 0, -1, 24], bounds={"left": 0, "bottom": 0, "right": 32, "top": 24})
        right = metadata(crs="EPSG:3857", transform=[1, 0, 0, 0, -1, 24], bounds={"left": 0, "bottom": 0, "right": 32, "top": 24})
        result = validate_pair_compatibility(InputMode.CROSS_MODAL, Modality.OPTICAL, Modality.SAR, left, right)
        self.assertFalse(result.compatible)
        self.assertEqual(result.same_crs, False)

    def test_overlapping_but_non_aligned_pair(self):
        left = metadata(crs="EPSG:4326", transform=[1, 0, 0, 0, -1, 24], bounds={"left": 0, "bottom": 0, "right": 32, "top": 24})
        right = metadata(crs="EPSG:4326", transform=[1, 0, 2, 0, -1, 24], bounds={"left": 2, "bottom": 0, "right": 34, "top": 24})
        result = validate_pair_compatibility(InputMode.CROSS_MODAL, Modality.OPTICAL, Modality.SAR, left, right)
        self.assertTrue(result.compatible)
        self.assertEqual(result.alignment_level.value, "geospatial_overlap")
        self.assertTrue(result.bounds_overlap)
        self.assertTrue(result.resampling_required)

    def test_non_georeferenced_visual_only_pair(self):
        result = validate_pair_compatibility(InputMode.CROSS_MODAL, Modality.OPTICAL, Modality.SAR, metadata(), metadata())
        self.assertTrue(result.compatible)
        self.assertEqual(result.alignment_level.value, "visual_only")
        self.assertIsNone(result.same_crs)

    def test_invalid_optical_optical_cross_modal_pair(self):
        image = ("optical.png", pillow_bytes("PNG"), "image/png")
        response = self.submit(
            image,
            secondary=image,
            query="Use both images",
            input_mode="cross_modal",
            primary_modality="optical",
            secondary_modality="optical",
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertFalse(body["pair_compatibility"]["compatible"])

    def test_valid_optical_sar_pair(self):
        optical_buffer = io.BytesIO()
        rgb = np.zeros((12, 18, 3), dtype=np.uint8)
        rgb[:, :, 0] = 80
        rgb[:, :, 1] = 160
        rgb[:, :, 2] = 30
        Image.fromarray(rgb).save(optical_buffer, format="PNG")
        response = self.submit(
            ("optical.png", optical_buffer.getvalue(), "image/png"),
            secondary=("sar.tif", tiff_bytes(np.stack([np.ones((12, 18)), np.full((12, 18), 2.0)]), metadata={"axes": "CYX"}), "image/tiff"),
            query="Use optical and SAR together",
            input_mode="cross_modal",
            primary_modality="optical",
            secondary_modality="sar",
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["pair_compatibility"]["compatible"])
        self.assertEqual(body["task"], "cross_modal_analysis")
        self.assertEqual(body["status"], "partial")
        self.assertEqual(body["cross_modal_analysis"]["status"], "partial")
        self.assertIsNone(body["cross_modal_analysis"]["statistics"])

    def test_missing_secondary_image(self):
        response = self.submit(
            ("optical.png", pillow_bytes("PNG"), "image/png"),
            query="Use both images",
            input_mode="cross_modal",
            primary_modality="optical",
            secondary_modality="sar",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "MISSING_SECONDARY_IMAGE")

    def test_bitemporal_pair_with_dates(self):
        image = ("date.png", pillow_bytes("PNG"), "image/png")
        response = self.submit(
            image,
            secondary=image,
            query="What changed between these dates?",
            input_mode="bi_temporal",
            primary_modality="optical",
            secondary_modality="optical",
            primary_date="2024-01-01",
            secondary_date="2025-01-01",
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["pair_compatibility"]["compatible"])
        self.assertEqual(body["task"], "change_description")

    def test_execution_trace_contains_ingestion_steps(self):
        response = self.submit(("scene.png", pillow_bytes("PNG"), "image/png"))
        tools = [step["tool"] for step in response.json()["execution"]["steps"]]
        for expected in ("upload_received", "file_type_validation", "metadata_extraction", "preview_generation", "query_routing", "specialist_selection"):
            self.assertIn(expected, tools)

    def test_non_captioning_answer_remains_null(self):
        response = self.submit(("scene.png", pillow_bytes("PNG"), "image/png"))
        body = response.json()
        self.assertIsNone(body["answer"])
        self.assertEqual(body["evidence"], [])
        self.assertEqual(body["confidence"]["level"], "unavailable")

    def test_sar_description_uses_sar_scene_analyzer(self):
        source = np.stack([np.ones((8, 12), dtype=np.float32), np.full((8, 12), 2.0, dtype=np.float32)])
        response = self.submit(
            ("sar.tif", tiff_bytes(source, metadata={"axes": "CYX"}), "image/tiff"),
            query="Describe this image",
            primary_modality="sar",
        )
        body = response.json()
        self.assertEqual(body["execution"]["selected_tools"][-1], "sar_scene_analyzer")
        self.assertEqual(body["status"], "partial")
        self.assertEqual(body["result_status"], "COMPLETED_WITH_LIMITATIONS")
        self.assertIsNotNone(body["answer"])
        self.assertIsNotNone(body["sar_scene_analysis"])


if __name__ == "__main__":
    unittest.main()
