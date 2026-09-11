"""Deterministic bi-temporal change-analysis contract tests."""

from __future__ import annotations

import io
import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent.specialists.ttp_change import TTP_CLIENT, TTPClientError, TTPClientResult


def png_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def geotiff_bytes(values: np.ndarray, crs: str) -> bytes:
    channels = values if values.ndim == 3 else values[None, :, :]
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=channels.shape[2],
            height=channels.shape[1],
            count=channels.shape[0],
            dtype=str(channels.dtype),
            crs=crs,
            transform=from_origin(70.0, 20.0, 0.01, 0.01),
        ) as dataset:
            dataset.write(channels)
        return memory.read()


class ChangeAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def submit(self, before, after, **fields):
        data = {
            "before_date": "2024-01-01",
            "after_date": "2025-01-01",
            "modality": "optical",
        }
        data.update(fields)
        return self.client.post(
            "/api/agent/change",
            data=data,
            files={"before_image": before, "after_image": after},
        )

    @staticmethod
    def upload(name: str, values: np.ndarray):
        return (name, png_bytes(values), "image/png")

    def test_valid_pair_generates_complete_change_product(self):
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        after = before.copy()
        after[16:48, 20:44] = 255
        response = self.submit(self.upload("before.png", before), self.upload("after.png", after))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertGreater(body["statistics"]["changed_pixels"], 700)
        self.assertEqual(body["statistics"]["number_of_regions"], 1)
        self.assertGreaterEqual(body["runtime_ms"], 0)
        self.assertNotIn("answer", body)

    def test_dimension_mismatch_requires_alignment(self):
        before = np.zeros((32, 32, 3), dtype=np.uint8)
        after = np.zeros((64, 64, 3), dtype=np.uint8)
        response = self.submit(self.upload("before.png", before), self.upload("after.png", after))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "alignment_required")
        self.assertIsNone(body["statistics"])
        self.assertIsNone(body["previews"]["difference"])
        self.assertTrue(body["compatibility"]["resampling_required"])

    def test_crs_mismatch_requires_explicit_reprojection(self):
        values = np.zeros((3, 32, 32), dtype=np.uint8)
        response = self.submit(
            ("before.tif", geotiff_bytes(values, "EPSG:4326"), "image/tiff"),
            ("after.tif", geotiff_bytes(values, "EPSG:3857"), "image/tiff"),
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "alignment_required")
        self.assertFalse(body["compatibility"]["compatible"])
        self.assertEqual(body["compatibility"]["same_crs"], False)

    def test_empty_image_is_rejected(self):
        after = np.zeros((16, 16, 3), dtype=np.uint8)
        response = self.submit(
            ("before.png", b"", "image/png"),
            self.upload("after.png", after),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_FILE")

    def test_all_black_pair_has_no_change(self):
        values = np.zeros((48, 48, 3), dtype=np.uint8)
        response = self.submit(self.upload("before.png", values), self.upload("after.png", values))
        self.assertEqual(response.status_code, 200, response.text)
        statistics = response.json()["statistics"]
        self.assertEqual(statistics["changed_pixels"], 0)
        self.assertEqual(statistics["percentage_changed"], 0)
        self.assertEqual(statistics["number_of_regions"], 0)

    def test_identical_nonconstant_images_have_no_change(self):
        x = np.arange(64, dtype=np.uint8)
        values = np.repeat(np.tile(x, (64, 1))[:, :, None], 3, axis=2)
        response = self.submit(self.upload("before.png", values), self.upload("after.png", values.copy()))
        self.assertEqual(response.json()["statistics"]["changed_pixels"], 0)

    def test_large_changed_region_statistics(self):
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = before.copy()
        after[10:90, 10:90] = 200
        body = self.submit(self.upload("before.png", before), self.upload("after.png", after)).json()
        statistics = body["statistics"]
        self.assertEqual(statistics["number_of_regions"], 1)
        self.assertGreater(statistics["largest_connected_region"], 6300)
        self.assertGreater(statistics["percentage_changed"], 63)

    def test_multiple_changed_regions_are_extracted(self):
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = before.copy()
        after[10:30, 10:30] = 255
        after[60:90, 65:90] = 255
        body = self.submit(self.upload("before.png", before), self.upload("after.png", after)).json()
        statistics = body["statistics"]
        self.assertEqual(statistics["number_of_regions"], 2)
        self.assertEqual(len(statistics["bounding_boxes"]), 2)
        self.assertGreater(statistics["regions"][0]["area_pixels"], statistics["regions"][1]["area_pixels"])

    def test_mask_and_overlay_are_uuid_png_images(self):
        before = np.zeros((40, 40, 3), dtype=np.uint8)
        after = before.copy()
        after[8:32, 8:32] = 255
        body = self.submit(self.upload("before.png", before), self.upload("after.png", after)).json()
        for key in ("before", "after", "difference", "mask", "overlay"):
            url = body["previews"][key]
            self.assertRegex(url, r"/api/agent/previews/[0-9a-f]{32}\.png$")
            preview = self.client.get(url)
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.content[:8], b"\x89PNG\r\n\x1a\n")
        mask = Image.open(io.BytesIO(self.client.get(body["previews"]["mask"]).content))
        self.assertEqual(set(np.unique(np.asarray(mask))).issubset({0, 255}), True)
        overlay = Image.open(io.BytesIO(self.client.get(body["previews"]["overlay"]).content))
        self.assertEqual(overlay.mode, "RGB")

    def test_execution_trace_contains_every_required_stage(self):
        values = np.zeros((32, 32, 3), dtype=np.uint8)
        body = self.submit(self.upload("before.png", values), self.upload("after.png", values)).json()
        tools = [step["tool"] for step in body["execution"]["steps"]]
        self.assertEqual(
            tools,
            [
                "upload",
                "metadata",
                "pair_validation",
                "deterministic_analysis",
                "difference_computation",
                "thresholding",
                "morphology",
                "connected_components",
                "preview_generation",
                "ttp_eligibility_check",
                "semantic_change_interpretation",
                "response_generation",
            ],
        )
        self.assertTrue(all(step["status"] == "success" for step in body["execution"]["steps"]))

    def test_hybrid_mode_uses_ttp_primary_and_keeps_deterministic_evidence(self):
        before = np.zeros((32, 32, 3), dtype=np.uint8)
        after = before.copy()
        after[8:24, 8:24] = 255
        learned = np.zeros((32, 32), dtype=bool)
        learned[7:25, 7:25] = True
        result = TTPClientResult(
            mask=learned, changed_percentage=31.640625, changed_pixels=324,
            region_count=1, largest_region_pixels=324, runtime_ms=82,
            model_load_ms=1200, reused_model=True, device="cuda",
            limitations=["Binary change detection only"],
        )
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_DEFAULT_MODE": "hybrid"}), \
             patch.object(TTP_CLIENT, "health", return_value={"lifecycle": "ready"}), \
             patch.object(TTP_CLIENT, "predict", return_value=result):
            body = self.submit(self.upload("before.png", before), self.upload("after.png", after)).json()
        self.assertEqual(body["change_engine"]["mode"], "hybrid")
        self.assertEqual(body["change_engine"]["primary_tool"], "ttp_change_detector")
        self.assertEqual(body["statistics"]["changed_pixels"], 324)
        self.assertIsNotNone(body["deterministic_statistics"])
        self.assertIsNotNone(body["mask_comparison"]["iou"])
        for key in ("ttp_raw_mask", "ttp_mask", "ttp_overlay", "deterministic_mask", "agreement", "disagreement", "intersection", "union"):
            self.assertRegex(body["previews"][key], r"/api/agent/previews/[0-9a-f]{32}\.png$")
        self.assertIn("ttp_inference", [step["tool"] for step in body["execution"]["steps"]])

    def test_ttp_timeout_falls_back_without_failing_request(self):
        values = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_DEFAULT_MODE": "hybrid"}), \
             patch.object(TTP_CLIENT, "health", side_effect=TTPClientError("TIMEOUT", "TTP health check timed out; deterministic fallback was used.")):
            response = self.submit(self.upload("before.png", values), self.upload("after.png", values))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertTrue(body["change_engine"]["fallback_used"])
        self.assertEqual(body["change_engine"]["fallback_reason"], "timeout")
        self.assertEqual(body["statistics"]["changed_pixels"], 0)
        self.assertIn("deterministic_fallback", [step["tool"] for step in body["execution"]["steps"]])


if __name__ == "__main__":
    unittest.main()
