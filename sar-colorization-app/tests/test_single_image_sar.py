"""Focused end-to-end tests for modality-safe single-image SAR analysis."""

from __future__ import annotations

import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent.image_ingestion import apply_modality_override
from satquery_agent.models import ImageModality
from satquery_agent.reporting import cache_key_for
from satquery_agent.router import route_query
from satquery_agent.models import AgentQueryRequest, InputMode, Modality, RepresentationType, TaskType
from satquery_agent.specialists.sar_preprocessing import preprocess_sar


def png_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="JPEG")
    return buffer.getvalue()


def sar_preview(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(42)
    values = np.clip(rng.normal(150, 23, (size, size)), 0, 255).astype(np.uint8)
    yy, xx = np.indices(values.shape)
    water = ((xx - size * 0.50) / (size * 0.29)) ** 2 + ((yy - size * 0.68) / (size * 0.17)) ** 2 <= 1
    values[water] = np.clip(rng.normal(28, 4, int(water.sum())), 0, 255).astype(np.uint8)
    return values


def geotiff_bytes(
    values: np.ndarray,
    descriptions: list[str],
    nodata: float | None = None,
    *,
    crs: str = "EPSG:4326",
    transform=None,
) -> bytes:
    channels = values if values.ndim == 3 and values.shape[0] <= 4 else values[None, :, :]
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=channels.shape[2], height=channels.shape[1], count=channels.shape[0],
            dtype=str(channels.dtype), crs=crs, transform=transform or from_origin(70, 20, 0.01, 0.01), nodata=nodata,
        ) as dataset:
            dataset.write(channels)
            for index, description in enumerate(descriptions, start=1):
                dataset.set_band_description(index, description)
        return memory.read()


class SingleImageSarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def inspect(self, data: bytes, name: str, mime: str):
        response = self.client.post("/api/agent/inspect", files={"image": (name, data, mime)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def analyze(self, values: np.ndarray, *, override="sar_preview", query="Highlight the water body."):
        return self.client.post(
            "/api/agent/query",
            data={"query": query, "input_mode": "single", "primary_modality": "sar", "primary_image_modality": override, "use_cache": "false"},
            files={"primary_image": ("sar 2.png", png_bytes(values), "image/png")},
        )

    def test_one_band_png_is_ambiguous_not_optical(self):
        body = self.inspect(png_bytes(sar_preview()), "gray.png", "image/png")
        metadata = body["metadata"]
        self.assertEqual(metadata["representation"], "display_preview")
        self.assertEqual(metadata["auto_detected_modality"], "unknown")
        self.assertEqual(metadata["auto_detection_confidence"], "low")
        self.assertTrue(body["requires_modality_confirmation"])

    def test_one_band_jpeg_is_ambiguous(self):
        body = self.inspect(jpeg_bytes(sar_preview()), "gray.jpg", "image/jpeg")
        self.assertEqual(body["metadata"]["auto_detected_modality"], "unknown")
        self.assertTrue(body["requires_modality_confirmation"])

    def test_ambiguous_grayscale_stops_before_specialist(self):
        response = self.client.post(
            "/api/agent/query",
            data={"query": "Highlight the water body.", "input_mode": "single", "primary_modality": "optical", "primary_image_modality": "auto", "use_cache": "false"},
            files={"primary_image": ("gray.png", png_bytes(sar_preview()), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["result_status"], "NEEDS_USER_CONFIRMATION")
        self.assertEqual(body["execution"]["selected_tools"], ["input_validator"])
        self.assertNotIn("rs_grounder", body["execution"]["selected_tools"])

    def test_sar_water_routes_to_sar_segmenter(self):
        request = AgentQueryRequest(
            query="Highlight the water body.", input_mode=InputMode.SINGLE,
            primary_modality=Modality.SAR, primary_image_modality=ImageModality.SAR_PREVIEW,
            primary_representation=RepresentationType.DISPLAY_PREVIEW, primary_band_count=1,
            has_primary_image=True, has_secondary_image=False,
        )
        plan = route_query(request)
        self.assertEqual(plan.detected_task, TaskType.SAR_WATER_SEGMENTATION)
        self.assertEqual(plan.selected_tools[-1], "sar_water_segmenter")
        self.assertNotIn("rs_grounder", plan.selected_tools)

    def test_sar_water_full_flow_generates_evidence(self):
        response = self.analyze(sar_preview())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["task"], "sar_water_segmentation")
        self.assertEqual(body["result_status"], "COMPLETED_WITH_LIMITATIONS")
        self.assertEqual(body["execution"]["selected_tools"][-1], "sar_water_segmenter")
        result = body["sar_water_analysis"]
        self.assertEqual(result["method"], "heuristic_candidate_detector")
        self.assertIsNone(result["model_confidence"])
        self.assertIsNotNone(result["heuristic_reliability"])
        self.assertGreater(result["candidate_pixels"], 0)
        kinds = {item["type"] for item in result["evidence_products"]}
        self.assertTrue({"source_preview", "normalized_sar_preview", "binary_candidate_mask", "candidate_overlay"}.issubset(kinds))
        for item in result["evidence_products"]:
            self.assertEqual(self.client.get(item["reference"]).status_code, 200)
        self.assertFalse(any(step["tool"] == "rs_grounder" for step in body["execution"]["steps"]))

    def test_all_zero_image_is_honest_low_information_result(self):
        response = self.analyze(np.zeros((64, 64), dtype=np.uint8))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["sar_water_analysis"]
        self.assertFalse(result["water_detected"])
        self.assertEqual(result["image_area_percent"], 0)
        self.assertIsNone(result["model_confidence"])

    def test_sar_scene_and_quality_queries_use_sar_analyzer(self):
        for query, task in (("Describe the major radar-backscatter patterns.", "sar_scene_analysis"), ("Assess whether this image is suitable for SAR analysis.", "sar_quality_inspection")):
            with self.subTest(query=query):
                response = self.analyze(sar_preview(), query=query)
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["task"], task)
                self.assertEqual(body["execution"]["selected_tools"][-1], "sar_scene_analyzer")
                self.assertIsNotNone(body["sar_scene_analysis"])
                self.assertEqual(body["confidence"]["level"], "unavailable")

    def test_three_band_sar_like_content_with_optical_hint_is_ambiguous(self):
        rgb = np.dstack([sar_preview(), sar_preview(), sar_preview()])
        body = self.inspect(png_bytes(rgb), "optical.png", "image/png")
        self.assertEqual(body["metadata"]["auto_detected_modality"], "unknown")
        self.assertTrue(body["requires_modality_confirmation"])

    def test_scientific_vv_and_vv_vh_detection(self):
        one = geotiff_bytes(sar_preview().astype(np.float32)[None, :, :], ["VV"])
        one_body = self.inspect(one, "scientific.tif", "image/tiff")
        self.assertEqual(one_body["metadata"]["auto_detected_modality"], "sar_vv")
        dual_values = np.stack([sar_preview(), np.flipud(sar_preview())]).astype(np.float32)
        dual = geotiff_bytes(dual_values, ["VV", "VH"])
        dual_body = self.inspect(dual, "dual.tif", "image/tiff")
        self.assertEqual(dual_body["metadata"]["auto_detected_modality"], "sar_vv_vh")

    def test_cache_key_changes_with_override_and_specialist(self):
        common = dict(primary_hash="a" * 64, secondary_hash=None, task=TaskType.SAR_WATER_SEGMENTATION, normalized_query="Highlight water")
        left = cache_key_for(**common, safe_parameters={"effective_modality": "sar_preview", "specialist": "sar_water_segmenter", "version": "1"})
        right = cache_key_for(**common, safe_parameters={"effective_modality": "optical_grayscale", "specialist": "rs_grounder", "version": "2"})
        self.assertNotEqual(left, right)

    def test_nan_inf_and_nodata_do_not_crash_preprocessing(self):
        values = sar_preview(32).astype(np.float32)
        values[0, 0], values[0, 1], values[0, 2] = np.nan, np.inf, -9999
        tiff = geotiff_bytes(values[None, :, :], ["VV"], nodata=-9999)
        body = self.inspect(tiff, "vv.tif", "image/tiff")
        stats = body["metadata"]["band_statistics"][0]
        self.assertGreaterEqual(stats["nan_count"], 1)
        self.assertGreaterEqual(stats["inf_count"], 1)
        self.assertGreaterEqual(stats["nodata_count"], 1)

    def test_projected_scientific_sar_may_report_estimated_area(self):
        data = geotiff_bytes(
            sar_preview().astype(np.float32)[None, :, :],
            ["VV"],
            crs="EPSG:32643",
            transform=from_origin(500000, 2000000, 10, 10),
        )
        response = self.client.post(
            "/api/agent/query",
            data={
                "query": "Highlight the water body.",
                "input_mode": "single",
                "primary_modality": "sar",
                "primary_image_modality": "sar_vv",
                "use_cache": "false",
            },
            files={"primary_image": ("vv.tif", data, "image/tiff")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreater(body["sar_water_analysis"]["geographic_area_square_meters"], 0)
        self.assertTrue(body["execution"]["permitted_parameters"]["geographic_area_reported"])

    def test_json_export_contains_authoritative_sar_result(self):
        analysis = self.analyze(sar_preview()).json()
        response = self.client.post("/api/agent/report", json={"request_id": analysis["request_id"], "formats": ["json"]})
        self.assertEqual(response.status_code, 200, response.text)
        artifact = response.json()["artifacts"][0]
        exported = self.client.get(artifact["url"])
        self.assertEqual(exported.status_code, 200)
        self.assertIn(b'"sar_water_analysis"', exported.content)
        self.assertIn(b'"heuristic_reliability"', exported.content)


if __name__ == "__main__":
    unittest.main()


def test_modality_question_uses_inspected_metadata_without_neural_inference():
    from unittest.mock import patch
    client = TestClient(app)
    from satquery_agent.services.sve_service import get_sve_service
    from satquery_agent.specialists.vqa import get_vqa
    with patch.object(get_sve_service(), "analyze") as sve, patch.object(get_vqa(), "answer") as vqa, patch("satquery_agent.api.run_sar_translated_optical_evidence") as translation:
        response = client.post("/api/agent/query", data={"query": "Is this SAR?", "input_mode": "single", "primary_modality": "sar", "primary_image_modality": "auto", "use_cache": "false"}, files={"primary_image": ("sar.png", png_bytes(sar_preview()), "image/png")})
        body = response.json()
        assert response.status_code == 200, body
        assert body["status"] == "success", body
        assert body["answer"].startswith("Yes.")
        assert body["execution"]["selected_tools"] == ["input_validator"]
        assert body["primary_image_metadata"]["auto_detected_modality"] == "sar_preview"
        assert body["confidence"]["score"] is None
        sve.assert_not_called()
        vqa.assert_not_called()
        translation.assert_not_called()
        assert body["sar_translated_optical_analysis"] is None
        assert body["sar_scene_analysis"] is None
