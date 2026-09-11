"""Controlled evidence-grounded VQA regression tests."""

from __future__ import annotations

import io
import math
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent.models import SingleImageEvidenceStatistics
from satquery_agent.specialists.single_image_evidence import DEFAULT_THRESHOLDS, _dominant_scene


VQA_TRACE = [
    "upload_received", "file_type_validation", "metadata_extraction", "preview_generation",
    "query_routing", "question_classification", "optical_image_preparation", "evidence_extraction",
    "scene_summary_computation", "controlled_answer_generation", "evidence_preview_generation", "response_generation",
]


def png_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def geotiff_bytes(values: np.ndarray, transform=None) -> bytes:
    channels = values if values.ndim == 3 else np.moveaxis(values, -1, 0)
    if channels.shape[0] not in {1, 2, 3, 4}:
        channels = np.moveaxis(values, -1, 0)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=channels.shape[2], height=channels.shape[1], count=channels.shape[0],
            dtype=str(channels.dtype), crs="EPSG:4326", transform=transform or from_origin(70.0, 20.0, 0.01, 0.01),
        ) as dataset:
            dataset.write(channels)
        return memory.read()


def evidence_scene() -> np.ndarray:
    height = width = 96
    yy, xx = np.indices((height, width))
    base = (95 + ((xx + yy) % 35)).astype(np.uint8)
    scene = np.dstack((base + 15, base + 8, base)).astype(np.uint8)
    scene[8:38, 8:38] = np.array([18, 30, 58], dtype=np.uint8)
    for row in range(48, 90):
        green = 145 + (row % 8) * 7
        scene[row, 5:46] = np.array([45, green, 55], dtype=np.uint8)
        if row % 6 < 2:
            scene[row, 5:46] = np.array([70, 175, 65], dtype=np.uint8)
    for row in range(50, 90):
        for column in range(52, 92):
            scene[row, column] = 235 if (row // 3 + column // 3) % 2 else 125
    return scene


class ControlledVQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.scene = evidence_scene()

    def single(self, question: str, *, image=None, modality="optical", name="scene.png"):
        values = self.scene if image is None else image
        return self.client.post(
            "/api/agent/query",
            data={"query": question, "input_mode": "single", "primary_modality": modality},
            files={"primary_image": (name, png_bytes(values), "image/png")},
        )

    def assert_category(self, question: str, category: str):
        response = self.single(question)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["vqa_details"]["question_category"], category)
        self.assertTrue(body["vqa_details"]["supported"])
        self.assertIsNotNone(body["answer"])
        return body

    def test_supported_single_question_taxonomy(self):
        cases = {
            "What is the dominant land-cover type?": "dominant_land_cover",
            "Is a water body visible?": "presence_water",
            "Are buildings present?": "presence_buildings",
            "Is vegetation visible?": "presence_vegetation",
            "Does this scene contain agricultural land?": "presence_agriculture",
            "Is this mostly built-up?": "composition_built_up",
            "Is this mainly urban or rural?": "scene_type",
            "How much of the image appears vegetated?": "relative_coverage",
            "What is the image resolution?": "metadata_question",
        }
        for question, category in cases.items():
            with self.subTest(question=question):
                self.assert_category(question, category)

    def test_answers_use_cautious_supported_terms(self):
        water = self.assert_category("Can you see water?", "presence_water")
        self.assertIn("water-like", water["answer"])
        buildings = self.assert_category("Are buildings present?", "presence_buildings")
        self.assertNotIn("confirmed buildings", buildings["answer"].lower())
        vegetation = self.assert_category("Is vegetation visible?", "presence_vegetation")
        self.assertIn("not an NDVI", vegetation["answer"])
        agriculture = self.assert_category("Does this contain agricultural land?", "presence_agriculture")
        self.assertIn("crop type is not inferred", agriculture["answer"].lower())

    def test_metadata_dimensions_are_exact(self):
        body = self.assert_category("What is the image resolution?", "metadata_question")
        self.assertEqual(body["vqa_details"]["statistics_used"], {"width": 96, "height": 96})
        self.assertIn("96 × 96", body["answer"])

    def test_unsupported_and_empty_questions(self):
        body = self.single("Who owns the buildings in this image?").json()
        self.assertEqual(body["status"], "not_implemented")
        self.assertFalse(body["vqa_details"]["supported"])
        response = self.client.post(
            "/api/agent/query", data={"query": "   ", "input_mode": "single", "primary_modality": "optical"},
            files={"primary_image": ("scene.png", png_bytes(self.scene), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_QUERY")

    def test_single_sar_vqa_is_explicitly_unsupported(self):
        body = self.single("Is there water?", modality="sar").json()
        self.assertEqual(body["status"], "not_implemented")
        self.assertEqual(body["result_status"], "UNSUPPORTED_TASK")
        self.assertIn("SAR VQA is not implemented", body["answer"])
        self.assertNotIn("rs_grounder", body["execution"]["selected_tools"])

    def test_statistics_are_finite_and_bounded(self):
        evidence = self.assert_category("What is the dominant land-cover type?", "dominant_land_cover")["vqa_details"]["single_image_evidence"]
        for key, value in evidence["statistics"].items():
            self.assertTrue(math.isfinite(value), key)
            self.assertGreaterEqual(value, 0, key)
            self.assertLessEqual(value, 100, key)

    def test_dominant_margin_and_mixed_behavior(self):
        kwargs = dict(water_support_percent=30, vegetation_support_percent=5, built_up_support_percent=10, barren_support_percent=4, agriculture_support_percent=1, edge_density_percent=10, valid_pixel_percent=100)
        self.assertEqual(_dominant_scene(SingleImageEvidenceStatistics(**kwargs), DEFAULT_THRESHOLDS), "water_dominant")
        kwargs.update(water_support_percent=30, built_up_support_percent=27)
        self.assertEqual(_dominant_scene(SingleImageEvidenceStatistics(**kwargs), DEFAULT_THRESHOLDS), "mixed")

    def test_previews_regions_confidence_and_limitations(self):
        body = self.assert_category("Is a water body visible?", "presence_water")
        details = body["vqa_details"]
        evidence = details["single_image_evidence"]
        for key, url in evidence["previews"].items():
            self.assertRegex(url, r"/api/agent/previews/[0-9a-f]{32}\.png$", key)
            self.assertEqual(self.client.get(url).content[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(evidence["regions"])
        self.assertIsNone(details["confidence"]["score"])
        self.assertIn(details["confidence"]["level"], {"moderate", "low"})
        self.assertTrue(any("not a trained" in item for item in details["limitations"]))

    def test_world_coordinates_only_when_georeferenced(self):
        png_body = self.assert_category("Is vegetation visible?", "presence_vegetation")
        self.assertTrue(all(region["bbox_world"] is None for region in png_body["vqa_details"]["single_image_evidence"]["regions"]))
        payload = geotiff_bytes(np.moveaxis(self.scene, -1, 0))
        response = self.client.post(
            "/api/agent/query", data={"query": "Is vegetation visible?", "input_mode": "single", "primary_modality": "optical"},
            files={"primary_image": ("mapped.tif", payload, "image/tiff")},
        )
        regions = response.json()["vqa_details"]["single_image_evidence"]["regions"]
        self.assertTrue(any(region["bbox_world"] is not None for region in regions))

    def test_trace_complete_safe_and_exposes_thresholds(self):
        response = self.single("Is a water body visible?")
        body = response.json()
        self.assertEqual([step["tool"] for step in body["execution"]["steps"]], VQA_TRACE)
        evidence_step = body["execution"]["steps"][7]
        self.assertEqual(evidence_step["parameters"]["dominant_class_margin_percent"], 7.0)
        self.assertNotIn("/Users/", response.text)
        self.assertNotIn(".codex", response.text)
        self.assertNotIn("api_key", response.text.lower())

    def test_no_captioner_or_generic_llm_is_invoked(self):
        with patch("satquery_agent.api.get_captioner") as captioner:
            body = self.single("Is a water body visible?").json()
            captioner.assert_not_called()
        self.assertFalse(body["vqa_details"]["method"]["uses_language_model"])
        self.assertEqual(body["vqa_details"]["answer_source"], "computed optical support maps and metadata")

    def test_registry_truthfulness(self):
        tools = {item["id"]: item for item in self.client.get("/api/agent/tools").json()}
        vqa = tools["rs_vqa"]
        self.assertEqual(vqa["status"], "available")
        self.assertFalse(vqa["remote_sensing_adapted"])
        self.assertEqual(vqa["method_type"], "deterministic evidence-grounded VQA")
        self.assertIn("dominant_land_cover", vqa["supported_question_categories"])
        self.assertTrue(tools["rs_captioner"]["remote_sensing_adapted"])
        self.assertIn("RSICD", tools["rs_captioner"]["adaptation_dataset"])

    def change_query(self, question: str, before: np.ndarray, after: np.ndarray):
        return self.client.post(
            "/api/agent/query",
            data={"query": question, "input_mode": "bi_temporal", "primary_modality": "optical", "secondary_modality": "optical", "primary_date": "2024-01-01", "secondary_date": "2025-01-01"},
            files={"primary_image": ("before.png", png_bytes(before), "image/png"), "secondary_image": ("after.png", png_bytes(after), "image/png")},
        )

    def test_change_vqa_uses_real_statistics_and_direction(self):
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = before.copy(); after[5:35, 65:95] = 255
        percentage = self.change_query("How much of the image changed?", before, after).json()
        self.assertEqual(percentage["task"], "change_vqa")
        self.assertAlmostEqual(percentage["vqa_details"]["statistics_used"]["percentage_changed"], percentage["change_analysis"]["statistics"]["percentage_changed"])
        largest = self.change_query("Where did the largest change occur?", before, after).json()
        self.assertIn("upper-right", largest["answer"])
        self.assertNotIn("northeast", largest["answer"])
        self.assertEqual(largest["vqa_details"]["question_category"], "largest_change")

    def test_change_summary_count_magnitude_and_causal_rejection(self):
        before = np.zeros((80, 80, 3), dtype=np.uint8)
        after = before.copy(); after[10:25, 10:25] = 255; after[50:70, 50:70] = 255
        for question, category in (
            ("What changed between these dates?", "change_summary"),
            ("How many major changed regions were detected?", "change_region_count"),
            ("Did the image remain mostly unchanged?", "change_magnitude"),
        ):
            body = self.change_query(question, before, after).json()
            self.assertEqual(body["vqa_details"]["question_category"], category)
            self.assertTrue(body["vqa_details"]["supported"])
        rejected = self.change_query("What caused the change?", before, after).json()
        self.assertEqual(rejected["status"], "not_implemented")
        self.assertFalse(rejected["vqa_details"]["supported"])

    def cross_query(self, question: str):
        optical = np.moveaxis(self.scene, -1, 0)
        sar = np.full((2, 96, 96), 100, dtype=np.uint16)
        sar[:, 8:38, 8:38] = 10
        for row in range(50, 90):
            for column in range(52, 92):
                sar[:, row, column] = 260 if (row // 2 + column // 2) % 2 else 120
        return self.client.post(
            "/api/agent/query",
            data={"query": question, "input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar"},
            files={"primary_image": ("optical.tif", geotiff_bytes(optical), "image/tiff"), "secondary_image": ("sar.tif", geotiff_bytes(sar), "image/tiff")},
        )

    def test_cross_modal_vqa_uses_real_statistics(self):
        cases = (
            ("Where do both modalities agree?", "cross_modal_agreement", "agreement_percent"),
            ("What percentage appears water-like?", "cross_modal_water", "water_likelihood_percent"),
            ("Are structurally complex regions present?", "cross_modal_structure", "built_up_likelihood_percent"),
            ("Do optical and SAR evidence disagree?", "cross_modal_disagreement", "disagreement_percent"),
            ("How many joint evidence regions were found?", "cross_modal_region_count", "joint_region_count"),
        )
        for question, category, statistic in cases:
            with self.subTest(question=question):
                body = self.cross_query(question).json()
                self.assertEqual(body["vqa_details"]["question_category"], category)
                self.assertIn(statistic, body["vqa_details"]["statistics_used"])
                self.assertFalse(body["vqa_details"]["method"]["uses_language_model"])

    def test_direct_specialists_and_caption_route_remain_available(self):
        before = np.zeros((32, 32, 3), dtype=np.uint8); after = before.copy(); after[8:24, 8:24] = 255
        direct_change = self.client.post(
            "/api/agent/change", data={"before_date": "2024-01-01", "after_date": "2025-01-01", "modality": "optical"},
            files={"before_image": ("before.png", png_bytes(before), "image/png"), "after_image": ("after.png", png_bytes(after), "image/png")},
        )
        self.assertEqual(direct_change.status_code, 200)
        self.assertEqual(self.cross_query("Where do both modalities agree?").status_code, 200)
        route = self.client.post("/api/agent/route", json={"query": "Describe this image", "input_mode": "single", "primary_modality": "optical", "secondary_modality": None, "has_primary_image": True, "has_secondary_image": False}).json()
        self.assertEqual(route["task"], "captioning")
        grounding = self.client.post("/api/agent/route", json={"query": "Highlight the water", "input_mode": "single", "primary_modality": "optical", "secondary_modality": None, "has_primary_image": True, "has_secondary_image": False}).json()
        self.assertEqual(grounding["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
