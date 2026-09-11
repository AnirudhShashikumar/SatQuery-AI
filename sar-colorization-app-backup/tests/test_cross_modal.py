"""Cross-modal optical-SAR deterministic evidence-fusion tests."""

from __future__ import annotations

import io
import math
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app


TRACE = [
    "optical_upload_received",
    "sar_upload_received",
    "optical_metadata_extraction",
    "sar_metadata_extraction",
    "modality_validation",
    "pair_compatibility_check",
    "optical_preparation",
    "sar_preparation",
    "optical_evidence_extraction",
    "sar_evidence_extraction",
    "joint_evidence_fusion",
    "region_extraction",
    "preview_generation",
    "response_generation",
]


def geotiff_bytes(values: np.ndarray, *, crs: str = "EPSG:4326", transform=None) -> bytes:
    transform = transform or from_origin(70.0, 20.0, 0.01, 0.01)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            dtype=str(values.dtype),
            crs=crs,
            transform=transform,
        ) as dataset:
            dataset.write(values)
        return memory.read()


def png_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def evidence_pair(*, disagreement: bool = False, constant_sar: bool = False, dark_optical: bool = False):
    height = width = 96
    optical = np.full((3, height, width), 120, dtype=np.uint8)
    optical[2] = 100
    sar = np.full((2, height, width), 100, dtype=np.uint16)
    if dark_optical:
        optical[:] = 0
    else:
        optical[:, 10:40, 10:40] = np.array([20, 35, 55], dtype=np.uint8)[:, None, None]
        for row in range(55, 88):
            for column in range(55, 88):
                optical[:, row, column] = 230 if (row // 3 + column // 3) % 2 else 150
        optical[:, 10:35, 58:85] = np.array([45, 150, 55], dtype=np.uint8)[:, None, None]
    if constant_sar:
        sar[:] = 75
    else:
        if not disagreement:
            sar[:, 10:40, 10:40] = 10
        else:
            sar[:, 10:40, 48:78] = 10
        for row in range(55, 88):
            for column in range(55, 88):
                sar[:, row, column] = 260 if (row // 2 + column // 2) % 2 else 130
    return optical, sar


class CrossModalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def submit(self, optical, sar, **fields):
        data = {"optical_modality": "optical", "sar_modality": "sar"}
        data.update(fields)
        return self.client.post(
            "/api/agent/cross-modal",
            data=data,
            files={"optical_image": optical, "sar_image": sar},
        )

    @staticmethod
    def exact_uploads(**pair_options):
        optical, sar = evidence_pair(**pair_options)
        return (
            ("optical.tif", geotiff_bytes(optical), "image/tiff"),
            ("sar.tif", geotiff_bytes(sar), "image/tiff"),
        )

    def test_valid_exact_pair_produces_joint_evidence(self):
        response = self.submit(*self.exact_uploads())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["compatibility"]["alignment_level"], "exact")
        self.assertEqual(body["result"]["status"], "success")
        self.assertGreater(body["result"]["statistics"]["water_likelihood_percent"], 0)
        self.assertGreater(body["result"]["statistics"]["built_up_likelihood_percent"], 0)
        self.assertFalse(body["result"]["method"]["uses_trained_model"])

    def test_missing_optical_image(self):
        _, sar = self.exact_uploads()
        response = self.client.post("/api/agent/cross-modal", data={"sar_modality": "sar"}, files={"sar_image": sar})
        self.assertEqual(response.status_code, 422)

    def test_missing_sar_image(self):
        optical, _ = self.exact_uploads()
        response = self.client.post("/api/agent/cross-modal", data={"optical_modality": "optical"}, files={"optical_image": optical})
        self.assertEqual(response.status_code, 422)

    def test_optical_plus_optical_is_invalid(self):
        body = self.submit(*self.exact_uploads(), sar_modality="optical").json()
        self.assertEqual(body["result"]["status"], "failed")
        self.assertEqual(body["execution"]["steps"][4]["status"], "failed")

    def test_sar_plus_sar_is_invalid(self):
        body = self.submit(*self.exact_uploads(), optical_modality="sar").json()
        self.assertEqual(body["result"]["status"], "failed")

    def test_corrupt_optical_image(self):
        _, sar = self.exact_uploads()
        response = self.submit(("optical.tif", b"II*\x00broken", "image/tiff"), sar)
        self.assertEqual(response.status_code, 400)

    def test_corrupt_sar_image(self):
        optical, _ = self.exact_uploads()
        response = self.submit(optical, ("sar.tif", b"II*\x00broken", "image/tiff"))
        self.assertEqual(response.status_code, 400)

    def test_dimension_mismatch_requires_alignment(self):
        optical, sar = evidence_pair()
        smaller = sar[:, :80, :80]
        body = self.submit(
            ("optical.tif", geotiff_bytes(optical), "image/tiff"),
            ("sar.tif", geotiff_bytes(smaller), "image/tiff"),
        ).json()
        self.assertEqual(body["result"]["status"], "alignment_required")
        self.assertIsNone(body["result"]["statistics"])

    def test_crs_mismatch_is_incompatible(self):
        optical, sar = evidence_pair()
        body = self.submit(
            ("optical.tif", geotiff_bytes(optical, crs="EPSG:4326"), "image/tiff"),
            ("sar.tif", geotiff_bytes(sar, crs="EPSG:3857"), "image/tiff"),
        ).json()
        self.assertEqual(body["result"]["status"], "failed")
        self.assertEqual(body["compatibility"]["same_crs"], False)

    def test_overlapping_unaligned_pair_requires_alignment(self):
        optical, sar = evidence_pair()
        body = self.submit(
            ("optical.tif", geotiff_bytes(optical), "image/tiff"),
            ("sar.tif", geotiff_bytes(sar, transform=from_origin(70.02, 20.0, 0.01, 0.01)), "image/tiff"),
        ).json()
        self.assertEqual(body["compatibility"]["alignment_level"], "geospatial_overlap")
        self.assertEqual(body["result"]["status"], "alignment_required")

    def test_visual_only_pair_is_partial_without_fusion(self):
        optical, sar = evidence_pair()
        optical_png = np.moveaxis(optical, 0, -1)
        sar_png = np.repeat(sar[0, :, :, None].astype(np.uint8), 3, axis=2)
        body = self.submit(
            ("optical.png", png_bytes(optical_png), "image/png"),
            ("sar.png", png_bytes(sar_png), "image/png"),
        ).json()
        self.assertEqual(body["compatibility"]["alignment_level"], "visual_only")
        self.assertEqual(body["result"]["status"], "partial")
        self.assertIsNone(body["result"]["statistics"])
        self.assertIsNone(body["result"]["previews"]["joint_evidence"])

    def test_nonoverlapping_pair_fails(self):
        optical, sar = evidence_pair()
        body = self.submit(
            ("optical.tif", geotiff_bytes(optical), "image/tiff"),
            ("sar.tif", geotiff_bytes(sar, transform=from_origin(90.0, 40.0, 0.01, 0.01)), "image/tiff"),
        ).json()
        self.assertEqual(body["compatibility"]["bounds_overlap"], False)
        self.assertEqual(body["result"]["status"], "failed")

    def test_identical_low_information_inputs_are_partial(self):
        optical = np.full((3, 48, 48), 50, dtype=np.uint8)
        sar = np.full((1, 48, 48), 50, dtype=np.uint8)
        body = self.submit(
            ("optical.tif", geotiff_bytes(optical), "image/tiff"),
            ("sar.tif", geotiff_bytes(sar), "image/tiff"),
        ).json()
        self.assertEqual(body["result"]["status"], "partial")
        self.assertEqual(body["result"]["statistics"]["water_likelihood_percent"], 0)

    def test_all_dark_optical_is_truthfully_low_information(self):
        body = self.submit(*self.exact_uploads(dark_optical=True)).json()
        self.assertEqual(body["result"]["status"], "partial")
        self.assertIn("negligible dynamic range", " ".join(body["result"]["warnings"]))

    def test_constant_sar_returns_empty_sar_evidence(self):
        body = self.submit(*self.exact_uploads(constant_sar=True)).json()
        self.assertEqual(body["result"]["status"], "partial")
        self.assertIn("SAR input has negligible dynamic range", " ".join(body["result"]["warnings"]))

    def test_water_structural_and_disagreement_regions(self):
        agreeing = self.submit(*self.exact_uploads()).json()["result"]
        types = {region["type"] for region in agreeing["regions"]}
        self.assertIn("water_likelihood", types)
        self.assertIn("built_up_likelihood", types)
        disagreeing = self.submit(*self.exact_uploads(disagreement=True)).json()["result"]
        self.assertGreater(disagreeing["statistics"]["disagreement_percent"], 0)
        self.assertIn("disagreement", {region["type"] for region in disagreeing["regions"]})

    def test_region_pixel_and_world_coordinates(self):
        regions = self.submit(*self.exact_uploads()).json()["result"]["regions"]
        joint = next(region for region in regions if region["type"] == "water_likelihood")
        self.assertEqual(len(joint["bbox_pixels"]), 4)
        self.assertEqual(len(joint["centroid_pixels"]), 2)
        self.assertEqual(len(joint["bbox_world"]), 4)
        self.assertEqual(len(joint["centroid_world"]), 2)

    def test_visual_only_regions_have_no_world_coordinates(self):
        optical, sar = evidence_pair()
        body = self.submit(
            ("optical.png", png_bytes(np.moveaxis(optical, 0, -1)), "image/png"),
            ("sar.png", png_bytes(np.repeat(sar[0, :, :, None].astype(np.uint8), 3, axis=2)), "image/png"),
        ).json()
        self.assertEqual(body["result"]["regions"], [])

    def test_previews_statistics_confidence_and_limitations(self):
        body = self.submit(*self.exact_uploads()).json()
        result = body["result"]
        for name, url in result["previews"].items():
            self.assertIsNotNone(url, name)
            preview = self.client.get(url)
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.content[:8], b"\x89PNG\r\n\x1a\n")
        for key, value in result["statistics"].items():
            if key.endswith("_percent") and value is not None:
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 100)
        self.assertIsNone(result["confidence"]["score"])
        self.assertIn(result["confidence"]["level"], {"moderate", "low"})
        self.assertTrue(result["method"]["limitations"])

    def test_execution_trace_is_complete_and_safe(self):
        response = self.submit(*self.exact_uploads())
        body = response.json()
        self.assertEqual([step["tool"] for step in body["execution"]["steps"]], TRACE)
        self.assertTrue(all(step["status"] == "success" for step in body["execution"]["steps"]))
        self.assertNotIn("/Users/", response.text)
        self.assertNotIn(".codex", response.text)
        self.assertNotIn("checkpoint", response.text.lower())

    def test_registry_truthfully_describes_deterministic_specialist(self):
        response = self.client.get("/api/agent/tools")
        tool = next(item for item in response.json() if item["id"] == "cross_modal_optical_sar_analyzer")
        self.assertEqual(tool["status"], "available")
        self.assertFalse(tool["remote_sensing_adapted"])
        self.assertEqual(tool["method_type"], "deterministic evidence fusion")
        self.assertEqual(tool["service_path"], "/api/agent/cross-modal")
        self.assertIn("cross_modal_analysis", tool["supported_tasks"])
        self.assertTrue(tool["limitations"])

    def test_supported_query_phrases_route_to_cross_modal(self):
        queries = [
            "Use both images to identify water-covered and built-up regions.",
            "Analyse the optical and SAR images together.",
            "Where do the optical and SAR observations agree?",
            "Show areas supported by both modalities.",
            "Compare structural information between the optical and SAR images.",
        ]
        for query in queries:
            body = self.client.post(
                "/api/agent/route",
                json={
                    "query": query,
                    "input_mode": "cross_modal",
                    "primary_modality": "optical",
                    "secondary_modality": "sar",
                    "has_primary_image": True,
                    "has_secondary_image": True,
                },
            ).json()
            self.assertEqual(body["task"], "cross_modal_analysis", query)

    def test_agent_query_invokes_specialist_and_templates_real_statistics(self):
        optical, sar = self.exact_uploads()
        response = self.client.post(
            "/api/agent/query",
            data={
                "query": "Where do the optical and SAR observations agree?",
                "input_mode": "cross_modal",
                "primary_modality": "optical",
                "secondary_modality": "sar",
            },
            files={"primary_image": optical, "secondary_image": sar},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertIsNotNone(body["cross_modal_analysis"])
        self.assertIn("valid pixels", body["answer"])
        self.assertEqual([step["tool"] for step in body["execution"]["steps"]], TRACE)

    def test_vqa_and_grounding_remain_not_implemented(self):
        for query, task in (("Is there water?", "vqa"), ("Highlight the water", "grounding")):
            body = self.client.post(
                "/api/agent/route",
                json={
                    "query": query,
                    "input_mode": "single",
                    "primary_modality": "optical",
                    "secondary_modality": None,
                    "has_primary_image": True,
                    "has_secondary_image": False,
                },
            ).json()
            self.assertEqual(body["task"], task)
            self.assertEqual(body["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
