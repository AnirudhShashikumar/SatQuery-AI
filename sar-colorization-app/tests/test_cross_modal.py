"""Cross-modal optical-SAR deterministic evidence-fusion tests."""

from __future__ import annotations

import io
import math
import unittest
import pytest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent.models import CrossModalResult
from satquery_agent.specialists.vqa import answer_cross_modal_question, classify_cross_modal_question


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
    "quantitative_spatial_fusion_eligibility",
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
        self.assertEqual(
            {fact["source"] for fact in body["result"]["evidence_facts"]},
            {"optical", "native_sar", "fused"},
        )

    def test_combined_water_and_built_up_answer_retains_provenance(self):
        payload = self.submit(*self.exact_uploads()).json()["result"]
        controlled = answer_cross_modal_question(
            "Identify built-up and water-covered regions using both images.",
            CrossModalResult.model_validate(payload),
        )
        self.assertTrue(controlled.details.supported)
        self.assertIn("water-like", controlled.answer)
        self.assertIn("structural-likelihood", controlled.answer)
        self.assertEqual(
            controlled.details.statistics_used["evidence_sources"],
            ["optical", "native_sar", "fused"],
        )

    def test_sar_complementary_intent_is_supported_without_semantic_invention(self):
        intent = classify_cross_modal_question("What does SAR reveal that optical does not?")
        self.assertEqual(intent.target, "sar_complementary")
        payload = self.submit(*self.exact_uploads(disagreement=True)).json()["result"]
        controlled = answer_cross_modal_question(
            "What does SAR reveal that optical does not?",
            CrossModalResult.model_validate(payload),
        )
        self.assertTrue(controlled.details.supported)
        self.assertIn("Native SAR", controlled.answer)
        self.assertIn("not confirmed semantic classes", controlled.answer)

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

    def test_ambiguous_visual_pair_is_rejected_without_fusion(self):
        optical, sar = evidence_pair()
        optical_png = np.moveaxis(optical, 0, -1)
        sar_png = np.repeat(sar[0, :, :, None].astype(np.uint8), 3, axis=2)
        body = self.submit(
            ("optical.png", png_bytes(optical_png), "image/png"),
            ("sar.png", png_bytes(sar_png), "image/png"),
        ).json()
        self.assertEqual(body["compatibility"]["role_validation_status"], "ambiguous")
        self.assertEqual(body["result"]["status"], "failed")
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


# Reproducible 256x256 regression pair: color optical and textured grayscale SAR.
@pytest.fixture
def preview_pair():
    y, x = np.indices((256, 256))
    optical = np.stack([70 + x // 3, 85 + y // 3, 45 + (x + y) // 5], axis=2).astype(np.uint8)
    rng = np.random.default_rng(26167)
    sar = np.clip(rng.gamma(3, 32, (256, 256)), 15, 240).astype(np.uint8)
    optical[170:245, 15:100] = [15, 35, 65]
    sar[170:245, 15:100] = 4
    optical[30:95, 165:240] = np.where(((x[30:95, 165:240] // 4 + y[30:95, 165:240] // 4) % 2)[..., None], 230, 130)
    sar[30:95, 165:240] = np.where((x[30:95, 165:240] + y[30:95, 165:240]) % 2, 240, 125)
    return ("optical-demo.png", png_bytes(optical), "image/png"), ("sar-demo.png", png_bytes(sar), "image/png")


def query_pair(pair, query="Where do both modalities agree?", **data):
    return TestClient(app).post("/api/agent/query", data={"query": query, "input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar", "primary_observation_role": "optical", "secondary_observation_role": "sar", **data}, files={"primary_image": pair[0], "secondary_image": pair[1]})


@pytest.mark.parametrize("query", ["Where do both modalities agree?", "Is water visible in both observations?", "What does SAR reveal that optical does not?", "Compare built-up areas.", "Where is structural evidence strongest?", "Describe the scene using both modalities.", "Use the optical and SAR images together to identify built-up and water-covered regions."])
def test_corrected_png_qualitative_intents(preview_pair, query):
    response = query_pair(preview_pair, query)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success", body
    assert body["result_status"] == "COMPLETED"
    result = body["cross_modal_analysis"]
    assert result["analysis_level"] == "qualitative"
    assert result["statistics"] is None and result["regions"] == []
    assert not result["quantitative_metrics_available"]
    assert result["previews"]["joint_evidence"] is None
    assert result["previews"]["agreement"] is None
    assert result["optical_preparation"] and result["sar_preparation"]
    assert result["agreements"]
    assert {fact["source"] for fact in result["evidence_facts"]} == {"optical", "native_sar", "fused"}
    assert "%" not in body["answer"]
    assert "qualitative" in body["answer"].lower()
    assert "geospatial co-registration cannot be independently verified" in body["answer"]
    steps = {step["tool"]: step for step in body["execution"]["steps"]}
    for name in ("optical_preparation", "sar_preparation", "optical_evidence_extraction", "sar_evidence_extraction", "joint_evidence_fusion"):
        assert steps[name]["status"] == "success"
    assert steps["quantitative_spatial_fusion_eligibility"]["status"] == "skipped"
    assert steps["quantitative_spatial_fusion_eligibility"]["parameters"]["reason"]


@pytest.mark.parametrize("endpoint", ["query", "cross-modal"])
def test_actual_reversed_png_regression_is_blocked(preview_pair, endpoint):
    with patch("satquery_agent.api.get_cross_modal_analyzer") as analyzer:
        if endpoint == "query":
            response = query_pair(preview_pair[::-1])
        else:
            response = TestClient(app).post("/api/agent/cross-modal", files={"optical_image": preview_pair[1], "sar_image": preview_pair[0]})
        body = response.json()
        if endpoint == "query":
            analyzer.assert_not_called()
            compatibility = body["pair_compatibility"]
            assert body["status"] == "failed"
        else:
            # Invalid compatibility is guarded before source preparation.
            compatibility = body["compatibility"]
        assert compatibility["role_validation_status"] == "mismatch"
        assert compatibility["role_match"] is False
        assert compatibility["pair_valid"] is False
        assert len(compatibility["errors"]) >= 2


@pytest.mark.parametrize("query", ["What percentage overlaps exactly?", "What is the IoU?", "Give geographic coordinates for water", "What area is water in hectares?", "How many joint regions?"])
def test_png_quantitative_requests_remain_unavailable(preview_pair, query):
    body = query_pair(preview_pair, query).json()
    assert body["status"] == "partial", body
    assert body["result_status"] == "COMPLETED_WITH_LIMITATIONS"
    assert body["cross_modal_analysis"]["statistics"] is None
    assert body["vqa_details"]["statistics_used"] == {}
    assert "%" not in body["answer"]


def test_source_roles_and_legacy_explicit_order(preview_pair):
    normal = query_pair(preview_pair).json()
    legacy = query_pair(preview_pair[::-1], primary_modality="sar", secondary_modality="optical", primary_observation_role="sar", secondary_observation_role="optical").json()
    for body in (normal, legacy):
        assert body["status"] == "success"
        metadata = {item["observation_role"]: item for item in (body["primary_image_metadata"], body["secondary_image_metadata"])}
        for product in body["cross_modal_analysis"]["evidence_products"]:
            role = product["source_role"]
            assert product["source_observation_id"] == metadata[role]["file_id"]
            assert product["source_modality"] == metadata[role]["auto_detected_modality"]
            if product["evidence_type"] == "source":
                assert product["reference"] == metadata[role]["preview_url"]
    assert legacy["primary_image_metadata"]["auto_detected_modality"] == "sar_preview"
    assert legacy["primary_image_metadata"]["user_confirmed_modality"] is None


@pytest.mark.parametrize("extractor", ["_optical_evidence", "_sar_evidence"])
def test_native_specialist_failure_does_not_fabricate_fusion(preview_pair, extractor):
    with patch(f"satquery_agent.specialists.cross_modal.{extractor}", side_effect=RuntimeError("unavailable")):
        body = query_pair(preview_pair).json()
    assert body["status"] == "failed"
    assert body["cross_modal_analysis"]["statistics"] is None
    assert not body["cross_modal_analysis"]["agreements"]
    assert any(step["status"] == "failed" for step in body["execution"]["steps"])


@pytest.mark.parametrize("translation_enabled", ["true", "false"])
def test_cross_modal_native_evidence_independent_of_optional_translation(preview_pair, monkeypatch, translation_enabled):
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", translation_enabled)
    with patch("satquery_agent.api.run_sar_translated_optical_evidence", side_effect=AssertionError("Native pair must not require translation")):
        body = query_pair(preview_pair).json()
    assert body["status"] == "success"
    assert body["sar_translated_optical_analysis"] is None


def test_cross_modal_json_pdf_zip_roles(preview_pair):
    import zipfile
    from satquery_agent.models import AgentResponse
    from satquery_agent.reporting import build_report_document, _report_previews
    body = query_pair(preview_pair).json()
    report = build_report_document(AgentResponse.model_validate(body), "2026-09-05T00:00:00Z", [])
    roles = report["input_summary"]["observations_by_role"]
    assert roles["optical"]["original_name"] == preview_pair[0][0]
    assert roles["sar"]["original_name"] == preview_pair[1][0]
    previews = dict(_report_previews(body))
    assert previews["Optical source"] == roles["optical"]["preview_url"]
    assert previews["SAR source"] == roles["sar"]["preview_url"]
    client = TestClient(app)
    for format in ("json", "pdf", "zip"):
        response = client.post("/api/agent/report", json={"request_id": body["request_id"], "formats": [format]})
        assert response.status_code == 200, response.text
        payload = response.json()
        artifacts = {item["format"]: item for item in payload["artifacts"]}
        content = client.get(artifacts[format]["url"]).content
        if format == "json":
            import json
            exported = json.loads(content)
            assert exported["input_summary"]["observations_by_role"] == roles
        elif format == "pdf":
            assert content.startswith(b"%PDF")
        else:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                import json
                reports = [name for name in archive.namelist() if name.endswith(".json")]
                assert reports
                exported = json.loads(archive.read(reports[0]))
                assert exported["input_summary"]["observations_by_role"] == roles


def test_misleading_filename_cannot_override_chromatic_rgb(preview_pair):
    response = TestClient(app).post("/api/agent/inspect", files={"image": ("sar-vv.png", preview_pair[0][1], "image/png")})
    assert response.json()["metadata"]["auto_detected_modality"] == "optical_rgb"


@pytest.mark.parametrize("name", ["sar.png", "optical.png", "unknown.png"])
def test_uninformative_or_conflicting_preview_is_ambiguous(name):
    rng = np.random.default_rng(22)
    gray = np.clip(rng.gamma(3, 32, (256, 256)), 0, 255).astype(np.uint8) if name == "optical.png" else np.full((256, 256), 80, dtype=np.uint8)
    body = TestClient(app).post("/api/agent/inspect", files={"image": (name, png_bytes(gray), "image/png")}).json()
    assert body["metadata"]["auto_detected_modality"] == "unknown"


def test_sar_alone_cannot_create_water_agreement(preview_pair):
    low_information_optical = np.full((256, 256, 3), [20, 40, 90], dtype=np.uint8)
    body = query_pair((("optical.png", png_bytes(low_information_optical), "image/png"), preview_pair[1]), "Is water visible in both observations?").json()
    assert body["status"] == "success"
    assert body["confidence"]["level"] == "low"
    assert body["cross_modal_analysis"]["agreements"] == []
    assert "did not establish" in body["answer"]

@pytest.mark.parametrize("question, marker", [("How reliable is this comparison?", "not a calibrated probability"), ("What are the limitations?", "qualitative")])
def test_query_specific_evidence_review(preview_pair, question, marker):
    body = query_pair(preview_pair, question).json()
    assert body["status"] == "success", body
    assert marker in body["answer"].lower()
    assert body["cross_modal_analysis"]["statistics"] is None
    assert body["vqa_details"]["statistics_used"]["review_intent"] in {"confidence", "limitations"}


def test_stored_analysis_reopens_without_new_inference(preview_pair):
    body = query_pair(preview_pair).json()
    client = TestClient(app)
    with patch("satquery_agent.api.get_cross_modal_analyzer") as analyzer:
        reopened = client.get(f'/api/agent/results/{body["request_id"]}')
        assert reopened.status_code == 200
        assert reopened.json() == body
        analyzer.assert_not_called()
    item = next(row for row in client.get("/api/agent/comparison-items").json() if row["request_id"] == body["request_id"])
    assert item["query"] == "Where do both modalities agree?"
    assert item["answer_summary"] == body["answer"]
    assert client.get("/api/agent/results/not-a-real-result").status_code == 404


def test_sector_summary_does_not_promote_sector_counts_to_pixel_coverage():
    from satquery_agent.specialists.vqa import summarize_image_sectors
    all_sectors = [f"{row}-{col}" for row in ("upper", "middle", "lower") for col in ("left", "center", "right")]
    assert summarize_image_sectors(all_sectors) == "all nine image-relative sectors"
    assert summarize_image_sectors(all_sectors[:3]) == "the upper row of image-relative sectors"
    assert "%" not in summarize_image_sectors(all_sectors[:7])
