"""Mission reporting, demo safety, cache identity, and readiness integration tests."""

from __future__ import annotations

import io
import json
import os
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend import app
from satquery_agent.models import (
    CaptionResult,
    Confidence,
    ConfidenceLevel,
    ModelProvenance,
    SpecialistHealth,
    TaskType,
)
from satquery_agent.reporting import MISSION_STORE, cache_key_for
from satquery_agent.specialists.ttp_change import TTP_CLIENT, TTPClientResult


def png(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()


def geotiff(array: np.ndarray) -> bytes:
    bands = np.moveaxis(array, -1, 0)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=array.shape[0], width=array.shape[1], count=bands.shape[0],
            dtype=str(array.dtype), crs=CRS.from_epsg(4326), transform=from_origin(70, 20, .01, .01),
        ) as dataset:
            dataset.write(bands)
        return memory.read()


def scene(size: int = 96) -> np.ndarray:
    values = np.full((size, size, 3), (160, 145, 115), dtype=np.uint8)
    values[8:40, 8:40] = (20, 55, 85)
    values[50:88, 8:42] = (45, 135, 55)
    values[48:90, 52:92] = (190, 185, 175)
    for coordinate in range(52, 92, 8):
        values[48:90, coordinate:coordinate + 2] = 60
    return values


class FakeCaptioner:
    limitations = ["Caption is a model-generated scene summary and is not ground truth."]

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(status="ready", device="cpu")

    def describe(self, *_args, **_kwargs) -> CaptionResult:
        return CaptionResult(
            caption="A satellite scene containing water, vegetation, and developed patterns.",
            confidence=Confidence(level=ConfidenceLevel.MODERATE, score=None, reason="Model-generated caption requiring review."),
            model=ModelProvenance(
                tool_id="rs_captioner", checkpoint="test-rsicd-captioner", base_architecture="BLIP",
                adaptation_dataset="RSICD", remote_sensing_adapted=True, license="test", source="local test double",
            ),
            warnings=[], runtime_ms=4, device="cpu", image_representation="RGB", bands_used=["r", "g", "b"],
            model_load_ms=0, reused_model=True,
        )


class SatQueryReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        MISSION_STORE.clear()
        self.client = TestClient(app)
        self.image = scene()

    def query(self, question: str = "Is a water body visible?", image: np.ndarray | None = None, **forms):
        data = {"query": question, "input_mode": "single", "primary_modality": "optical", **forms}
        return self.client.post("/api/agent/query", data=data, files={"primary_image": ("scene.png", png(self.image if image is None else image), "image/png")})

    def report(self, request_id: str, formats=None):
        return self.client.post("/api/agent/report", json={"request_id": request_id, "formats": formats or ["pdf", "json", "zip"]})

    def test_single_vqa_report_pdf_json_zip(self):
        result = self.query().json()
        response = self.report(result["request_id"])
        self.assertEqual(response.status_code, 200)
        artifacts = {item["format"]: item for item in response.json()["artifacts"]}
        pdf = self.client.get(artifacts["pdf"]["url"])
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF"))
        report = self.client.get(artifacts["json"]["url"]).json()
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["user_query"]["question_category"], "presence_water")
        self.assertGreaterEqual(len(report["evidence"]["preview_products"]), 5)
        self.assertTrue(report["execution_trace"]["steps"])
        serialized = json.dumps(report)
        self.assertNotIn("/Users/", serialized)
        self.assertNotRegex(serialized.lower(), r'api[_-]?key[^\"]*\":\s*\"[^\"]+')
        package = zipfile.ZipFile(io.BytesIO(self.client.get(artifacts["zip"]["url"]).content))
        self.assertIn("report/mission-report.pdf", package.namelist())
        self.assertIn("report/mission-report.json", package.namelist())
        self.assertIn("report/statistics-and-regions.csv", package.namelist())
        self.assertIn("README.txt", package.namelist())
        self.assertGreaterEqual(len([name for name in package.namelist() if name.startswith("evidence/")]), 5)

    def test_caption_result_can_generate_mission_report(self):
        with patch("satquery_agent.api.get_captioner", return_value=FakeCaptioner()):
            result = self.query("Describe the land cover and major objects visible in this image.").json()
        self.assertEqual(result["task"], "captioning")
        report = self.report(result["request_id"], ["json"])
        self.assertEqual(report.status_code, 200)
        document = self.client.get(report.json()["artifacts"][0]["url"]).json()
        self.assertEqual(document["specialist_provenance"]["model"]["adaptation_dataset"], "RSICD")
        self.assertIn("satellite scene", document["answer"]["text"])

    def test_change_report_contains_before_after_and_change_products(self):
        before = self.image
        after = before.copy()
        after[10:44, 56:90] = (235, 220, 180)
        response = self.client.post(
            "/api/agent/query",
            data={"query": "How much of the image changed?", "input_mode": "bi_temporal", "primary_modality": "optical", "secondary_modality": "optical", "primary_date": "2025-01-01", "secondary_date": "2025-02-01"},
            files={"primary_image": ("before.png", png(before), "image/png"), "secondary_image": ("after.png", png(after), "image/png")},
        ).json()
        self.assertEqual(response["task"], "change_vqa")
        package_response = self.report(response["request_id"], ["zip"])
        self.assertEqual(package_response.status_code, 200)
        package = zipfile.ZipFile(io.BytesIO(self.client.get(package_response.json()["artifacts"][0]["url"]).content))
        names = package.namelist()
        self.assertGreaterEqual(len([name for name in names if name.startswith("evidence/")]), 5)
        document = json.loads(package.read("report/mission-report.json"))
        self.assertGreater(document["statistics"]["change_statistics"]["percentage_changed"], 0)
        labels = " ".join(item["label"].lower() for item in document["evidence"]["preview_products"])
        for expected in ("before", "after", "difference", "mask", "overlay"):
            self.assertIn(expected, labels)

    def test_hybrid_change_report_contains_source_labelled_masks_and_disclaimer(self):
        before = self.image
        after = before.copy()
        after[12:46, 54:90] = (235, 220, 180)
        learned = np.zeros(before.shape[:2], dtype=bool)
        learned[12:46, 54:90] = True
        ttp = TTPClientResult(
            mask=learned, changed_percentage=float(learned.mean() * 100), changed_pixels=int(learned.sum()),
            region_count=1, largest_region_pixels=int(learned.sum()), runtime_ms=88,
            model_load_ms=1200, reused_model=True, device="cuda", limitations=["Binary change detection only"],
        )
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_DEFAULT_MODE": "hybrid"}), \
             patch.object(TTP_CLIENT, "health", return_value={"lifecycle": "ready"}), \
             patch.object(TTP_CLIENT, "predict", return_value=ttp):
            result = self.client.post(
                "/api/agent/query",
                data={"query": "What changed between these dates?", "input_mode": "bi_temporal", "primary_modality": "optical", "secondary_modality": "optical", "primary_date": "2025-01-01", "secondary_date": "2025-02-01"},
                files={"primary_image": ("before.png", png(before), "image/png"), "secondary_image": ("after.png", png(after), "image/png")},
            ).json()
        self.assertEqual(result["change_engine"]["mode"], "hybrid")
        report_response = self.report(result["request_id"], ["json", "csv", "zip"])
        self.assertEqual(report_response.status_code, 200, report_response.text)
        artifacts = {item["format"]: item for item in report_response.json()["artifacts"]}
        document = self.client.get(artifacts["json"]["url"]).json()
        labels = " ".join(item["label"].lower() for item in document["evidence"]["preview_products"])
        for expected in ("ttp mask", "deterministic mask", "agreement", "disagreement"):
            self.assertIn(expected, labels)
        self.assertEqual(document["specialist_provenance"]["change_engine"]["mode"], "hybrid")
        self.assertIsNotNone(document["statistics"]["mask_comparison"]["iou"])
        self.assertIn("TTP output is a model-generated binary change prediction and is not ground truth.", document["scientific_limitations"])
        csv_text = self.client.get(artifacts["csv"]["url"]).text
        self.assertIn("mask_comparison", csv_text)

    def test_cross_modal_report_contains_modality_and_joint_evidence(self):
        optical = self.image
        gray = optical.astype(np.float32).mean(axis=2) / 255.0
        sar = np.stack([gray, np.clip(gray * .8 + .1, 0, 1)], axis=-1).astype(np.float32)
        response = self.client.post(
            "/api/agent/query",
            data={"query": "Where do both modalities agree?", "input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar"},
            files={"primary_image": ("optical.tif", geotiff(optical), "image/tiff"), "secondary_image": ("sar.tif", geotiff(sar), "image/tiff")},
        ).json()
        self.assertEqual(response["task"], "cross_modal_analysis")
        report_response = self.report(response["request_id"], ["json", "zip"])
        artifacts = {item["format"]: item for item in report_response.json()["artifacts"]}
        document = self.client.get(artifacts["json"]["url"]).json()
        labels = " ".join(item["label"].lower() for item in document["evidence"]["preview_products"])
        for expected in ("optical", "sar", "joint", "agreement", "disagreement"):
            self.assertIn(expected, labels)
        self.assertIsNotNone(document["statistics"]["cross_modal_statistics"]["agreement_percent"])

    def test_content_cache_is_labelled_and_force_rerun_bypasses_it(self):
        first = self.query(use_cache="true").json()
        second = self.query(use_cache="true").json()
        self.assertFalse(first["cache"]["cached"])
        self.assertTrue(second["cache"]["cached"])
        self.assertEqual(first["request_id"], second["request_id"])
        changed = self.image.copy(); changed[0, 0] = 255
        third = self.query(image=changed, use_cache="true").json()
        self.assertNotEqual(third["request_id"], first["request_id"])
        rerun = self.query(use_cache="true", force_rerun="true").json()
        self.assertFalse(rerun["cache"]["cached"])
        self.assertNotEqual(rerun["request_id"], first["request_id"])

    def test_cache_key_changes_with_inputs_query_and_safe_parameters(self):
        base = dict(primary_hash="a" * 64, secondary_hash=None, task=TaskType.VQA, normalized_query="Is water visible?", safe_parameters={"mode": "single"})
        first = cache_key_for(**base)
        self.assertNotEqual(first, cache_key_for(**{**base, "primary_hash": "b" * 64}))
        self.assertNotEqual(first, cache_key_for(**{**base, "normalized_query": "Is vegetation visible?"}))
        self.assertNotEqual(first, cache_key_for(**{**base, "safe_parameters": {"mode": "single", "threshold": .2}}))
        hybrid = cache_key_for(**{**base, "safe_parameters": {"mode": "hybrid", "ttp_model": "TTP", "checkpoint": "60294429b3d"}})
        deterministic = cache_key_for(**{**base, "safe_parameters": {"mode": "deterministic", "deterministic_version": "bitemporal-change-1.0"}})
        self.assertNotEqual(hybrid, deterministic)

    def test_demo_mode_is_disabled_by_default_and_serves_only_approved_files(self):
        with patch.dict(os.environ, {"SATQUERY_DEMO_MODE": "false"}):
            disabled = self.client.get("/api/agent/demo").json()
            self.assertFalse(disabled["enabled"])
            self.assertEqual(self.client.get("/api/agent/demo/files/single-optical.png").status_code, 404)
        with patch.dict(os.environ, {"SATQUERY_DEMO_MODE": "true"}):
            manifest = self.client.get("/api/agent/demo").json()
            self.assertEqual(len(manifest["workflows"]), 3)
            self.assertEqual(self.client.get(manifest["workflows"][0]["files"][0]["url"]).status_code, 200)
            self.assertEqual(self.client.get("/api/agent/demo/files/../../README.md").status_code, 404)

    def test_approved_demo_samples_execute_real_local_workflows(self):
        expected_tasks = {
            "single_vqa": "vqa",
            "change_vqa": "change_vqa",
            "cross_modal": "cross_modal_analysis",
        }
        with patch.dict(os.environ, {"SATQUERY_DEMO_MODE": "true"}):
            workflows = self.client.get("/api/agent/demo").json()["workflows"]
            for workflow in workflows:
                data = {
                    "query": workflow["query"],
                    "input_mode": workflow["input_mode"],
                    "primary_modality": workflow["primary_modality"],
                    "use_cache": "false",
                }
                for field in ("secondary_modality", "primary_date", "secondary_date"):
                    if workflow.get(field):
                        data[field] = workflow[field]
                files = {}
                for sample in workflow["files"]:
                    payload = self.client.get(sample["url"])
                    self.assertEqual(payload.status_code, 200)
                    files[f"{sample['role']}_image"] = (sample["filename"], payload.content, sample["mime_type"])
                response = self.client.post("/api/agent/query", data=data, files=files)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertIn(result["status"], {"success", "partial"})
                self.assertEqual(result["task"], expected_tasks[workflow["id"]])

    def test_alignment_required_outcome_remains_reportable(self):
        smaller = self.image[:80, :80]
        result = self.client.post(
            "/api/agent/query",
            data={
                "query": "How much of the image changed?",
                "input_mode": "bi_temporal",
                "primary_modality": "optical",
                "secondary_modality": "optical",
                "primary_date": "2025-01-01",
                "secondary_date": "2025-02-01",
            },
            files={"primary_image": ("before.png", png(self.image), "image/png"), "secondary_image": ("after.png", png(smaller), "image/png")},
        ).json()
        self.assertEqual(result["status"], "alignment_required")
        report_response = self.report(result["request_id"], ["json"])
        self.assertEqual(report_response.status_code, 200)
        document = self.client.get(report_response.json()["artifacts"][0]["url"]).json()
        self.assertFalse(document["input_summary"]["compatibility"]["same_dimensions"])
        self.assertTrue(document["input_summary"]["compatibility"]["resampling_required"])
        self.assertEqual(document["report"]["status"], "alignment_required")

    def test_registry_and_compliance_are_truthful(self):
        tools = {item["id"]: item for item in self.client.get("/api/agent/tools").json()}
        self.assertEqual(tools["report_generator"]["status"], "available")
        self.assertEqual(tools["rs_grounder"]["status"], "available")
        self.assertFalse(tools["rs_grounder"]["remote_sensing_adapted"])
        self.assertEqual(tools["rs_grounder"]["checkpoint"], "IDEA-Research/grounding-dino-tiny")
        self.assertFalse(tools["rs_vqa"]["remote_sensing_adapted"])
        self.assertTrue(tools["rs_captioner"]["remote_sensing_adapted"])
        matrix = self.client.get("/api/agent/compliance").json()
        grounding = next(item for item in matrix["requirements"] if item["requirement"] == "Text-guided grounding")
        sar = next(item for item in matrix["requirements"] if item["requirement"] == "Single SAR input acceptance")
        self.assertEqual(grounding["status"], "optional_available")
        self.assertEqual(matrix["mandatory_satisfied"], matrix["mandatory_total"])
        self.assertNotIn("Text-guided grounding", matrix["optional_not_implemented"])
        self.assertIn("not implemented", sar["limitation"].lower())

    def test_expired_or_unknown_report_is_precise(self):
        response = self.report("00000000-0000-0000-0000-000000000000", ["pdf"])
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "REPORT_SOURCE_EXPIRED")


if __name__ == "__main__":
    unittest.main()
