"""Unified mission comparison contracts, identity rules, and reports."""

from __future__ import annotations

import base64
import io
import json
import time
import unittest
from typing import Optional
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.comparison import COMPARISON_STORE, ComparisonIdentity
from satquery_agent.models import (
    ComparisonItem,
    ComparisonLineageNode,
    ComparisonPreview,
    ComparisonTask,
    InputMode,
    Modality,
    SVEResult,
    SVESemanticComparison,
)
from satquery_agent.services.sve_service import SVECall


def item(request_id: str, task: ComparisonTask, *, runtime: int = 20, warning: Optional[str] = None) -> ComparisonItem:
    return ComparisonItem(
        request_id=request_id,
        task=task,
        display_name=task.value,
        status="success",
        created_at="2026-08-24T00:00:00Z",
        input_mode=InputMode.SINGLE,
        modalities=[Modality.OPTICAL],
        input_previews=[ComparisonPreview(label="Input", url=f"/api/agent/previews/{'a' * 32}.png", kind="input", width=64, height=64)],
        output_previews=[ComparisonPreview(label="Output", url=f"/api/agent/previews/{'b' * 32}.png", kind="evidence", width=64, height=64)],
        statistics={"measured": runtime},
        confidence={"level": "moderate", "score": None, "reason": "test basis"},
        provenance={"method": "test"},
        selected_tools=[task.value],
        execution_duration_ms=runtime,
        warnings=[warning] if warning else [],
        limitations=["test limitation"],
        lineage=[ComparisonLineageNode(kind="input", label="Input"), ComparisonLineageNode(kind="tool", label=task.value)],
    )


def identity(primary: str, family: str, **updates) -> ComparisonIdentity:
    return ComparisonIdentity(primary_hash=primary * 64, task_family=family, **updates)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        COMPARISON_STORE.clear()
        self.max_items = COMPARISON_STORE.max_items
        self.ttl = COMPARISON_STORE.ttl_seconds

    def tearDown(self) -> None:
        COMPARISON_STORE.max_items = self.max_items
        COMPARISON_STORE.ttl_seconds = self.ttl
        COMPARISON_STORE.clear()

    def put(self, request_id: str, task: ComparisonTask, primary: str = "a", family: str = "single_image", **identity_updates) -> None:
        runtime = identity_updates.pop("runtime", 20)
        COMPARISON_STORE.put(item(request_id, task, runtime=runtime), identity(primary, family, **identity_updates))

    def test_comparison_list_returns_recent_safe_summaries(self):
        self.put("one", ComparisonTask.CAPTIONING)
        response = self.client.get("/api/agent/comparison-items")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["request_id"], "one")
        self.assertNotIn("statistics", response.json()[0])

    def test_result_detail_is_authoritative_normalized_item(self):
        self.put("detail", ComparisonTask.VQA)
        body = self.client.get("/api/agent/comparison-items/detail").json()
        self.assertEqual(body["statistics"], {"measured": 20})
        self.assertIn("identity_basis", body["input_identity"])

    def test_history_is_bounded(self):
        COMPARISON_STORE.max_items = 2
        for request_id in ("one", "two", "three"):
            self.put(request_id, ComparisonTask.CAPTIONING)
        self.assertEqual([entry["request_id"] for entry in self.client.get("/api/agent/comparison-items").json()], ["three", "two"])

    def test_safe_contract_removes_secrets(self):
        unsafe = item("safe", ComparisonTask.VQA, warning="api_key=supersecret")
        unsafe.provenance = {"api_key": "supersecret", "method": "safe"}
        COMPARISON_STORE.put(unsafe, identity("a", "single_image", safe_parameters={"secret": "supersecret"}))
        self.assertNotIn("supersecret", self.client.get("/api/agent/comparison-items/safe").text)

    def test_safe_contract_removes_local_paths(self):
        unsafe = item("paths", ComparisonTask.VQA, warning="/Users/example/private/model.bin")
        COMPARISON_STORE.put(unsafe, identity("a", "single_image"))
        text = self.client.get("/api/agent/comparison-items/paths").text
        self.assertNotIn("/Users/", text)

    def test_same_input_detection_uses_content_hash(self):
        self.put("left", ComparisonTask.CAPTIONING, primary="a")
        self.put("right", ComparisonTask.CAPTIONING, primary="a")
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertTrue(body["assessments"][0]["shared_inputs"])
        self.assertEqual(body["assessments"][0]["level"], "direct")

    def test_filename_only_match_is_rejected(self):
        COMPARISON_STORE.put(item("left", ComparisonTask.CAPTIONING), identity("a", "single_image", safe_parameters={"filename": "same.png"}))
        COMPARISON_STORE.put(item("right", ComparisonTask.CAPTIONING), identity("b", "single_image", safe_parameters={"filename": "same.png"}))
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertFalse(body["assessments"][0]["shared_inputs"])
        self.assertEqual(body["assessments"][0]["level"], "not_direct")

    def test_repeated_vqa_requires_same_question(self):
        self.put("left", ComparisonTask.VQA, query_hash="1" * 64)
        self.put("right", ComparisonTask.VQA, query_hash="1" * 64)
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertEqual(body["overall_level"], "direct")

    def test_different_vqa_question_is_partial(self):
        self.put("left", ComparisonTask.VQA, query_hash="1" * 64)
        self.put("right", ComparisonTask.VQA, query_hash="2" * 64)
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertEqual(body["overall_level"], "partial")

    def test_caption_and_grounding_same_image_are_partial(self):
        self.put("left", ComparisonTask.CAPTIONING)
        self.put("right", ComparisonTask.GROUNDING, target_hash="3" * 64)
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertEqual(body["overall_level"], "partial")

    def test_unrelated_tasks_are_not_direct(self):
        self.put("left", ComparisonTask.CHANGE, primary="a", family="change", secondary_hash="c" * 64)
        self.put("right", ComparisonTask.GROUNDING, primary="b")
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["left", "right"]}).json()
        self.assertEqual(body["overall_level"], "not_direct")
        self.assertIn("Not directly comparable", " ".join(body["warnings"]))

    def test_pix2pix_and_sarfusionformer_same_input_are_direct(self):
        self.put("pix", ComparisonTask.PIX2PIX, family="reconstruction")
        self.put("sar", ComparisonTask.SARFUSIONFORMER, family="reconstruction")
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["pix", "sar"]}).json()
        self.assertEqual(body["overall_level"], "direct")
        self.assertTrue(body["assessments"][0]["overlay_allowed"])

    def test_unrelated_scene_warning_is_explicit(self):
        self.put("pix", ComparisonTask.PIX2PIX, primary="a", family="reconstruction")
        self.put("sar", ComparisonTask.SARFUSIONFORMER, primary="b", family="reconstruction")
        body = self.client.post("/api/agent/comparison-assessment", json={"request_ids": ["pix", "sar"]}).json()
        self.assertEqual(body["overall_level"], "not_direct")
        self.assertIn("input identity differs", " ".join(body["warnings"]))

    def test_report_generation_returns_pdf_json_and_zip(self):
        self.put("left", ComparisonTask.CAPTIONING)
        self.put("right", ComparisonTask.VQA, query_hash="1" * 64)
        response = self.client.post("/api/agent/comparison-report", json={"request_ids": ["left", "right"], "formats": ["pdf", "json", "zip"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({artifact["format"] for artifact in response.json()["artifacts"]}, {"pdf", "json", "zip"})

    def test_expired_result_returns_gone(self):
        self.put("expired", ComparisonTask.CAPTIONING)
        COMPARISON_STORE.ttl_seconds = 1
        COMPARISON_STORE._records["expired"].touched_at = time.time() - 2
        response = self.client.get("/api/agent/comparison-items/expired")
        self.assertEqual(response.status_code, 410)

    def test_invalid_request_id_returns_not_found(self):
        self.assertEqual(self.client.get("/api/agent/comparison-items/unknown").status_code, 404)

    def test_two_item_minimum_is_enforced(self):
        response = self.client.post("/api/agent/comparison-report", json={"request_ids": ["one"]})
        self.assertEqual(response.status_code, 422)

    def test_four_item_maximum_is_enforced(self):
        response = self.client.post("/api/agent/comparison-report", json={"request_ids": ["1", "2", "3", "4", "5"]})
        self.assertEqual(response.status_code, 422)

    def test_report_uses_backend_statistics_not_frontend_values(self):
        self.put("left", ComparisonTask.CAPTIONING, runtime=17)
        self.put("right", ComparisonTask.CAPTIONING, runtime=29)
        response = self.client.post("/api/agent/comparison-report", json={"request_ids": ["left", "right"], "statistics": {"measured": 999}, "formats": ["json"]})
        artifact = response.json()["artifacts"][0]
        document = self.client.get(artifact["url"]).json()
        self.assertEqual(document["selected_results"][0]["statistics"], {"measured": 17})
        self.assertNotIn("999", json.dumps(document))

    def test_lineage_is_preserved_for_derived_output(self):
        parent_identity = identity("a", "reconstruction", output_hashes=("d" * 64,))
        COMPARISON_STORE.put(item("parent", ComparisonTask.SARFUSIONFORMER), parent_identity)
        COMPARISON_STORE.put(item("child", ComparisonTask.CAPTIONING), identity("d", "single_image"))
        body = self.client.get("/api/agent/comparison-items/child").json()
        self.assertIn("parent", [node.get("request_id") for node in body["lineage"]])

    def test_report_never_declares_a_false_winner(self):
        self.put("fast", ComparisonTask.CAPTIONING, runtime=1)
        self.put("slow", ComparisonTask.CAPTIONING, runtime=100)
        response = self.client.post("/api/agent/comparison-report", json={"request_ids": ["fast", "slow"], "formats": ["json"]})
        document = self.client.get(response.json()["artifacts"][0]["url"]).json()
        self.assertFalse(document["comparison"]["winner_declared"])
        self.assertNotIn("best", json.dumps(document).lower())

    def test_pix2pix_endpoint_persists_an_authoritative_model_result(self):
        source = io.BytesIO(); Image.new("RGB", (8, 8), "gray").save(source, format="PNG")
        output = io.BytesIO(); Image.new("RGB", (8, 8), "green").save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        with patch("backend.pix2pix_payload", return_value={"input_preview": encoded, "output": encoded, "metrics": None, "inference_time_ms": 4.0, "checkpoint": "pix.pt"}):
            response = self.client.post("/api/pix2pix/infer", files={"file": ("scene.png", source.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        detail = self.client.get(f"/api/agent/comparison-items/{response.json()['request_id']}").json()
        self.assertEqual(detail["task"], "pix2pix")
        self.assertEqual(detail["statistics"], {"reconstruction": {}})

    def test_sarfusionformer_endpoint_persists_an_authoritative_model_result(self):
        array = np.ones((2, 8, 8), dtype=np.float32)
        source = io.BytesIO(); np.save(source, array)
        preview = Image.new("L", (8, 8), 80)
        generated = {"raw_rgb": np.zeros((8, 8, 3), dtype=np.float32), "display_rgb": np.ones((8, 8, 3), dtype=np.float32) * .5, "corrected_raw_rgb": None, "corrected_display_rgb": None, "previews": {"vv": preview, "vh": preview, "sar": preview.convert("RGB")}, "duration_ms": 6.0, "warning": None, "diagnostics": {}, "corrected_diagnostics": None}
        with patch("backend.sarfusionformer_generate", return_value=generated):
            response = self.client.post("/api/sarfusionformer/infer", files={"combined_file": ("scene.npy", source.getvalue(), "application/octet-stream")}, data={"apply_color_correction": "false"})
        self.assertEqual(response.status_code, 200, response.text)
        detail = self.client.get(f"/api/agent/comparison-items/{response.json()['request_id']}").json()
        self.assertEqual(detail["task"], "sarfusionformer")
        self.assertEqual(len(detail["input_previews"]), 3)

    def test_sarfusionformer_reference_optical_gets_generated_rgb_semantic_support(self):
        class FakeSVE:
            enabled = True
            def compare(self, *_args, **_kwargs):
                return SVECall(result=SVEResult(
                    available=True, status="success", device="cpu", runtime_ms=3,
                    semantic_comparison=SVESemanticComparison(
                        label="Optical-to-generated-RGB semantic consistency", status="supporting_evidence",
                        similarity=.42, disclaimer="Supporting scene-level evidence only.",
                    ),
                ))

        array = np.ones((2, 8, 8), dtype=np.float32)
        source = io.BytesIO(); np.save(source, array)
        optical = io.BytesIO(); Image.new("RGB", (8, 8), "green").save(optical, format="PNG")
        preview = Image.new("L", (8, 8), 80)
        generated = {"raw_rgb": np.zeros((8, 8, 3), dtype=np.float32), "display_rgb": np.ones((8, 8, 3), dtype=np.float32) * .5, "corrected_raw_rgb": None, "corrected_display_rgb": None, "previews": {"vv": preview, "vh": preview, "sar": preview.convert("RGB")}, "duration_ms": 6.0, "warning": None, "diagnostics": {}, "corrected_diagnostics": None}
        with patch("backend.sarfusionformer_generate", return_value=generated), patch("backend.get_sve_service", return_value=FakeSVE()):
            response = self.client.post(
                "/api/sarfusionformer/infer",
                files={
                    "combined_file": ("scene.npy", source.getvalue(), "application/octet-stream"),
                    "ground_truth": ("optical.png", optical.getvalue(), "image/png"),
                },
                data={"apply_color_correction": "false"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        comparison = response.json()["sve_result"]["semantic_comparison"]
        self.assertEqual(comparison["label"], "Optical-to-generated-RGB semantic consistency")
        self.assertEqual(comparison["similarity"], .42)


if __name__ == "__main__":
    unittest.main()
