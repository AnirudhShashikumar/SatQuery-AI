"""Contract tests for SatQuery routing and its isolated FastAPI endpoints."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend import app
from satquery_agent.specialists.captioner import get_captioner


class SatQueryAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def query(self, query: str, **overrides):
        payload = {
            "query": query,
            "input_mode": "single",
            "primary_modality": "optical",
            "secondary_modality": None,
            "has_primary_image": True,
            "has_secondary_image": False,
        }
        payload.update(overrides)
        response = self.client.post("/api/agent/route", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_captioning_route(self):
        body = self.query("Describe this image")
        self.assertEqual(body["task"], "captioning")
        self.assertEqual(body["execution"]["selected_tools"], ["input_validator", "rs_captioner"])
        self.assertEqual(body["status"], "not_implemented")
        self.assertIsNone(body["answer"])

    def test_vqa_route(self):
        body = self.query("Is there a water body visible?")
        self.assertEqual(body["task"], "vqa")
        self.assertEqual(body["execution"]["selected_tools"][-1], "rsvqa_vqa_specialist")

    def test_grounding_route(self):
        body = self.query("Highlight the water body")
        self.assertEqual(body["task"], "grounding")
        self.assertEqual(body["execution"]["steps"][-1]["status"], "skipped")
        self.assertIn("multipart", body["warnings"][0])

    def test_change_description_route(self):
        body = self.query(
            "What changed between these dates?",
            input_mode="bi_temporal",
            secondary_modality="optical",
            has_secondary_image=True,
        )
        self.assertEqual(body["task"], "change_description")
        self.assertTrue(body["execution"]["validation"]["valid"])

    def test_change_vqa_route(self):
        body = self.query(
            "Has vegetation decreased?",
            input_mode="bi_temporal",
            secondary_modality="optical",
            has_secondary_image=True,
        )
        self.assertEqual(body["task"], "change_vqa")

    def test_cross_modal_route(self):
        body = self.query(
            "Use optical and SAR together",
            input_mode="cross_modal",
            secondary_modality="sar",
            has_secondary_image=True,
        )
        self.assertEqual(body["task"], "cross_modal_analysis")
        self.assertEqual(body["execution"]["selected_tools"][-1], "cross_modal_optical_sar_analyzer")

    def test_report_route(self):
        body = self.query("Generate report")
        self.assertEqual(body["task"], "report_generation")
        self.assertEqual(body["status"], "not_implemented")

    def test_missing_second_image_for_temporal_query(self):
        body = self.query(
            "Describe the change",
            input_mode="bi_temporal",
            secondary_modality="optical",
            has_secondary_image=False,
        )
        self.assertEqual(body["status"], "failed")
        self.assertFalse(body["execution"]["validation"]["valid"])
        self.assertIn("A secondary image is required for the selected input mode.", body["warnings"])

    def test_invalid_optical_optical_cross_modal_pair(self):
        body = self.query(
            "Use both images",
            input_mode="cross_modal",
            secondary_modality="optical",
            has_secondary_image=True,
        )
        self.assertEqual(body["status"], "failed")
        self.assertIn("one optical or multispectral image and one SAR image", " ".join(body["warnings"]))

    def test_valid_optical_sar_pair(self):
        body = self.query(
            "Combine the images",
            input_mode="cross_modal",
            secondary_modality="sar",
            has_secondary_image=True,
        )
        self.assertTrue(body["execution"]["validation"]["valid"])
        self.assertEqual(body["status"], "not_implemented")

    def test_unsupported_vague_query(self):
        body = self.query("Please help with this satellite data")
        self.assertEqual(body["task"], "unsupported")
        self.assertEqual(body["execution"]["selected_tools"], ["input_validator"])
        self.assertIsNone(body["answer"])

    def test_tool_registry_response(self):
        response = self.client.get("/api/agent/tools")
        self.assertEqual(response.status_code, 200)
        tools = {tool["id"]: tool for tool in response.json()}
        self.assertEqual(len(tools), 15)
        self.assertIn("rsvqa_vqa_specialist", tools)
        self.assertEqual(tools["satquery_vision_encoder_v1"]["base_architecture"], "OpenCLIP ViT-L-14")
        self.assertEqual(tools["satquery_vision_encoder_v1"]["adaptation_dataset"], "BigEarthNet.txt")
        self.assertEqual(tools["ttp_change_detector"]["checkpoint"], "epoch_260.pth")
        self.assertEqual(tools["ttp_change_detector"]["adaptation_dataset"], "LEVIR-CD")
        self.assertEqual(tools["rs_captioner"]["status"], "available")
        self.assertTrue(tools["rs_captioner"]["remote_sensing_adapted"])
        self.assertEqual(tools["rs_captioner"]["adaptation_dataset"], "RSICD (Remote Sensing Image Caption Dataset)")
        self.assertEqual(tools["rs_captioner"]["supported_modalities"], ["optical", "multispectral"])
        self.assertEqual(tools["pix2pix_reconstruction"]["status"], "available")
        self.assertEqual(tools["color_corrector"]["maximum_bands"], 2)
        self.assertEqual(tools["cross_modal_optical_sar_analyzer"]["status"], "available")
        self.assertEqual(tools["cross_modal_optical_sar_analyzer"]["method_type"], "deterministic evidence fusion")
        self.assertNotIn("/Users/", response.text)

    def test_health_route(self):
        get_captioner().reset_for_tests()
        response = self.client.get("/api/agent/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual({key: body[key] for key in ("status", "module", "router", "registry")}, {"status": "ok", "module": "satquery-agent", "router": "ready", "registry": "ready"})
        self.assertEqual(body["specialists"]["rs_captioner"], {"status": "unloaded", "device": None, "error": None})


if __name__ == "__main__":
    unittest.main()
