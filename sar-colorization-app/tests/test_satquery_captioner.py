"""Captioner lifecycle, provenance, execution, and safe-failure tests."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.models import (
    CaptionResult,
    Confidence,
    ConfidenceLevel,
    ImageMetadata,
    ModelProvenance,
)
from satquery_agent.registry import public_tool_registry
from satquery_agent.specialists.captioner import (
    ADAPTATION_DATASET,
    BASE_ARCHITECTURE,
    CHECKPOINT,
    LIMITATIONS,
    MODEL_LICENSE,
    MODEL_SOURCE,
    CaptionerError,
    RemoteSensingCaptioner,
    get_captioner,
)


def optical_png() -> bytes:
    values = np.zeros((32, 32, 3), dtype=np.uint8)
    values[:, :16] = (45, 130, 55)
    values[:, 16:] = (80, 120, 180)
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def metadata() -> ImageMetadata:
    return ImageMetadata(
        file_id="fixture",
        original_name="scene.png",
        safe_name="scene.png",
        format="png",
        mime_type="image/png",
        size_bytes=100,
        width=32,
        height=32,
        band_count=3,
        dtype="uint8",
        color_interpretation=["red", "green", "blue"],
        warnings=[],
    )


class FakeProcessor:
    def __call__(self, **_kwargs):
        return {"pixel_values": torch.zeros((1, 3, 16, 16))}

    def decode(self, *_args, **_kwargs):
        return "A river runs beside vegetation and a built-up area."


class FakeModel:
    def generate(self, **_kwargs):
        return torch.tensor([[1, 2, 3]])


class LifecycleCaptioner(RemoteSensingCaptioner):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load(self) -> None:
        if self._state == "ready":
            return
        self.load_calls += 1
        self._processor = FakeProcessor()
        self._model = FakeModel()
        self._device = "cpu"
        self._state = "ready"


class ApiCaptioner:
    limitations = LIMITATIONS

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def describe(self, *_args, **_kwargs) -> CaptionResult:
        self.calls += 1
        if self.fail:
            raise CaptionerError("CAPTION_INFERENCE_FAILED", "The local captioning model could not generate a description safely.")
        return CaptionResult(
            caption="A river runs beside vegetation and a built-up area.",
            confidence=Confidence(
                level=ConfidenceLevel.MODERATE,
                score=None,
                reason="The captioning model does not provide calibrated probability estimates. Confidence reflects supported optical input only.",
            ),
            model=ModelProvenance(
                tool_id="rs_captioner",
                checkpoint=CHECKPOINT,
                base_architecture=BASE_ARCHITECTURE,
                adaptation_dataset=ADAPTATION_DATASET,
                remote_sensing_adapted=True,
                license=MODEL_LICENSE,
                source=MODEL_SOURCE,
            ),
            warnings=[],
            runtime_ms=17,
            device="cpu",
            image_representation="percentile-stretched RGB display representation",
            bands_used=["r", "g", "b"],
            model_load_ms=9,
            reused_model=False,
        )


class SatQueryCaptionerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def submit(self, modality="optical"):
        return self.client.post(
            "/api/agent/query",
            data={
                "query": "Describe the land cover and major objects visible in this image.",
                "input_mode": "single",
                "primary_modality": modality,
            },
            files={"primary_image": ("scene.png", optical_png(), "image/png")},
        )

    def test_registry_available_only_when_configured(self):
        with patch.dict(os.environ, {"SATQUERY_CAPTIONER_ENABLED": "0"}):
            tool = next(item for item in public_tool_registry() if item.id == "rs_captioner")
            self.assertEqual(tool.status.value, "not_implemented")
        tool = next(item for item in public_tool_registry() if item.id == "rs_captioner")
        self.assertEqual(tool.status.value, "available")

    def test_lazy_load_and_model_reuse(self):
        captioner = LifecycleCaptioner()
        image = Image.new("RGB", (32, 32), color=(60, 120, 80))
        first = captioner.describe(image, metadata(), "optical", ["red", "green", "blue"], "RGB display")
        second = captioner.describe(image, metadata(), "optical", ["red", "green", "blue"], "RGB display")
        self.assertEqual(captioner.load_calls, 1)
        self.assertFalse(first.reused_model)
        self.assertTrue(second.reused_model)
        self.assertTrue(first.caption)

    def test_valid_optical_caption_contract(self):
        fake = ApiCaptioner()
        with patch("satquery_agent.api.get_captioner", return_value=fake):
            response = self.submit()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertTrue(body["answer"])
        self.assertNotIn("placeholder", body["answer"].lower())
        self.assertEqual(body["evidence"], [])
        self.assertEqual(body["confidence"]["score"], None)
        self.assertIn("does not provide calibrated probability", body["confidence"]["reason"])
        self.assertEqual(body["model"]["adaptation_dataset"], ADAPTATION_DATASET)
        self.assertTrue(body["model"]["remote_sensing_adapted"])
        self.assertEqual(body["caption_details"]["runtime_ms"], 17)
        self.assertIn("caption_inference", [step["tool"] for step in body["execution"]["steps"]])
        self.assertNotIn("/Users/", response.text)
        self.assertNotIn("/private/", response.text)

    def test_rgb_file_overridden_as_sar_is_rejected_before_captioner(self):
        fake = ApiCaptioner()
        with patch("satquery_agent.api.get_captioner", return_value=fake):
            response = self.submit(modality="sar")
        body = response.json()
        self.assertEqual(body["status"], "not_implemented")
        self.assertEqual(body["result_status"], "UNSUPPORTED_INPUT")
        self.assertEqual(body["execution"]["selected_tools"][-1], "sar_scene_analyzer")
        self.assertIsNone(body["sar_scene_analysis"])
        self.assertIn("supports at most 2 bands", body["answer"])
        self.assertEqual(fake.calls, 0)

    def test_model_failure_returns_safe_error(self):
        fake = ApiCaptioner(fail=True)
        with patch("satquery_agent.api.get_captioner", return_value=fake):
            response = self.submit()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertIsNone(body["answer"])
        self.assertIn("could not generate a description safely", " ".join(body["warnings"]))
        self.assertNotIn("Traceback", response.text)

    def test_health_does_not_force_model_load(self):
        get_captioner().reset_for_tests()
        response = self.client.get("/api/agent/health")
        self.assertEqual(response.json()["specialists"]["rs_captioner"]["status"], "unloaded")

    @unittest.skipUnless(os.getenv("SATQUERY_RUN_MODEL_TESTS") == "1", "real checkpoint test is opt-in")
    def test_real_cached_checkpoint_caption(self):
        image_path = Path(os.environ["SATQUERY_CAPTION_TEST_IMAGE"])
        get_captioner().reset_for_tests()
        response = self.client.post(
            "/api/agent/query",
            data={
                "query": "Describe the land cover and major objects visible in this image.",
                "input_mode": "single",
                "primary_modality": "optical",
            },
            files={"primary_image": (image_path.name, image_path.read_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertTrue(body["answer"].strip())
        self.assertEqual(body["model"]["checkpoint"], CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
