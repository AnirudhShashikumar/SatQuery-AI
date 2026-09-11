"""Production regression tests for automatic single-image SAR preview translation."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import satquery_agent.api as api_module
from backend import app
from satquery_agent.models import DetectionConfidence, ImageModality, RepresentationType
from satquery_agent.services.sar_translation_service import SarTranslationError, SarTranslationService
from satquery_agent.specialists import sar_translated_optical as translated_module


def _png(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(values).save(output, format="PNG")
    return output.getvalue()


def _sar_values(size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(84)
    values = np.clip(rng.gamma(4.0, 30.0, (size, size)), 0, 255).astype(np.uint8)
    values[size // 2 :, : size // 3] //= 4
    return values


def _sar_rgb() -> bytes:
    values = _sar_values()
    return _png(np.dstack([values, values, values]))


def _query(client: TestClient, query: str, data: bytes | None = None, name: str = "sentinel-1-sar-preview.png"):
    return client.post(
        "/api/agent/query",
        data={
            "query": query,
            "input_mode": "single",
            "primary_modality": "optical",
            "primary_image_modality": "auto",
            "use_cache": "false",
        },
        files={"primary_image": (name, data or _sar_rgb(), "image/png")},
    )


def _translation_result(image: Image.Image):
    return SimpleNamespace(
        image=image,
        width=256,
        height=256,
        model_used="Pix2Pix",
        fallback_used=False,
        fallback_reason=None,
        color_correction_used=False,
        device="cpu",
        runtime_ms=5,
        stage_durations_ms={
            "sar_translation_model_load": 0,
            "sar_translation_preprocessing": 1,
            "sar_translation_inference": 3,
            "sar_translation_color_correction": 0,
            "sar_translation_artifact_generation": 1,
        },
        preprocessing_method="Pix2Pix test preprocessing",
        input_channel_interpretation="near-grayscale SAR preview",
        output_value_range=[0.0, 1.0],
        warnings=[],
        provenance={"generated_representation": True, "model_reused": True},
        artifact_url="/api/agent/previews/00000000000000000000000000000000.png",
        normalized_sar_preview_url=None,
        color_corrected_artifact_url=None,
        content_hash="b" * 64,
    )


class FakeTranslationService:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.generated_images: list[Image.Image] = []

    def translate(self, *_args, automatic_preview: bool = False):
        self.calls += 1
        assert automatic_preview is True
        if self.fail:
            raise SarTranslationError("PIX2PIX_FAILED", "Pix2Pix failed safely for this SAR preview.")
        image = Image.new("RGB", (256, 256), (17, 91, 203))
        self.generated_images.append(image)
        return _translation_result(image)


@pytest.fixture
def automatic_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "0")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_OPTICAL_SPECIALISTS_ENABLED", "1")
    service = FakeTranslationService()
    monkeypatch.setattr(translated_module, "get_sar_translation_service", lambda: service)
    monkeypatch.setattr(translated_module, "get_sve_service", lambda: SimpleNamespace(enabled=False))
    return service


def test_three_channel_sar_preview_is_detected_and_pix2pix_caption_is_primary(
    automatic_path: FakeTranslationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caption_calls = []
    monkeypatch.setattr(translated_module, "get_captioner", lambda: SimpleNamespace(
        describe=lambda image, *_args: caption_calls.append((image.mode, image.getpixel((0, 0)))) or SimpleNamespace(caption="generated optical scene"),
    ))
    monkeypatch.setattr(translated_module, "get_grounder", lambda: pytest.fail("grounder must not run"))
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _query: None)

    response = _query(TestClient(app), "Describe this image.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["primary_image_metadata"]["auto_detected_modality"] == "sar_preview"
    assert body["primary_image_metadata"]["user_confirmed_modality"] is None
    assert "SAR-like intensity statistics" in body["primary_image_metadata"]["auto_detection_reason"]
    assert automatic_path.calls == 1
    assert caption_calls == [("RGB", (17, 91, 203))]
    assert body["sar_translated_optical_analysis"]["model"] == "Pix2Pix"
    assert body["sar_translated_optical_analysis"]["optical_specialists_executed"] == ["rs_captioner"]
    assert body["sar_translated_optical_analysis"]["optical_caption"] == "generated optical scene"
    assert "not an observed optical image" in body["sar_translated_optical_analysis"]["disclosure"]
    tools = [step["tool"] for step in body["execution"]["steps"]]
    assert "modality_detection" in tools
    assert "sar_translation_inference" in tools
    assert "translated_optical_captioning" in tools
    assert "response_generation" in tools


def test_original_vqa_question_reaches_rsvqa_on_generated_image(
    automatic_path: FakeTranslationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _query: "presence")
    monkeypatch.setattr(translated_module, "get_rsvqa_specialist", lambda: SimpleNamespace(
        predict=lambda image, question, content_hash: calls.append((image.getpixel((0, 0)), question, content_hash)) or {
            "answer": "yes", "confidence": 0.8, "task": "presence", "logits": [0.0, 1.0],
            "probabilities": [0.2, 0.8], "model_used": "RSVQA Specialist v1",
        },
    ))
    monkeypatch.setattr(translated_module, "get_captioner", lambda: pytest.fail("captioner must not run"))
    monkeypatch.setattr(translated_module, "get_grounder", lambda: pytest.fail("grounder must not run"))
    question = "Is there water?"
    response = _query(TestClient(app), question)
    assert response.status_code == 200, response.text
    body = response.json()
    assert calls == [((17, 91, 203), question, "b" * 64)]
    assert body["sar_translated_optical_analysis"]["optical_vqa"]["answer"] == "yes"
    assert "rsvqa_vqa_specialist" in body["sar_translated_optical_analysis"]["optical_specialists_executed"]


def test_original_grounding_question_reaches_grounder_on_generated_image(
    automatic_path: FakeTranslationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    grounding = {"target_phrase": "road", "accepted_detection_count": 1, "annotated_preview_url": None}
    monkeypatch.setattr(translated_module, "get_grounder", lambda: SimpleNamespace(
        ground=lambda image, question, *_args: calls.append((image.getpixel((0, 0)), question)) or SimpleNamespace(model_dump=lambda **_kwargs: grounding),
    ))
    monkeypatch.setattr(translated_module, "get_captioner", lambda: pytest.fail("captioner must not run"))
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _query: None)
    question = "Locate the road."
    response = _query(TestClient(app), question)
    assert response.status_code == 200, response.text
    body = response.json()
    assert calls == [((17, 91, 203), question)]
    assert body["sar_translated_optical_analysis"]["optical_grounding"] == grounding
    assert "rs_grounder" in body["sar_translated_optical_analysis"]["optical_specialists_executed"]


def test_true_optical_rgb_does_not_invoke_pix2pix(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = np.indices((96, 96))
    rgb = np.stack([(x * 2) % 256, (y * 2) % 256, ((x + y) * 2) % 256], axis=2).astype(np.uint8)
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", lambda **_kwargs: pytest.fail("Pix2Pix must not run"))
    response = _query(TestClient(app), "Unsupported diagnostic request", _png(rgb), "optical-rgb.png")
    assert response.status_code == 200
    assert response.json()["primary_image_metadata"]["auto_detected_modality"] == "optical_rgb"


def test_smooth_panchromatic_preview_is_not_forced_to_sar() -> None:
    values = np.tile(np.linspace(30, 220, 96, dtype=np.uint8), (96, 1))
    rgb = np.dstack([values, values, values])
    response = TestClient(app).post(
        "/api/agent/inspect",
        files={"image": ("panchromatic.png", _png(rgb), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["auto_detected_modality"] == "optical_grayscale"


def test_pix2pix_failure_retains_native_sar_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "0")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_OPTICAL_SPECIALISTS_ENABLED", "1")
    service = FakeTranslationService(fail=True)
    monkeypatch.setattr(translated_module, "get_sar_translation_service", lambda: service)
    monkeypatch.setattr(translated_module, "get_sve_service", lambda: SimpleNamespace(enabled=False))
    response = _query(TestClient(app), "Describe this image.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert service.calls == 1
    assert body["sar_scene_analysis"] is not None
    assert body["sar_translated_optical_analysis"]["status"] == "translation_unavailable"
    assert any("Pix2Pix failed safely" in warning for warning in body["warnings"])


class _FakePix2Pix:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(value)


def test_automatic_pix2pix_lifecycle_loads_once_and_reuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "0")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_DEVICE", "cpu")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_SAVE_ARTIFACTS", "0")
    model = _FakePix2Pix()
    factory_calls = []
    service = SarTranslationService(model_factories={"pix2pix": lambda: factory_calls.append(1) or model})
    metadata = SimpleNamespace(
        representation=RepresentationType.DISPLAY_PREVIEW,
        auto_detected_modality=ImageModality.SAR_PREVIEW,
        effective_modality=ImageModality.SAR_PREVIEW,
        user_confirmed_modality=None,
        band_count=3,
        nodata=None,
        dtype="uint8",
    )
    raster = np.dstack([_sar_values()] * 3)
    source = Image.fromarray(raster).convert("RGB")
    first = service.translate(raster, metadata, source, automatic_preview=True)
    second = service.translate(raster, metadata, source, automatic_preview=True)
    assert factory_calls == [1]
    assert model.calls == 2
    assert service.health_payload()["load_count"] == 1
    assert service.health_payload()["reuse_count"] >= 1
    assert service.health_payload()["selected_model"] == "pix2pix"
    first.image.close()
    second.image.close()
