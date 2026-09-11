"""Focused tests for the optional single-image SAR translation evidence path."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from satquery_agent.models import (
    DetectionConfidence,
    EvidenceItem,
    EvidenceLifecycleState,
    ImageFormat,
    ImageMetadata,
    ImageModality,
    RepresentationType,
    TaskType,
)
from satquery_agent.image_ingestion import IngestedImage
from satquery_agent.services.sar_translation_service import (
    SarTranslationError,
    SarTranslationService,
    evaluate_translation_eligibility,
    translation_enabled,
)
from satquery_agent.specialists.sar_optical_fusion import fuse_sar_optical_evidence
from satquery_agent.specialists import sar_translated_optical as translated_module
from satquery_agent.specialists.captioner import CaptionerError


def metadata(*, bands: int = 2, modality: ImageModality = ImageModality.SAR_VV_VH, representation: RepresentationType = RepresentationType.SCIENTIFIC_RASTER) -> ImageMetadata:
    return ImageMetadata(
        file_id="test", original_name="test.tif", safe_name="test.tif", format=ImageFormat.TIFF,
        mime_type="image/tiff", size_bytes=100, width=16, height=16, band_count=bands, dtype="float32",
        color_interpretation=[f"band_{index + 1}" for index in range(bands)],
        band_descriptions=["VV", "VH"][:bands], representation=representation,
        auto_detected_modality=modality, auto_detection_confidence=DetectionConfidence.HIGH,
        auto_detection_reason="test", user_confirmed_modality=modality, effective_modality=modality,
    )


class FakeSarFusionFormer:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def __call__(self, value: torch.Tensor):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls += 1
            lab = torch.full((value.shape[0], 3, value.shape[2], value.shape[3]), 0.5, device=value.device)
            return {"lab": lab}
        finally:
            with self.lock:
                self.active -= 1


class FakePix2Pix:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(value)


class FakeColorCorrector:
    def __call__(self, value: torch.Tensor):
        correction = torch.full_like(value, 0.05)
        return torch.clamp(value + correction, 0, 1), correction


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "1")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_DEVICE", "cpu")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_SAVE_ARTIFACTS", "false")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_USE_COLOR_CORRECTION", "false")


def test_feature_flag_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SATQUERY_SAR_TRANSLATION_ENABLED", raising=False)
    assert not translation_enabled()
    assert SarTranslationService().health_payload()["status"] == "disabled"


def test_selected_model_and_color_flags(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_MODEL", "pix2pix")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_USE_COLOR_CORRECTION", "true")
    service = SarTranslationService(model_factories={"pix2pix": FakePix2Pix}, color_factory=FakeColorCorrector)
    health = service.health_payload()
    assert health["selected_model"] == "pix2pix"
    assert health["color_corrector_available"] is True


def test_eligibility_distinguishes_verified_pair_and_preview() -> None:
    raster = np.ones((16, 16, 2), dtype=np.float32)
    image = Image.new("RGB", (16, 16))
    verified = evaluate_translation_eligibility(raster, metadata(), image, "sarfusionformer")
    assert verified.eligible and verified.selected_model == "sarfusionformer"
    preview_metadata = metadata(bands=1, modality=ImageModality.SAR_PREVIEW, representation=RepresentationType.DISPLAY_PREVIEW)
    preview = evaluate_translation_eligibility(raster[:, :, :1], preview_metadata, image, "sarfusionformer")
    assert preview.eligible and preview.selected_model == "pix2pix"
    assert preview.reason and "verified VV/VH" in preview.reason


def test_unsupported_and_non_finite_inputs_are_rejected() -> None:
    image = Image.new("RGB", (16, 16))
    unsupported = metadata(bands=3, modality=ImageModality.SAR_PREVIEW, representation=RepresentationType.DISPLAY_PREVIEW)
    decision = evaluate_translation_eligibility(np.ones((16, 16, 3)), unsupported, image, "sarfusionformer")
    assert decision.eligible and decision.selected_model == "pix2pix"
    invalid = evaluate_translation_eligibility(np.full((16, 16, 2), np.nan), metadata(), image, "sarfusionformer")
    assert not invalid.eligible


def test_lazy_load_reuse_dimensions_and_output_range(enabled: None) -> None:
    model = FakeSarFusionFormer()
    factory_calls = []
    service = SarTranslationService(model_factories={"sarfusionformer": lambda: factory_calls.append(1) or model})
    raster = np.random.default_rng(4).normal(size=(16, 16, 2)).astype(np.float32)
    image = Image.new("RGB", (16, 16))
    first = service.translate(raster, metadata(), image)
    second = service.translate(raster, metadata(), image)
    assert len(factory_calls) == 1
    assert model.calls == 2
    assert (first.width, first.height) == (256, 256)
    assert 0 <= first.output_value_range[0] <= first.output_value_range[1] <= 1
    assert service.health_payload()["load_count"] == 1
    assert service.health_payload()["reuse_count"] >= 1
    first.image.close()
    second.image.close()


def test_primary_failure_uses_pix2pix_fallback(enabled: None) -> None:
    pix = FakePix2Pix()
    service = SarTranslationService(model_factories={
        "sarfusionformer": lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
        "pix2pix": lambda: pix,
    })
    raster = np.random.default_rng(5).normal(size=(16, 16, 2)).astype(np.float32)
    result = service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert result.model_used == "Pix2Pix"
    assert result.fallback_used
    assert pix.calls == 1
    result.image.close()


def test_color_correction_success_and_safe_failure(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_USE_COLOR_CORRECTION", "true")
    raster = np.random.default_rng(6).normal(size=(16, 16, 2)).astype(np.float32)
    service = SarTranslationService(model_factories={"sarfusionformer": FakeSarFusionFormer}, color_factory=FakeColorCorrector)
    result = service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert result.color_correction_used
    result.image.close()

    broken = SarTranslationService(
        model_factories={"sarfusionformer": FakeSarFusionFormer},
        color_factory=lambda: (_ for _ in ()).throw(RuntimeError("bad corrector")),
    )
    fallback = broken.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert not fallback.color_correction_used
    assert any("failed safely" in warning for warning in fallback.warnings)
    fallback.image.close()


def test_failure_is_sticky_and_retry_is_explicit(enabled: None) -> None:
    service = SarTranslationService(model_factories={
        "sarfusionformer": lambda: (_ for _ in ()).throw(RuntimeError("bad")),
        "pix2pix": lambda: (_ for _ in ()).throw(RuntimeError("bad")),
    })
    raster = np.random.default_rng(7).normal(size=(16, 16, 2)).astype(np.float32)
    with pytest.raises(SarTranslationError):
        service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert service.state == "failed"
    with pytest.raises(SarTranslationError, match="remains unavailable"):
        service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    service.retry()
    assert service.state == "unloaded"


def test_registered_instance_is_reused_without_factory(enabled: None) -> None:
    model = FakeSarFusionFormer()
    service = SarTranslationService(model_factories={"sarfusionformer": lambda: pytest.fail("factory must not run")})
    service.register_external_models(sarfusionformer=model, device="cpu")
    raster = np.random.default_rng(8).normal(size=(16, 16, 2)).astype(np.float32)
    result = service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert model.calls == 1
    assert service.health_payload()["load_count"] == 0
    result.image.close()


def test_registered_host_load_failures_are_not_retried(enabled: None) -> None:
    service = SarTranslationService()
    service.register_external_models(
        sarfusionformer_error="host load failed", pix2pix_error="host load failed", device="cpu",
    )
    raster = np.random.default_rng(81).normal(size=(16, 16, 2)).astype(np.float32)
    with pytest.raises(SarTranslationError):
        service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
    assert service.health_payload()["load_count"] == 0


def test_inference_lock_serializes_threads(enabled: None) -> None:
    model = FakeSarFusionFormer(delay=0.02)
    service = SarTranslationService(model_factories={"sarfusionformer": lambda: model})
    raster = np.random.default_rng(9).normal(size=(16, 16, 2)).astype(np.float32)
    outputs = []

    def worker() -> None:
        result = service.translate(raster, metadata(), Image.new("RGB", (16, 16)))
        outputs.append(result)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outputs) == 3
    assert model.maximum_active == 1
    for output in outputs:
        output.image.close()


def _published_product() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="run:test", source_observation_id="test", source_modality=ImageModality.SAR_VV_VH,
        evidence_type="generated_optical_like", evidence_run_id="run", generator="Pix2Pix",
        status=EvidenceLifecycleState.SUCCEEDED, type="generated_optical_like_representation",
        label="Generated optical-like supporting evidence", reference="/api/agent/previews/00000000000000000000000000000000.png",
    )


def test_fusion_states_and_required_disclosures() -> None:
    translation = {"model_used": "Pix2Pix", "width": 256, "height": 256, "warnings": [], "provenance": {}}
    translation_only = fuse_sar_optical_evidence(
        query="Locate buildings", translation=translation,
        optical_evidence={"caption": "buildings beside roads"}, native_water=None, native_scene=None,
        generation_state=EvidenceLifecycleState.SUCCEEDED,
        semantic_comparison_state=EvidenceLifecycleState.SUCCEEDED, evidence_products=[_published_product()],
    )
    assert translation_only.agreement == "suggested_by_translation_only"
    assert "generated from sar by pix2pix" in translation_only.disclosure.lower()
    unavailable = fuse_sar_optical_evidence(
        query="Describe this SAR", translation=None, optical_evidence=None,
        native_water=None, native_scene=None, translation_error="checkpoint missing",
    )
    assert unavailable.agreement == "translation_unavailable"
    assert unavailable.status == "translation_unavailable"


@pytest.mark.parametrize(
    ("native_water", "optical", "expected"),
    [
        (True, {"vqa": {"question": "Is water present?", "answer": "yes"}}, "supported_by_both"),
        (False, {"vqa": {"question": "Is water present?", "answer": "yes"}}, "conflicting_evidence"),
        (True, {}, "supported_by_native_sar_only"),
    ],
)
def test_water_fusion_agreement_rules(native_water: bool, optical: dict, expected: str) -> None:
    water = SimpleNamespace(
        water_detected=native_water, image_area_percent=12.0, input_quality_score=0.9,
    )
    result = fuse_sar_optical_evidence(
        query="Is water present?",
        translation={"model_used": "Pix2Pix", "width": 256, "height": 256, "warnings": [], "provenance": {}},
        optical_evidence=optical,
        native_water=water,
        native_scene=None,
        generation_state=EvidenceLifecycleState.SUCCEEDED,
        semantic_comparison_state=EvidenceLifecycleState.SUCCEEDED, evidence_products=[_published_product()],
    )
    assert result.agreement == expected
    if expected == "conflicting_evidence":
        assert result.direct_answer.startswith("The two analysis branches disagree.")


def test_fusion_insufficient_evidence() -> None:
    result = fuse_sar_optical_evidence(
        query="Assess this", translation={"model_used": "Pix2Pix", "warnings": [], "provenance": {}},
        optical_evidence={}, native_water=None, native_scene=None,
    )
    assert result.agreement == "insufficient_evidence"


def _ingested() -> IngestedImage:
    raster = np.random.default_rng(11).normal(size=(16, 16, 2)).astype(np.float32)
    return IngestedImage(
        metadata=metadata(), durations_ms={}, model_image=Image.new("RGB", (16, 16)),
        image_representation="test", bands_used=["VV", "VH"], analysis_raster=raster,
        content_hash="a" * 64, source_bytes=b"test",
    )


def _native_scene():
    return SimpleNamespace(
        valid_pixel_percent=100.0, low_backscatter_percent=30.0, texture_index=0.4,
        input_quality_score=0.9,
    )


def _translation_result(image: Image.Image):
    return SimpleNamespace(
        image=image, width=256, height=256, model_used="Pix2Pix", fallback_used=False,
        fallback_reason=None, color_correction_used=False, device="cpu", runtime_ms=3,
        stage_durations_ms={"sar_translation_inference": 3}, preprocessing_method="test",
        input_channel_interpretation="test", output_value_range=[0.0, 1.0], warnings=[],
        provenance={"generated_representation": True}, artifact_url="/api/agent/previews/00000000000000000000000000000000.png",
        normalized_sar_preview_url=None, color_corrected_artifact_url=None, content_hash="b" * 64,
    )


def _patch_orchestrator_base(monkeypatch: pytest.MonkeyPatch, image: Image.Image) -> None:
    monkeypatch.setattr(translated_module, "evaluate_translation_eligibility", lambda *args, **kwargs: SimpleNamespace(
        eligible=True, requested_model="pix2pix", selected_model="pix2pix", band_count=2,
        finite_value_ratio=1.0, reason=None,
    ))
    monkeypatch.setattr(translated_module, "get_sar_translation_service", lambda: SimpleNamespace(
        translate=lambda *args: _translation_result(image),
    ))
    monkeypatch.setattr(translated_module, "get_sve_service", lambda: SimpleNamespace(enabled=False))


def test_captioner_receives_generated_rgb_only_for_description(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = Image.new("RGB", (256, 256))
    closed = []
    original_close = generated.close
    generated.close = lambda: (closed.append(True), original_close())  # type: ignore[method-assign]
    _patch_orchestrator_base(monkeypatch, generated)
    calls = []
    monkeypatch.setattr(translated_module, "get_captioner", lambda: SimpleNamespace(
        describe=lambda image, *args: calls.append((image.mode, image.size)) or SimpleNamespace(caption="generated roads"),
    ))
    monkeypatch.setattr(translated_module, "get_grounder", lambda: pytest.fail("grounder should not run"))
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _: None)
    run = translated_module.run_sar_translated_optical_evidence(
        ingested=_ingested(), query="Describe the scene", routed_task=TaskType.SAR_SCENE_ANALYSIS,
        native_water=None, native_scene=_native_scene(), evidence_run_id="run",
    )
    assert calls == [("RGB", (256, 256))]
    assert run.analysis.optical_specialists_executed == ["rs_captioner"]
    assert closed == [True]


def test_caption_failure_does_not_relabel_successful_pix2pix_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = Image.new("RGB", (256, 256))
    _patch_orchestrator_base(monkeypatch, generated)
    monkeypatch.setattr(translated_module, "get_captioner", lambda: SimpleNamespace(
        describe=lambda *_args: (_ for _ in ()).throw(CaptionerError("CAPTION_FAILED", "Caption unavailable.")),
    ))
    monkeypatch.setattr(translated_module, "get_grounder", lambda: pytest.fail("grounder should not run"))
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _: None)
    run = translated_module.run_sar_translated_optical_evidence(
        ingested=_ingested(), query="Describe the scene", routed_task=TaskType.SAR_SCENE_ANALYSIS,
        native_water=None, native_scene=_native_scene(), evidence_run_id="run",
    )
    assert run.analysis.generation_state == EvidenceLifecycleState.SUCCEEDED
    assert run.analysis.semantic_comparison_state == EvidenceLifecycleState.FAILED
    assert len(run.analysis.evidence_products) == 1
    assert run.analysis.translated_findings == []
    assert run.analysis.agreement == "supported_by_native_sar_only"
    states = {item["tool"]: item["status"] for item in run.steps}
    assert states["sar_translation_inference"] == "success"
    assert states["translated_optical_captioning"] == "failed"
    assert states["translation_semantic_comparison"] == "failed"


def test_grounder_dispatches_only_for_grounding_query(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = Image.new("RGB", (256, 256))
    _patch_orchestrator_base(monkeypatch, generated)
    calls = []
    grounding = {"target_phrase": "building", "accepted_detection_count": 1, "annotated_preview_url": None}
    monkeypatch.setattr(translated_module, "get_grounder", lambda: SimpleNamespace(
        ground=lambda image, *args: calls.append(image.mode) or SimpleNamespace(model_dump=lambda **_: grounding),
    ))
    monkeypatch.setattr(translated_module, "get_captioner", lambda: pytest.fail("captioner should not run"))
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _: None)
    run = translated_module.run_sar_translated_optical_evidence(
        ingested=_ingested(), query="Locate the buildings", routed_task=TaskType.GROUNDING,
        native_water=None, native_scene=_native_scene(), evidence_run_id="run",
    )
    assert calls == ["RGB"]
    assert run.analysis.optical_grounding == grounding


def test_rsvqa_dispatches_only_for_supported_question(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = Image.new("RGB", (256, 256))
    _patch_orchestrator_base(monkeypatch, generated)
    calls = []
    monkeypatch.setattr(translated_module, "classify_rsvqa_task", lambda _: "presence")
    monkeypatch.setattr(translated_module, "get_rsvqa_specialist", lambda: SimpleNamespace(
        predict=lambda image, question, content_hash: calls.append((image.mode, question)) or {
            "answer": "yes", "confidence": 0.7, "task": "presence", "logits": [0.0, 1.0],
            "probabilities": [0.3, 0.7], "model_used": "RSVQA Specialist v1",
        },
    ))
    monkeypatch.setattr(translated_module, "get_captioner", lambda: pytest.fail("captioner should not run"))
    monkeypatch.setattr(translated_module, "get_grounder", lambda: pytest.fail("grounder should not run"))
    run = translated_module.run_sar_translated_optical_evidence(
        ingested=_ingested(), query="Is there a building?", routed_task=TaskType.VQA,
        native_water=None, native_scene=_native_scene(), evidence_run_id="run",
    )
    assert calls == [("RGB", "Is there a building?")]
    assert run.analysis.optical_vqa and run.analysis.optical_vqa["answer"] == "yes"
    assert run.analysis.rsvqa_disclosure is not None
