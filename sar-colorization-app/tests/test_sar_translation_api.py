"""API compatibility and opt-in orchestration tests for translated SAR evidence."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import satquery_agent.api as api_module
from backend import app
from satquery_agent.models import EvidenceItem, EvidenceLifecycleState, ImageModality
from satquery_agent.specialists.sar_optical_fusion import fuse_sar_optical_evidence
from satquery_agent.specialists.sar_translated_optical import SarTranslatedEvidenceRun


def _png() -> bytes:
    values = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


def _query(client: TestClient):
    return client.post(
        "/api/agent/query",
        data={
            "query": "Describe the major radar-backscatter patterns.",
            "input_mode": "single",
            "primary_modality": "sar",
            "primary_image_modality": "sar_preview",
            "use_cache": "false",
        },
        files={"primary_image": ("sample.png", _png(), "image/png")},
    )


def test_disabled_path_has_no_optional_field_or_translation_steps(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "0")
    response = _query(TestClient(app))
    assert response.status_code == 200
    body = response.json()
    assert body["sar_translated_optical_analysis"] is None
    assert not any(step["tool"].startswith("sar_translation_") for step in body["execution"]["steps"])


def test_enabled_path_publishes_disclosure_and_fused_answer(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "1")

    def fake_run(**kwargs):
        product = EvidenceItem(
            evidence_id=f'{kwargs["evidence_run_id"]}:pix2pix', source_observation_id=kwargs["ingested"].metadata.file_id,
            source_modality=ImageModality.SAR_PREVIEW, evidence_type="generated_optical_like",
            evidence_run_id=kwargs["evidence_run_id"], generator="Pix2Pix", status=EvidenceLifecycleState.SUCCEEDED,
            type="generated_optical_like_representation", label="Generated optical-like supporting evidence",
            reference="/api/agent/previews/00000000000000000000000000000000.png",
        )
        analysis = fuse_sar_optical_evidence(
            query=kwargs["query"],
            translation={
                "model_used": "Pix2Pix", "device": "cpu", "runtime_ms": 4, "width": 256, "height": 256,
                "artifact_url": "/api/agent/previews/00000000000000000000000000000000.png",
                "fallback_used": True, "fallback_reason": "VV/VH pair unavailable", "warnings": [],
                "provenance": {"generated_representation": True}, "stage_durations_ms": {"sar_translation_inference": 4},
            },
            optical_evidence={"caption": "a generated scene with roads", "specialists_executed": ["rs_captioner"]},
            native_water=kwargs["native_water"], native_scene=kwargs["native_scene"],
            generation_state=EvidenceLifecycleState.SUCCEEDED,
            semantic_comparison_state=EvidenceLifecycleState.SUCCEEDED,
            evidence_products=[product],
        )
        return SarTranslatedEvidenceRun(
            analysis=analysis,
            native_scene=kwargs["native_scene"],
            steps=[{"tool": "sar_translation_inference", "status": "success", "duration_ms": 4, "parameters": {"model": "Pix2Pix"}}],
        )

    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", fake_run)
    response = _query(TestClient(app))
    assert response.status_code == 200, response.text
    body = response.json()
    translated = body["sar_translated_optical_analysis"]
    assert translated["model"] == "Pix2Pix"
    assert translated["fallback_used"] is True
    assert "not an observed optical image" in translated["disclosure"]
    assert body["answer"] == translated["direct_answer"]
    assert any(step["tool"] == "sar_translation_inference" for step in body["execution"]["steps"])


def test_unexpected_optional_failure_retains_native_result(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "1")
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", lambda **_: (_ for _ in ()).throw(RuntimeError("private details")))
    response = _query(TestClient(app))
    assert response.status_code == 200
    body = response.json()
    assert body["sar_scene_analysis"] is not None
    assert body["sar_translated_optical_analysis"]["generation_state"] == "FAILED"
    assert body["sar_translated_optical_analysis"]["evidence_products"] == []
    assert body["result_status"] == "COMPLETED_WITH_LIMITATIONS"
    assert any("failed safely" in warning for warning in body["warnings"])
    assert "private details" not in response.text


def test_health_reports_translation_configuration(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "1")
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_MODEL", "pix2pix")
    body = TestClient(app).get("/api/agent/health").json()
    health = body["specialists"]["sar_translation_service"]
    assert health["enabled"] is True
    assert health["selected_model"] == "pix2pix"
    assert health["status"] in {"unloaded", "ready", "failed"}
    assert "fallback_available" in health
    assert "color_corrector_available" in health
