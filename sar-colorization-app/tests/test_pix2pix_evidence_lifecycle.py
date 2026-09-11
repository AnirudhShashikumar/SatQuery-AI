"""End-to-end contracts for authoritative Pix2Pix evidence publication."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import satquery_agent.api as api_module
from backend import app
from satquery_agent.evidence_lifecycle import normalize_translation_evidence
from satquery_agent.image_ingestion import save_preview
from satquery_agent.models import AgentResponse, EvidenceItem, EvidenceLifecycleState
from satquery_agent.reporting import MISSION_STORE
from satquery_agent.specialists.sar_optical_fusion import fuse_sar_optical_evidence
from satquery_agent.specialists.sar_translated_optical import SarTranslatedEvidenceRun


def _png() -> bytes:
    rng = np.random.default_rng(917)
    values = np.clip(rng.gamma(3.5, 32.0, (72, 72)), 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(values).save(output, format="PNG")
    return output.getvalue()


def _request(client: TestClient, *, use_cache: bool = False):
    return client.post(
        "/api/agent/query",
        data={
            "query": "Describe this SAR scene.",
            "input_mode": "single",
            "primary_modality": "sar",
            "primary_image_modality": "sar_preview",
            "use_cache": str(use_cache).lower(),
        },
        files={"primary_image": ("sar.png", _png(), "image/png")},
    )


@pytest.fixture(autouse=True)
def _clear_store(monkeypatch: pytest.MonkeyPatch):
    MISSION_STORE.clear()
    monkeypatch.setenv("SATQUERY_SAR_TRANSLATION_ENABLED", "1")
    yield
    MISSION_STORE.clear()


def _fake_run(mode: str, preview_url: str):
    def run(**kwargs):
        run_id = kwargs["evidence_run_id"]
        source_id = kwargs["ingested"].metadata.file_id
        if mode == "failed":
            analysis = fuse_sar_optical_evidence(
                query=kwargs["query"], translation=None, optical_evidence=None,
                native_water=kwargs["native_water"], native_scene=kwargs["native_scene"],
                translation_error="Pix2Pix inference was unavailable.",
                generation_state=EvidenceLifecycleState.FAILED,
            )
            return SarTranslatedEvidenceRun(analysis=analysis, native_scene=kwargs["native_scene"], steps=[
                {"tool": "sar_translation_inference", "status": "failed", "duration_ms": 2, "parameters": {"error_code": "PIX2PIX_FAILED"}},
                {"tool": "translation_semantic_comparison", "status": "skipped", "duration_ms": 0, "parameters": {"reason": "generation_not_succeeded"}},
            ])
        product = EvidenceItem(
            evidence_id=f"{run_id}:pix2pix:evidence", source_observation_id=source_id,
            source_modality=kwargs["ingested"].metadata.effective_modality,
            evidence_type="generated_optical_like", evidence_run_id=run_id, generator="Pix2Pix",
            status=EvidenceLifecycleState.SUCCEEDED, type="generated_optical_like_representation",
            label="Generated optical-like supporting evidence",
            description="Learned optical-like representation generated from SAR; not observed optical imagery and not ground truth.",
            reference=preview_url,
        )
        semantic_state = EvidenceLifecycleState.FAILED if mode == "semantic_failed" else EvidenceLifecycleState.SUCCEEDED
        analysis = fuse_sar_optical_evidence(
            query=kwargs["query"],
            translation={
                "model_used": "Pix2Pix", "device": "cpu", "runtime_ms": 3, "width": 32, "height": 32,
                "artifact_url": preview_url, "warnings": [], "provenance": {"generated_representation": True},
            },
            optical_evidence={"caption": "translation suggests buildings"},
            native_water=kwargs["native_water"], native_scene=kwargs["native_scene"],
            generation_state=EvidenceLifecycleState.SUCCEEDED,
            semantic_comparison_state=semantic_state,
            evidence_products=[product],
        )
        return SarTranslatedEvidenceRun(analysis=analysis, native_scene=kwargs["native_scene"], steps=[
            {"tool": "sar_translation_inference", "status": "success", "duration_ms": 2, "parameters": {"model": "Pix2Pix"}},
            {"tool": "translation_semantic_comparison", "status": "failed" if semantic_state == EvidenceLifecycleState.FAILED else "success", "duration_ms": 1, "parameters": {}},
        ])
    return run


def test_success_publishes_one_run_scoped_product_and_cache_preserves_it(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = save_preview(Image.new("RGB", (32, 32), (20, 80, 160)))
    calls = []
    run = _fake_run("success", preview)
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", lambda **kwargs: calls.append(1) or run(**kwargs))
    client = TestClient(app)
    first = _request(client, use_cache=True).json()
    second = _request(client, use_cache=True).json()
    assert len(calls) == 1
    assert second["cache"]["cached"] is True
    assert first["request_id"] == second["request_id"]
    translated = second["sar_translated_optical_analysis"]
    assert translated["generation_state"] == "SUCCEEDED"
    assert translated["semantic_comparison_state"] == "SUCCEEDED"
    assert translated["evidence_products"][0]["evidence_run_id"] == second["request_id"]
    assert translated["evidence_products"][0]["source_observation_id"] == second["primary_image_metadata"]["file_id"]
    assert any(item["evidence_type"] == "generated_optical_like" for item in second["evidence"])


def test_generation_failure_has_no_product_or_translation_claim_and_native_result_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", _fake_run("failed", "/stale.png"))
    client = TestClient(app)
    body = _request(client).json()
    translated = body["sar_translated_optical_analysis"]
    assert body["status"] == "partial"
    assert body["result_status"] == "COMPLETED_WITH_LIMITATIONS"
    assert translated["generation_state"] == "FAILED"
    assert translated["generated_preview_url"] is None
    assert translated["evidence_products"] == []
    assert not any(item.get("generator") == "Pix2Pix" for item in body["evidence"])
    assert "translation suggests" not in body["answer"].lower()
    assert translated["agreement"] == "translation_unavailable"
    assert body["sar_scene_analysis"] is not None
    report = client.post("/api/agent/report", json={"request_id": body["request_id"], "formats": ["json", "zip"]}).json()
    artifacts = {item["format"]: item for item in report["artifacts"]}
    document = client.get(artifacts["json"]["url"]).json()
    assert not any(item.get("generator") == "Pix2Pix" for item in document["evidence"]["items"])
    package = zipfile.ZipFile(io.BytesIO(client.get(artifacts["zip"]["url"]).content))
    failed_manifest = json.loads(package.read("evidence/manifest.json"))
    assert not any(item.get("generator") == "Pix2Pix" for item in failed_manifest["evidence_products"])


def test_semantic_failure_keeps_generated_product_but_excludes_failed_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = save_preview(Image.new("RGB", (32, 32), (20, 80, 160)))
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", _fake_run("semantic_failed", preview))
    body = _request(TestClient(app)).json()
    translated = body["sar_translated_optical_analysis"]
    assert translated["generation_state"] == "SUCCEEDED"
    assert translated["semantic_comparison_state"] == "FAILED"
    assert len(translated["evidence_products"]) == 1
    assert translated["generated_preview_url"] == preview
    assert translated["translated_findings"] == []
    assert "translation suggests" not in body["answer"].lower()
    assert translated["agreement"] == "supported_by_native_sar_only"


def test_stale_or_legacy_product_is_suppressed_instead_of_reinterpreted(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = save_preview(Image.new("RGB", (32, 32)))
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", _fake_run("success", preview))
    body = _request(TestClient(app)).json()
    body["sar_translated_optical_analysis"]["evidence_products"][0]["evidence_run_id"] = "prior-run"
    stale = normalize_translation_evidence(AgentResponse.model_validate(body))
    assert stale.sar_translated_optical_analysis.generated_preview_url is None
    assert stale.sar_translated_optical_analysis.evidence_products == []
    assert not any(item.generator == "Pix2Pix" for item in stale.evidence)
    assert "translation suggests" not in (stale.answer or "").lower()


def test_json_pdf_and_zip_use_the_same_published_product(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = save_preview(Image.new("RGB", (32, 32), (20, 80, 160)))
    monkeypatch.setattr(api_module, "run_sar_translated_optical_evidence", _fake_run("success", preview))
    client = TestClient(app)
    result = _request(client).json()
    report = client.post("/api/agent/report", json={"request_id": result["request_id"], "formats": ["pdf", "json", "zip"]})
    assert report.status_code == 200, report.text
    artifacts = {item["format"]: item for item in report.json()["artifacts"]}
    document = client.get(artifacts["json"]["url"]).json()
    published = [item for item in document["evidence"]["items"] if item.get("generator") == "Pix2Pix"]
    previews = [item for item in document["evidence"]["preview_products"] if item["url"] == preview]
    assert len(published) == len(previews) == 1
    pdf = client.get(artifacts["pdf"]["url"]).content
    assert pdf.startswith(b"%PDF") and len(pdf) > 10_000
    package = zipfile.ZipFile(io.BytesIO(client.get(artifacts["zip"]["url"]).content))
    evidence_names = [name for name in package.namelist() if name.startswith("evidence/") and "Generated_optical-like_supporting_evidence" in name]
    assert len(evidence_names) == 1
    manifest = json.loads(package.read("evidence/manifest.json"))
    assert [item for item in manifest["evidence_products"] if item.get("generator") == "Pix2Pix"] == published
    packaged = json.loads(package.read("report/mission-report.json"))
    assert packaged["evidence"]["items"] == document["evidence"]["items"]
