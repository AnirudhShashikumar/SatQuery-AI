from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.models import (
    SVECaptionConsistency,
    SVEResult,
    SVEScenePrior,
    SVEVQAConsistency,
    SpecialistHealth,
)
from satquery_agent.reporting import MISSION_STORE
from satquery_agent.services.sve_service import SVECall


def optical_png() -> bytes:
    values = np.zeros((64, 64, 3), dtype=np.uint8)
    values[:32] = (45, 140, 60)
    values[32:] = (25, 55, 100)
    output = io.BytesIO()
    Image.fromarray(values).save(output, format="PNG")
    return output.getvalue()


class FakeSVE:
    enabled = True

    def health(self):
        return SpecialistHealth(status="ready", device="cpu")

    def analyze(self, _image, _content_hash, *, captions=(), vqa_category=None, vqa_answer=None, grounding_target=None, top_k=5):
        caption = SVECaptionConsistency(
            score=.27, selected_candidate_index=0, original_candidates=list(captions),
            original_candidate_order=list(range(len(captions))), reranked_candidate_order=list(range(len(captions))),
            candidate_scores=[.27] * len(captions), reranked=False,
        ) if captions else None
        vqa = SVEVQAConsistency(
            state="disagreement", target_concept="inland water", similarity=.11,
            explanation="The VQA answer and scene-level embedding evidence disagree.",
        ) if vqa_category and vqa_answer else None
        return SVECall(
            result=SVEResult(
                available=True, status="success", device="cpu", runtime_ms=12,
                scene_priors=[SVEScenePrior(label="vegetation", similarity=.31)],
                caption_consistency=caption, vqa_consistency=vqa,
                limitations=["Scene-level only."],
            ),
            trace=[
                {"tool": "sve_artifact_verification", "status": "success", "duration_ms": 2, "parameters": {"checksum_verified": True}},
                {"tool": "sve_vqa_consistency", "status": "success", "duration_ms": 1, "parameters": {"consistency_state": "disagreement"}},
            ],
        )


def test_vqa_sve_evidence_and_all_reports_are_safe(monkeypatch) -> None:
    MISSION_STORE.clear()
    fake = FakeSVE()
    monkeypatch.setenv("SVE_ENABLED", "true")
    with patch("satquery_agent.api.get_sve_service", return_value=fake):
        client = TestClient(app)
        response = client.post(
            "/api/agent/query",
            data={"query": "Is a water body visible?", "input_mode": "single", "primary_modality": "optical"},
            files={"primary_image": ("scene.png", optical_png(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["vqa_details"]["supported"] is True
    assert body["sve_result"]["vqa_consistency"]["state"] == "disagreement"
    assert "scene-level embedding evidence disagree" in " ".join(body["warnings"])
    assert any(step["tool"] == "sve_artifact_verification" for step in body["execution"]["steps"])
    serialized = json.dumps(body)
    assert "embedding" not in body["sve_result"]
    assert "/Users/" not in serialized

    report = client.post(
        "/api/agent/report",
        json={"request_id": body["request_id"], "formats": ["pdf", "json", "csv", "zip"]},
    )
    assert report.status_code == 200
    artifacts = {item["format"]: item for item in report.json()["artifacts"]}
    document = client.get(artifacts["json"]["url"]).json()
    assert document["specialist_provenance"]["satquery_vision_encoder"]["backbone"] == "OpenCLIP ViT-L-14"
    assert document["statistics"]["sve_scene_evidence"]["top_scene_priors"][0]["label"] == "vegetation"
    assert client.get(artifacts["pdf"]["url"]).content.startswith(b"%PDF")
    assert b"sve_scene_evidence" in client.get(artifacts["csv"]["url"]).content
    package = zipfile.ZipFile(io.BytesIO(client.get(artifacts["zip"]["url"]).content))
    assert "report/mission-report.json" in package.namelist()
