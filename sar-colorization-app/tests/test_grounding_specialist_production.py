"""Production integration tests for Grounding Specialist v1.1 proposal rescoring."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.models import ImageFormat, ImageMetadata, Modality, SpecialistHealth
from satquery_agent.specialists.grounder import (
    PROVENANCE_CHECKPOINT,
    RemoteSensingGrounder,
    apply_grounding_score_policy,
    specialist_scores_for_accepted_proposals,
)
from satquery_agent.specialists.grounding_specialist import (
    CHECKPOINT_STEP,
    EXPECTED_CHECKPOINT_SHA256,
    GroundingSpecialist,
    GroundingSpecialistError,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "grounding_specialist_v1_1"


def metadata() -> ImageMetadata:
    return ImageMetadata(
        file_id="specialist-test",
        original_name="scene.png",
        safe_name="scene.png",
        format=ImageFormat.PNG,
        mime_type="image/png",
        size_bytes=100,
        width=100,
        height=100,
        band_count=3,
        dtype="uint8",
        is_georeferenced=False,
        color_interpretation=["r", "g", "b"],
    )


class ProposalProcessor:
    def __call__(self, *, images: Image.Image, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.zeros((1, 3, 32, 48)),
            "input_ids": torch.ones((1, 4), dtype=torch.long),
        }

    def post_process_grounded_object_detection(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [{
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]]),
            "scores": torch.tensor([0.74, 0.56]),
            "labels": ["building", "building"],
        }]


class ProposalModel:
    def eval(self) -> "ProposalModel":
        return self

    def to(self, device: str) -> "ProposalModel":
        self.device = device
        return self

    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=torch.ones((1, 2, 256)),
            pred_boxes=torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]]),
            logits=torch.tensor([[[2.0, -2.0], [1.0, -2.0]]]),
        )


class StubSpecialist:
    def __init__(self, scores: list[float] | None = None, *, fail: bool = False) -> None:
        self.scores = scores or [0.91, 0.81]
        self.fail = fail
        self.load_calls = 0
        self.score_calls = 0
        self.state = "unloaded"

    def load(self, _device: str) -> None:
        self.load_calls += 1
        if self.fail:
            self.state = "failed"
            raise GroundingSpecialistError("test load failure")
        self.state = "ready"

    def score_proposals(self, _hidden: torch.Tensor, _boxes: torch.Tensor) -> torch.Tensor:
        self.score_calls += 1
        if self.fail:
            raise GroundingSpecialistError("test score failure")
        return torch.tensor([self.scores])

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(
            status=self.state,
            device="cpu" if self.state == "ready" else None,
            error="Checkpoint loading failed." if self.state == "failed" else None,
        )

    def reset_for_tests(self) -> None:
        self.state = "unloaded"


def run_grounder(specialist: StubSpecialist) -> tuple[RemoteSensingGrounder, object]:
    grounder = RemoteSensingGrounder(grounding_specialist=specialist)  # type: ignore[arg-type]
    image = Image.new("RGB", (100, 100))
    try:
        with patch.object(grounder, "_load_components", return_value=(ProposalProcessor(), ProposalModel())):
            result = grounder.ground(
                image,
                "Locate the buildings.",
                metadata(),
                Modality.OPTICAL,
                ["r", "g", "b"],
                "RGB",
            )
    finally:
        image.close()
    return grounder, result


def test_verified_specialist_checkpoint_loads_strictly_once() -> None:
    specialist = GroundingSpecialist(model_dir=MODEL_DIR)
    assert specialist.health().status == "unloaded"
    specialist.load("cpu")
    assert specialist.health().status == "ready"
    assert specialist.checkpoint_hash == EXPECTED_CHECKPOINT_SHA256
    assert specialist.load_count == 1
    scores = specialist.score_proposals(
        torch.zeros((1, 3, 256)),
        torch.full((1, 3, 4), 0.25),
    )
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores).all()
    specialist.load("cpu")
    assert specialist.load_count == 1


def test_specialist_scores_follow_original_postprocessor_query_mask() -> None:
    outputs = SimpleNamespace(
        logits=torch.tensor([[[2.0, -2.0], [-2.0, -2.0], [1.0, -2.0]]]),
    )
    selected = specialist_scores_for_accepted_proposals(
        outputs,
        torch.tensor([[0.91, 0.51, 0.81]]),
        box_threshold=0.35,
        expected_count=2,
    )
    assert torch.allclose(selected, torch.tensor([0.91, 0.81]))


def test_rescored_confidence_is_used_without_changing_boxes_or_labels() -> None:
    specialist = StubSpecialist([0.91, 0.81])
    grounder, result = run_grounder(specialist)
    assert specialist.load_calls == 1
    assert specialist.score_calls == 1
    assert [item.score for item in result.detections] == [0.91, 0.81]
    assert [item.bbox_pixels for item in result.detections] == [[10, 10, 30, 30], [60, 60, 80, 80]]
    assert [item.label for item in result.detections] == ["building", "building"]
    assert result.confidence.score == 0.91
    assert "VRSBench-trained" in result.confidence.reason
    assert result.model.checkpoint == PROVENANCE_CHECKPOINT
    assert result.model.adaptation_dataset == "VRSBench"
    assert result.model.remote_sensing_adapted is True
    assert grounder.grounding_specialist_health().status == "ready"


def test_disabled_mode_exactly_preserves_dino_and_never_runs_specialist(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "disabled")
    specialist = StubSpecialist([0.01, 0.02])
    grounder, result = run_grounder(specialist)
    assert specialist.load_calls == 0
    assert specialist.score_calls == 0
    assert [item.score for item in result.detections] == [0.74, 0.56]
    assert [item.bbox_pixels for item in result.detections] == [[10, 10, 30, 30], [60, 60, 80, 80]]
    assert [item.label for item in result.detections] == ["building", "building"]
    assert grounder.grounding_specialist_health().status == "disabled"
    assert all(item["score_mode"] == "disabled" for item in grounder.last_score_diagnostics)


def test_fallback_tiny_scores_cannot_suppress_dino_detections(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "fallback")
    grounder, result = run_grounder(StubSpecialist([0.005, 0.04]))
    assert [item.score for item in result.detections] == [0.74, 0.56]
    assert len(result.detections) == 2
    assert all(item["fallback_used"] for item in grounder.last_score_diagnostics)
    assert {item["fallback_reason"] for item in grounder.last_score_diagnostics} == {"specialist_scores_extremely_small"}


def test_fallback_nonfinite_and_out_of_range_scores_restore_dino() -> None:
    for scores, reason in (([math.nan, 0.8], "specialist_scores_non_finite"), ([1.2, 0.8], "specialist_distribution_outside_probability_range")):
        decision = apply_grounding_score_policy([0.74, 0.56], scores, "fallback")
        assert decision.scores == [0.74, 0.56]
        assert decision.fallback_used is True
        assert decision.fallback_reason == reason


def test_blend_uses_normalized_probabilities_without_changing_proposal_geometry(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "blend")
    monkeypatch.setenv("SATQUERY_GROUNDING_DINO_WEIGHT", "0.70")
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_WEIGHT", "0.30")
    grounder, result = run_grounder(StubSpecialist([0.91, 0.81]))
    assert [item.bbox_pixels for item in result.detections] == [[10, 10, 30, 30]]
    assert [item.label for item in result.detections] == ["building"]
    assert result.detections[0].score == pytest.approx(0.818)
    diagnostics = grounder.last_score_diagnostics
    assert [item["original_grounding_score"] for item in diagnostics] == pytest.approx([0.74, 0.56])
    assert [item["specialist_score"] for item in diagnostics] == pytest.approx([0.91, 0.81])
    assert [item["final_score"] for item in diagnostics] == pytest.approx([0.818, 0.392])
    assert {item["calibration_method"] for item in diagnostics} == {"min_max_probability_normalization"}


def test_specialist_only_reproduces_pre_fix_rescoring(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "specialist_only")
    grounder, result = run_grounder(StubSpecialist([0.91, 0.81]))
    assert [item.score for item in result.detections] == [0.91, 0.81]
    assert [item.bbox_pixels for item in result.detections] == [[10, 10, 30, 30], [60, 60, 80, 80]]
    assert all(item["score_mode"] == "specialist_only" for item in grounder.last_score_diagnostics)


def test_fallback_preservation_reverts_if_specialist_changes_detection_order(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "fallback")
    grounder, result = run_grounder(StubSpecialist([0.60, 0.99]))
    assert [item.score for item in result.detections] == [0.74, 0.56]
    assert [item.bbox_pixels for item in result.detections] == [[10, 10, 30, 30], [60, 60, 80, 80]]
    assert {item["fallback_reason"] for item in grounder.last_score_diagnostics} == {"specialist_would_change_accepted_geometry_or_labels"}


def test_corrupted_checkpoint_restores_dino_scores(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE", "fallback")
    (tmp_path / "grounding_specialist_v1_1_head.pt").write_bytes(b"corrupt")
    grounder, result = run_grounder(GroundingSpecialist(model_dir=tmp_path))  # type: ignore[arg-type]
    assert [item.score for item in result.detections] == [0.74, 0.56]
    assert grounder.grounding_specialist_health().status == "failed"


def test_score_diagnostics_are_internal_and_api_model_schema_is_unchanged() -> None:
    from satquery_agent.models import GroundingResult

    fields = GroundingResult.model_fields
    assert "score_diagnostics" not in fields
    assert "original_grounding_score" not in fields


def test_specialist_failure_logs_warning_and_preserves_grounding_dino_scores() -> None:
    specialist = StubSpecialist(fail=True)
    with patch("satquery_agent.specialists.grounder.logger.warning") as warning:
        grounder, result = run_grounder(specialist)
    assert [item.score for item in result.detections] == [0.74, 0.56]
    assert "Grounding DINO text-region alignment" in result.confidence.reason
    assert grounder.health().status == "degraded"
    assert grounder.health().smoke_verified is False
    assert grounder.grounding_specialist_health().status == "failed"
    assert warning.called


def test_missing_specialist_checkpoint_is_sticky_safe_fallback(tmp_path: Path) -> None:
    specialist = GroundingSpecialist(model_dir=tmp_path)
    for _ in range(2):
        try:
            specialist.load("cpu")
        except GroundingSpecialistError:
            pass
    assert specialist.health().status == "failed"
    assert "model file missing" in specialist.health().error
    assert specialist.load_count == 0


def test_disabled_specialist_health_and_load_are_safe(tmp_path: Path) -> None:
    specialist = GroundingSpecialist(model_dir=tmp_path, enabled=False)
    assert specialist.health().status == "disabled"
    try:
        specialist.load("cpu")
    except GroundingSpecialistError:
        pass
    assert specialist.health().status == "disabled"
    assert specialist.load_count == 0


def test_health_endpoint_reports_grounding_specialist_without_schema_change() -> None:
    specialist = StubSpecialist()
    grounder = RemoteSensingGrounder(grounding_specialist=specialist)  # type: ignore[arg-type]
    specialist.load("cpu")
    with patch("satquery_agent.api.get_grounder", return_value=grounder):
        body = TestClient(app).get("/api/agent/health").json()
    assert body["specialists"]["grounding_specialist_v1_1"]["status"] == "ready"
    assert body["specialists"]["grounding_specialist_v1_1"]["device"] == "cpu"
    assert body["specialists"]["grounding_specialist_v1_1"]["error"] is None


def test_bundle_model_card_matches_production_provenance() -> None:
    import json

    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    assert card["training_dataset"].startswith("VRSBench")
    assert card["training_recipe"]["best_step"] == CHECKPOINT_STEP == 600
    assert card["training_recipe"]["balanced_sampling"] is True
    assert math.isclose(card["training_recipe"]["area_regularization_weight"], 0.05)
    assert math.isclose(card["training_recipe"]["query_entropy_weight"], 0.01)
