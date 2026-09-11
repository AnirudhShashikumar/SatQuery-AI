"""Evidence-gating and language tests for local bi-temporal interpretation."""

from __future__ import annotations

import pytest

from satquery_agent.models import (
    ChangeAnalysisResponse,
    ChangeEngine,
    ChangePreviewUrls,
    ChangeRegion,
    ChangeStatistics,
    PixelBoundingBox,
    SVEResult,
    SVESemanticComparison,
)
from satquery_agent.specialists.bitemporal_semantics import (
    classify_change_intent,
    interpret_change,
)
from satquery_agent.specialists.vqa import answer_change_question


def _statistics(
    percentage: float = 1.1,
    *,
    location: str = "northeast",
    regions: bool = True,
) -> ChangeStatistics:
    boxes = {
        "northeast": PixelBoundingBox(left=70, top=5, right=90, bottom=25, area_pixels=400),
        "center": PixelBoundingBox(left=40, top=40, right=60, bottom=60, area_pixels=400),
    }
    box = boxes[location]
    count = round(percentage * 100)
    region_values = [ChangeRegion(region_id=1, area_pixels=max(1, count), percentage_of_image=percentage, bounding_box=box)] if regions and count else []
    return ChangeStatistics(
        analysis_width=100,
        analysis_height=100,
        source_width=100,
        source_height=100,
        total_pixels=10_000,
        changed_pixels=count,
        percentage_changed=percentage,
        largest_connected_region=max(0, count),
        number_of_regions=len(region_values),
        bounding_boxes=[box] if region_values else [],
        regions=region_values,
        normalized_threshold=0.5,
    )


def _sve(changes, *, available: bool = True) -> SVEResult:
    return SVEResult(
        available=available,
        status="success" if available else "unavailable",
        semantic_comparison=SVESemanticComparison(
            label="Before/after",
            status="supporting_evidence" if available else "unavailable",
            similarity=0.8 if available else None,
            prior_changes=changes,
            disclaimer="Scene-level similarity evidence only.",
        ),
    )


def _change(
    *,
    percentage: float = 1.1,
    deterministic_percentage: float = 1.5,
    location: str = "northeast",
    sve: SVEResult | None = None,
    learned: bool = True,
    fallback: bool = False,
    regions: bool = True,
) -> ChangeAnalysisResponse:
    engine = ChangeEngine(
        mode="hybrid" if learned else "deterministic_fallback" if fallback else "deterministic",
        primary_tool="changerex_change_detector" if learned else "deterministic_change_analyzer",
        supporting_tool="deterministic_change_analyzer" if learned else None,
        fallback_used=fallback,
        fallback_reason="checkpoint_unavailable" if fallback else None,
    )
    return ChangeAnalysisResponse.model_construct(
        statistics=_statistics(percentage, location=location, regions=regions),
        deterministic_statistics=_statistics(deterministic_percentage, location=location, regions=regions),
        change_engine=engine,
        sve_result=sve,
        before_metadata=type("Metadata", (), {"is_georeferenced": False})(),
        semantic_change_summary=None,
        previews=ChangePreviewUrls(),
        ttp_result=None,
        evidence_consistency=None,
    )


@pytest.mark.parametrize(
    ("question", "delta", "expected"),
    [
        ("Did the built-up area increase?", 0.12, "built_up_increase"),
        ("Did built-up area decrease?", -0.12, "built_up_decrease"),
    ],
)
def test_built_up_direction_requires_scene_and_spatial_evidence(question, delta, expected) -> None:
    summary = interpret_change(question, _change(sve=_sve([
        {"label": "urban or built-up area", "difference": delta},
        {"label": "residential area", "difference": delta * 0.9},
    ])))
    assert summary.likely_change_type == expected
    assert summary.evidence_strength == "strong"
    assert "built-up" in summary.short_answer.lower()


@pytest.mark.parametrize(
    ("question", "label", "delta", "expected"),
    [
        ("Has vegetation increased?", "vegetation", 0.11, "vegetation_increase"),
        ("Has vegetation decreased?", "forest", -0.11, "vegetation_decrease"),
        ("Did the water body expand?", "inland water", 0.11, "water_increase"),
        ("Did the water body shrink?", "marine water", -0.11, "water_decrease"),
    ],
)
def test_vegetation_and_water_directional_transitions(question, label, delta, expected) -> None:
    summary = interpret_change(question, _change(sve=_sve([{"label": label, "difference": delta}])))
    assert summary.likely_change_type == expected
    assert summary.likely_transitions[0].confidence in {"moderate", "weak"}


def test_unknown_semantic_change_and_claim_gate_without_support() -> None:
    summary = interpret_change("What changed?", _change(sve=None))
    assert summary.likely_change_type == "unknown_change"
    assert "not strong enough" in summary.expanded_answer
    assert "new building" not in summary.expanded_answer.lower()
    assert not summary.likely_transitions


def test_no_significant_and_same_image_case() -> None:
    summary = interpret_change("Did the scene remain unchanged?", _change(percentage=0, deterministic_percentage=0, regions=False))
    assert summary.overall_change_level == "minimal"
    assert "largely consistent" in summary.short_answer
    assert summary.likely_change_type is None


@pytest.mark.parametrize(("percentage", "expected"), [(1.1, "localized"), (35.0, "widespread")])
def test_change_magnitude_language(percentage, expected) -> None:
    summary = interpret_change("What changed?", _change(percentage=percentage, deterministic_percentage=percentage))
    assert summary.overall_change_level == expected


def test_large_visual_small_learned_disagreement_is_not_averaged() -> None:
    summary = interpret_change("What changed?", _change(percentage=1.113, deterministic_percentage=39.0))
    assert summary.visual_structural_disagreement is True
    assert "differ much more in overall appearance" in summary.expanded_answer
    assert "39% of the area changed" not in summary.expanded_answer


@pytest.mark.parametrize(("location", "expected"), [("northeast", "upper-right"), ("center", "center")])
def test_location_generation(location, expected) -> None:
    summary = interpret_change("Where did the largest change occur?", _change(location=location))
    assert summary.dominant_location == expected
    assert expected in summary.short_answer
    assert "bounding box" not in summary.short_answer.lower()


def test_missing_and_unavailable_sve_block_semantic_claims() -> None:
    missing = interpret_change("Did built-up area increase?", _change(sve=None))
    unavailable = interpret_change("Did built-up area increase?", _change(sve=_sve([], available=False)))
    for summary in (missing, unavailable):
        assert summary.likely_change_type == "unknown_change"
        assert "not strong enough" in summary.short_answer


def test_fallback_describes_visual_not_learned_structural_change() -> None:
    summary = interpret_change("What changed?", _change(learned=False, fallback=True, sve=_sve([
        {"label": "urban or built-up area", "difference": 0.2},
    ])))
    assert not summary.likely_transitions
    assert "visual-difference evidence" in summary.expanded_answer
    assert any("learned change specialist was unavailable" in item for item in summary.caveats)


def test_malformed_scene_evidence_is_ignored_safely() -> None:
    summary = interpret_change("What changed?", _change(sve=_sve([
        {"label": "urban or built-up area", "difference": "invalid"},
        {"label": "vegetation", "difference": float("nan")},
        {"difference": 0.2},
    ])))
    assert summary.likely_change_type == "unknown_change"
    assert not summary.likely_transitions


def test_query_intents_cover_largest_and_summary() -> None:
    assert classify_change_intent("Where did the largest change occur?") == "largest_change"
    assert classify_change_intent("What changed between these dates?") == "change_summary"


def test_legacy_change_response_without_semantic_field_still_answers() -> None:
    change = _change(learned=False)
    change.semantic_change_summary = None
    controlled = answer_change_question("What changed between these dates?", change)
    assert controlled.details.supported is True
    assert "analysis grid" in (controlled.answer or "")

