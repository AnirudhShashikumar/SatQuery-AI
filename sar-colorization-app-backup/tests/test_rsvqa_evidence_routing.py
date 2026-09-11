"""Focused regression tests for RSVQA routing and evidence validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import backend
from satquery_agent.entity_registry import (
    get_remote_sensing_entity,
    grounding_prompt_for,
    resolve_remote_sensing_entity,
)
from satquery_agent.specialists.vqa import get_vqa


def _grounding(count: int | None):
    result = None if count is None else SimpleNamespace(detections=[object()] * count)
    return SimpleNamespace(grounding_result=result)


def _response(*, answer=None, priors=(), statistics=None):
    sve = SimpleNamespace(
        available=True,
        scene_priors=[SimpleNamespace(label=label, similarity=score) for label, score in priors],
    ) if priors else None
    details = SimpleNamespace(statistics_used=statistics or {})
    return SimpleNamespace(answer=answer, sve_result=sve, vqa_details=details)


@pytest.mark.parametrize(
    ("phrase", "canonical"),
    [
        ("water area", "water"),
        ("medium roads", "road"),
        ("rectangular roads", "road"),
        ("commercial buildings", "building"),
        ("heath", "heath"),
        ("grass area", "grassland"),
        ("arable land", "agriculture"),
    ],
)
def test_controlled_vocabulary_resolution(phrase: str, canonical: str) -> None:
    entity = resolve_remote_sensing_entity(phrase)
    assert entity is not None
    assert entity.canonical_name == canonical


def test_unknown_vocabulary_is_not_mapped_arbitrarily() -> None:
    assert resolve_remote_sensing_entity("solar-powered unicorn enclosure") is None


def test_grounding_prompt_preserves_supported_subtype_language() -> None:
    assert grounding_prompt_for("medium commercial buildings in the image") == "medium commercial building"
    assert grounding_prompt_for("rectangular roads") == "rectangular road"


@pytest.mark.parametrize(
    ("question", "target"),
    [
        ("Is a heath present?", "heath"),
        ("Is there a grass area?", "grassland"),
        ("Are any commercial buildings visible?", "building"),
        ("Does the image contain water?", "water"),
    ],
)
def test_presence_constructions_route_before_fallback(question: str, target: str) -> None:
    intent = get_vqa().classify_question(question)
    assert intent.category.value == "presence_vqa"
    assert intent.target == target


def test_zero_grounding_regions_are_not_a_count_of_zero() -> None:
    result = backend._count_evidence("road", _grounding(0))
    assert result.count is None
    assert result.reason_code == "zero_regions_unconfirmed"


def test_positive_grounding_regions_return_accepted_count() -> None:
    result = backend._count_evidence("building", _grounding(3))
    assert result.count == 3
    assert result.reason_code == "accepted_regions"


def test_unsupported_count_target_returns_null() -> None:
    result = backend._count_evidence("forest", None)
    assert result.count is None
    assert result.reason_code == "unsupported_count_target"
    assert get_remote_sensing_entity("forest").count_meaningful is False


def test_spatial_relation_count_returns_null_without_relation_validator() -> None:
    intent = get_vqa().classify_question("How many medium roads are next to water?")
    result = backend._count_evidence(intent.target, _grounding(4), spatial_relation=intent.spatial_relation)
    assert intent.spatial_relation == "next to"
    assert result.count is None
    assert result.reason_code == "spatial_relation_unsupported"


@pytest.mark.parametrize(
    ("relation", "first", "second", "answer"),
    [("less", 1, 2, "yes"), ("more", 3, 2, "yes"), ("equal", 2, 2, "yes"), ("equal", 1, 2, "no")],
)
def test_comparison_uses_two_available_counts(relation: str, first: int, second: int, answer: str) -> None:
    left = backend.CountEvidence(first, "accepted_regions", "grounding", first)
    right = backend.CountEvidence(second, "accepted_regions", "grounding", second)
    result, comparable, unavailable, reason = backend._compare_count_evidence(left, right, relation)
    assert (result, comparable, unavailable, reason) == (answer, True, [], "comparison_logic_applied")


def test_comparison_with_unavailable_operand_returns_null() -> None:
    left = backend.CountEvidence(2, "accepted_regions", "grounding", 2)
    right = backend.CountEvidence(None, "zero_regions_unconfirmed", "grounding", 0)
    answer, comparable, unavailable, reason = backend._compare_count_evidence(left, right, "more")
    assert answer is None
    assert comparable is False
    assert unavailable == ["entity_b"]
    assert reason == "comparison_operand_unavailable"


def test_comparison_rejects_incomparable_sources() -> None:
    left = backend.CountEvidence(2, "accepted_regions", "grounding", 2)
    right = backend.CountEvidence(1, "validated_count", "other validated counter", 1)
    answer, comparable, unavailable, reason = backend._compare_count_evidence(left, right, "more")
    assert answer is None
    assert comparable is False
    assert unavailable == []
    assert reason == "incomparable_evidence"


@pytest.mark.parametrize(
    ("question", "relation"),
    [
        ("Are there fewer buildings than roads?", "less"),
        ("Is the amount of roads greater than buildings?", "more"),
        ("Is the number of roads the same as buildings?", "equal"),
    ],
)
def test_comparison_relation_parsing(question: str, relation: str) -> None:
    assert get_vqa().classify_question(question).comparison_relation == relation


def test_positive_presence_grounding_has_priority() -> None:
    intent = get_vqa().classify_question("Does the image contain water?")
    answer, reason, _, _, _ = backend._presence_decision(
        intent,
        _response(answer="no"),
        _response(answer=""),
        _grounding(1),
    )
    assert answer == "yes"
    assert reason == "positive_grounding"


def test_presence_requires_multiple_sources_for_absence() -> None:
    intent = get_vqa().classify_question("Is there a building?")
    response = _response(answer="no", priors=(("forest", 0.4), ("grassland", 0.3)))
    answer, reason, _, _, _ = backend._presence_decision(intent, response, _response(answer=""), _grounding(0))
    assert answer == "no"
    assert reason == "multi_source_absence"


def test_presence_conflict_returns_null() -> None:
    intent = get_vqa().classify_question("Is there a building?")
    response = _response(answer="no", priors=(("urban or built-up area", 0.4),))
    answer, reason, _, _, _ = backend._presence_decision(intent, response, _response(answer=""), _grounding(0))
    assert answer is None
    assert reason == "conflicting_evidence"


def test_rural_urban_strong_built_evidence_is_urban() -> None:
    response = _response(
        priors=(("urban or built-up area", 0.5), ("residential area", 0.4), ("forest", 0.2)),
        statistics={"built_up_support_percent": 65.0, "rural_support_percent": 10.0},
    )
    answer, _, details = backend._rural_urban_evidence_fusion(response, _response(answer="dense urban buildings"))
    assert answer == "urban"
    assert details["urban_score"] > details["rural_score"]


def test_rural_urban_strong_agricultural_evidence_is_rural() -> None:
    response = _response(
        priors=(("agricultural land", 0.5), ("arable land", 0.4), ("urban or built-up area", 0.2)),
        statistics={"built_up_support_percent": 8.0, "rural_support_percent": 70.0},
    )
    answer, _, details = backend._rural_urban_evidence_fusion(response, _response(answer="rural farmland"))
    assert answer == "rural"
    assert details["rural_score"] > details["urban_score"]


def test_rural_urban_ambiguous_evidence_returns_null() -> None:
    response = _response(statistics={"built_up_support_percent": 30.0, "rural_support_percent": 30.0})
    answer, reason, details = backend._rural_urban_evidence_fusion(response, _response(answer=""))
    assert answer is None
    assert reason == "insufficient_evidence"
    assert details["margin"] < details["minimum_margin"]
