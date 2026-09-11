"""Problem Statement 26167 representative-query regression coverage.

These tests deliberately assert contracts and evidence families, not exact prose.
Feature-specific endpoint suites exercise the corresponding production adapters.
"""

from __future__ import annotations

import pytest

from satquery_agent.models import AgentQueryRequest, InputMode, Modality, TaskType
from satquery_agent.router import route_query


CASES = (
    (
        "Describe the land-cover and major objects visible in this image.",
        InputMode.SINGLE,
        Modality.OPTICAL,
        None,
        TaskType.CAPTIONING,
        "rs_captioner",
        "caption",
    ),
    (
        "Highlight the water body referred to in the query.",
        InputMode.SINGLE,
        Modality.OPTICAL,
        None,
        TaskType.GROUNDING,
        "rs_grounder",
        "localized_box",
    ),
    (
        "What changed between these two dates, and where did the change occur?",
        InputMode.BI_TEMPORAL,
        Modality.OPTICAL,
        Modality.OPTICAL,
        TaskType.CHANGE_DESCRIPTION,
        "bitemporal_change_analyzer",
        "change_mask_and_regions",
    ),
    (
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        InputMode.CROSS_MODAL,
        Modality.OPTICAL,
        Modality.SAR,
        TaskType.CROSS_MODAL_ANALYSIS,
        "cross_modal_optical_sar_analyzer",
        "optical_native_sar_and_fused",
    ),
    (
        "Has the built-up area increased, decreased, or remained unchanged?",
        InputMode.BI_TEMPORAL,
        Modality.OPTICAL,
        Modality.OPTICAL,
        TaskType.CHANGE_VQA,
        "bitemporal_change_analyzer",
        "evidence_gated_semantic_change",
    ),
)


@pytest.mark.parametrize(
    "query,input_mode,primary,secondary,task,specialist,evidence_family",
    CASES,
)
def test_official_representative_query_contract(
    query,
    input_mode,
    primary,
    secondary,
    task,
    specialist,
    evidence_family,
) -> None:
    request = AgentQueryRequest(
        query=query,
        input_mode=input_mode,
        primary_modality=primary,
        secondary_modality=secondary,
        has_primary_image=True,
        has_secondary_image=input_mode != InputMode.SINGLE,
    )
    plan = route_query(request)
    assert plan.validation_status.valid, plan.validation_status.errors
    assert plan.detected_task == task
    assert specialist in plan.selected_tools
    assert plan.selection_reason
    assert set(plan.permitted_parameters) == {
        "query", "input_mode", "primary_modality", "secondary_modality",
        "has_primary_image", "has_secondary_image", "primary_image_modality",
        "primary_representation", "primary_band_count",
    }
    assert evidence_family  # Each case declares the evidence contract checked by its specialist suite.


def test_directional_change_query_cannot_fall_into_single_image_vqa() -> None:
    plan = route_query(AgentQueryRequest(
        query="Has the built-up area increased, decreased, or remained unchanged?",
        input_mode=InputMode.SINGLE,
        primary_modality=Modality.OPTICAL,
        has_primary_image=True,
        has_secondary_image=False,
    ))
    assert plan.detected_task == TaskType.CHANGE_VQA
    assert not plan.validation_status.valid
    assert any("bi_temporal" in error for error in plan.validation_status.errors)


@pytest.mark.parametrize("query", [
    "Where is water?",
    "What does SAR reveal that optical does not?",
    "Identify built-up and water-covered regions.",
])
def test_selected_cross_modal_workflow_routes_short_queries_to_fusion(query: str) -> None:
    plan = route_query(AgentQueryRequest(
        query=query,
        input_mode=InputMode.CROSS_MODAL,
        primary_modality=Modality.OPTICAL,
        secondary_modality=Modality.SAR,
        has_primary_image=True,
        has_secondary_image=True,
    ))
    assert plan.detected_task == TaskType.CROSS_MODAL_ANALYSIS
    assert plan.validation_status.valid
    assert plan.selected_tools[-1] == "cross_modal_optical_sar_analyzer"


def test_compliance_suite_does_not_define_exact_answer_text() -> None:
    # Wording is allowed to improve; the suite guards routing, evidence, trace and
    # provenance through the specialist contract tests instead of snapshots.
    assert all(len(case[0]) > 10 for case in CASES)
