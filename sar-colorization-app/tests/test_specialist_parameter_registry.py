"""Permitted specialist parameter schema tests."""

import pytest

from satquery_agent.registry import (
    SpecialistParameterError,
    tool_definition,
    validate_specialist_parameters,
)


def test_defaults_are_exposed_and_applied() -> None:
    tool = tool_definition("rs_grounder")
    assert set(tool.allowed_parameters) == {"box_threshold", "text_threshold", "device"}
    values = validate_specialist_parameters("rs_grounder", {})
    assert values == tool.defaults


def test_allowed_values_are_typed_and_bounded() -> None:
    values = validate_specialist_parameters("rs_grounder", {
        "box_threshold": 0.25,
        "text_threshold": 0.2,
        "device": "mps",
    })
    assert values["box_threshold"] == 0.25
    assert values["device"] == "mps"


@pytest.mark.parametrize("payload,match", [
    ({"checkpoint": "/tmp/untrusted.pt"}, "Unsupported parameter"),
    ({"box_threshold": "high"}, "must have type float"),
    ({"box_threshold": 2.0}, "must be at most"),
    ({"device": "metal"}, "must be one of"),
])
def test_invalid_parameter_maps_are_rejected(payload, match) -> None:
    with pytest.raises(SpecialistParameterError, match=match):
        validate_specialist_parameters("rs_grounder", payload)


def test_trace_safe_parameter_names_contain_no_checkpoint_selector() -> None:
    tool = tool_definition("changerex_change_detector")
    assert "checkpoint" not in tool.allowed_parameters
    assert tool.model_version == tool.specialist_version
    assert tool.evidence_types == tool.evidence_outputs
