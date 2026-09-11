"""Regression coverage for the backward-compatible RSVQA image-analysis endpoint."""

from __future__ import annotations

import io
import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend
from satquery_agent.specialists.vqa import get_vqa, normalize_benchmark_answer


def _png_bytes() -> bytes:
    height = width = 48
    yy, xx = np.indices((height, width))
    image = np.dstack(
        (
            (70 + xx * 2) % 255,
            (100 + yy * 3) % 255,
            (50 + (xx + yy)) % 255,
        )
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _legacy_result() -> dict:
    return {
        "report": {
            "executive_summary": "A mixed remote-sensing scene with vegetation and built patterns.",
            "confidence": "medium",
        },
        "provider": "gemini",
        "model": backend.VISION_ANALYSIS.settings.model,
        "cached": False,
    }


def _form(**values: str) -> dict:
    return {
        "analysis_type": "ground_truth",
        "model_name": backend.VISION_ANALYSIS.settings.model,
        **values,
    }


client = TestClient(backend.app)


def test_image_only_preserves_the_legacy_response() -> None:
    with patch.object(backend.VISION_ANALYSIS, "analyze_image", side_effect=lambda *_args: _legacy_result()):
        response = client.post(
            "/api/analysis/image",
            data=_form(),
            files={"image": ("scene.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"] == _legacy_result()["report"]
    assert body["provider"] == "gemini"
    assert body["model"] == backend.VISION_ANALYSIS.settings.model
    assert body["cached"] is False
    assert len(body["request_id"]) == 12
    for additive_field in (
        "answer", "confidence", "caption", "evidence", "execution_trace", "task",
        "model_used", "processing_time_ms", "question",
    ):
        assert additive_field not in body


def test_image_and_question_runs_the_existing_satquery_vqa_pipeline() -> None:
    with patch.dict(os.environ, {"SATQUERY_CAPTIONER_ENABLED": "0"}), patch.object(
        backend.VISION_ANALYSIS,
        "analyze_image",
    ) as provider:
        response = client.post(
            "/api/analysis/image",
            data={**_form(question="What is the dominant land cover?"), "model_name": "satquery-agent"},
            files={"image": ("scene.png", _png_bytes(), "image/png")},
        )
    provider.assert_not_called()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["caption"] == "The local remote-sensing captioner did not produce a caption for this image."
    assert isinstance(body["evidence"], list)
    assert body["task"] == "vqa"
    assert "rs_captioner" in body["model_used"]
    assert "rs_vqa" in body["model_used"]
    assert body["processing_time_ms"] >= 0
    assert body["question"] == "What is the dominant land cover?"
    assert body["routed_question"] == body["question"]
    tools = [step["tool"] for step in body["execution_trace"]]
    assert "query_routing" in tools
    assert "question_classification" in tools
    assert "evidence_extraction" in tools
    assert "controlled_answer_generation" in tools
    assert any(step["pipeline_stage"] == "captioning" for step in body["execution_trace"])
    assert any(step["pipeline_stage"] == "vqa" for step in body["execution_trace"])
    assert body["execution_trace"][-1]["pipeline_stage"] == "evidence_fusion"
    # The original response fields remain present without requiring an external provider.
    assert body["report"]["executive_summary"] == body["caption"]
    assert body["provider"] == "satquery"
    assert "rs_vqa" in body["model"]
    assert body["cached"] is False


def test_common_rsvqa_aliases_and_count_grounding_are_normalized() -> None:
    assert backend._rsvqa_routed_question("Is there a river?") == "Is a water body visible?"
    assert backend._rsvqa_routed_question("Is this urban?") == "Is this mainly urban or rural?"
    assert backend._rsvqa_routed_question("What is the dominant land cover?") == "What is the dominant land cover?"
    assert backend._count_grounding_query("How many roads?") == ("Locate the road.", "road")
    assert backend._count_grounding_query("How many clouds?") is None


@pytest.mark.parametrize(
    ("question", "category", "target"),
    [
        ("Is it a rural or an urban area", "rural_urban_classification", "scene"),
        ("Is this urban?", "rural_urban_classification", "scene"),
        ("Is this a rural scene?", "rural_urban_classification", "scene"),
        ("Would you describe this area as rural or urban?", "rural_urban_classification", "scene"),
        ("Is there a grass area?", "presence_vqa", "grassland"),
        ("Is there a river?", "presence_vqa", "water"),
        ("Are any buildings visible?", "presence_vqa", "building"),
        ("Can you see any buildings?", "presence_vqa", "building"),
        ("Is any forest visible?", "presence_vqa", "forest"),
        ("What is the number of roads?", "count_vqa", "road"),
        ("How many large commercial buildings are there?", "count_vqa", "building"),
        ("What is the number of medium commercial buildings in the image?", "count_vqa", "building"),
        ("What is the amount of medium roads next to a water area?", "count_vqa", "road"),
        ("Are there less buildings than roads?", "comparison_vqa", "building"),
        ("Are there fewer buildings than roads?", "comparison_vqa", "building"),
        ("Is the number of roads equal to the number of commercial buildings?", "comparison_vqa", "road"),
        ("Are there more water areas than roads?", "comparison_vqa", "water"),
    ],
)
def test_rsvqa_lr_question_families_are_classified_before_caption_fallback(
    question: str,
    category: str,
    target: str,
) -> None:
    intent = get_vqa().classify_question(question)
    assert intent.category.value == category
    assert intent.target == target


def test_comparison_entities_and_relation_are_parsed() -> None:
    comparison = get_vqa().classify_question("Is the amount of roads greater than buildings?")
    assert comparison.secondary_target == "building"
    assert comparison.comparison_relation == "more"


def test_exact_rural_urban_regression_is_answered_without_grounding() -> None:
    with patch.dict(os.environ, {"SATQUERY_CAPTIONER_ENABLED": "0", "SVE_ENABLED": "0"}), patch.object(
        backend.VISION_ANALYSIS,
        "analyze_image",
    ) as provider:
        response = client.post(
            "/api/analysis/image",
            data={
                **_form(question="Is it a rural or an urban area"),
                "model_name": "satquery-agent",
            },
            files={"image": ("scene.png", _png_bytes(), "image/png")},
        )

    provider.assert_not_called()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "rural_urban_classification"
    assert body["answer"] in {"rural", "urban"}
    assert body["status"] != "not_implemented"
    assert body["result_status"] != "UNSUPPORTED_TASK"
    assert body["evidence"]
    tools = [step["tool"] for step in body["execution_trace"]]
    assert "question_classification" in tools
    assert "vqa_evidence_fusion" in tools
    assert not any("ground" in tool for tool in tools)


def test_benchmark_answer_normalization_rejects_descriptive_or_unsafe_counts() -> None:
    assert normalize_benchmark_answer(" YES! ", "yes_no") == "yes"
    assert normalize_benchmark_answer("negative", "yes_no") == "no"
    assert normalize_benchmark_answer("an urban area", "rural_urban") == "urban"
    assert normalize_benchmark_answer("2", "integer") == "2"
    assert normalize_benchmark_answer("There are 2 accepted regions", "integer") == "2"
    assert normalize_benchmark_answer("a scene with 2 roads and many buildings", "integer") is None
    assert normalize_benchmark_answer(None, "integer") is None


def test_invalid_image_is_rejected_before_provider_or_agent_execution() -> None:
    with patch.object(backend.VISION_ANALYSIS, "analyze_image") as provider, patch.object(backend, "agent_image_query") as agent:
        response = client.post(
            "/api/analysis/image",
            data=_form(question="Is there a river?"),
            files={"image": ("broken.png", b"not-an-image", "image/png")},
        )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE"
    provider.assert_not_called()
    agent.assert_not_called()


def test_missing_file_is_rejected_by_the_generated_multipart_schema() -> None:
    response = client.post("/api/analysis/image", data=_form(question="Is this urban?"))
    assert response.status_code == 422


def test_unsupported_format_is_rejected_before_image_loading() -> None:
    with patch.object(backend.VISION_ANALYSIS, "analyze_image") as provider:
        response = client.post(
            "/api/analysis/image",
            data=_form(question="Is this urban?"),
            files={"image": ("scene.gif", b"GIF89a", "image/gif")},
        )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE"
    provider.assert_not_called()


def test_openapi_documents_optional_question_and_additive_response() -> None:
    schema = backend.app.openapi()
    operation = schema["paths"]["/api/analysis/image"]["post"]
    request_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    request_schema = schema["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    assert "question" in request_schema["properties"]
    assert "question" not in request_schema["required"]
    response_ref = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_schema = schema["components"]["schemas"][response_ref.rsplit("/", 1)[-1]]
    for field in ("answer", "confidence", "caption", "evidence", "execution_trace", "task", "model_used", "processing_time_ms"):
        assert field in response_schema["properties"]
