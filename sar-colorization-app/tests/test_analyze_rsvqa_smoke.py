"""Focused validation for the immutable RSVQA smoke diagnostic analysis."""

from __future__ import annotations

import pytest

from scripts import analyze_rsvqa_smoke as analysis


def _row(
    question_type: str,
    truth: str,
    prediction: str = None,
    correct: bool = False,
    **updates: object,
) -> dict:
    row = {
        "question_type": question_type,
        "normalized_ground_truth": truth,
        "normalized_prediction": prediction,
        "correct": correct,
        "completed": True,
        "confidence": 0.5,
        "latency_ms": 100.0,
        "question": "Unrecognized diagnostic sample",
        "task": "",
        "status": "",
        "warnings": [],
        "answer": prediction,
        "model_used": "",
    }
    row.update(updates)
    return row


@pytest.mark.parametrize("prediction", [None, ""])
def test_null_detection(prediction: str) -> None:
    assert analysis.is_null_prediction(_row("presence", "yes", prediction))
    assert not analysis.is_answered(_row("presence", "yes", prediction))


def test_answered_only_accuracy_excludes_nulls() -> None:
    rows = [
        _row("presence", "yes", "yes", True),
        _row("presence", "no", "yes", False),
        _row("presence", "yes", None, False),
    ]
    metrics = analysis.compute_group_metrics(rows)
    assert metrics["overall_accuracy"] == pytest.approx(1 / 3)
    assert metrics["answered_only_accuracy"] == pytest.approx(1 / 2)


def test_per_type_coverage() -> None:
    rows = [
        _row("presence", "yes", "yes", True),
        _row("presence", "no", None, False),
        _row("count", "0", "0", True),
    ]
    metrics = analysis.compute_core_metrics(rows)
    assert metrics["by_type"]["presence"]["coverage"] == pytest.approx(0.5)
    assert metrics["by_type"]["count"]["coverage"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("answer", "bucket"),
    [("0", "ground truth 0"), ("1", "ground truth 1"), ("2", "ground truth 2+"), ("3428", "ground truth 2+"), (None, "invalid")],
)
def test_count_answer_buckets(answer: str, bucket: str) -> None:
    assert analysis.count_answer_bucket(answer) == bucket


def test_majority_baseline() -> None:
    rows = [_row("presence", "yes"), _row("presence", "yes"), _row("presence", "no")]
    baseline = analysis.majority_baseline(rows)
    assert baseline["label"] == "yes"
    assert baseline["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("question", "question_type", "pattern"),
    [
        ("Is it a rural or an urban area", "rural_urban", "rural-or-urban"),
        ("Is this urban?", "rural_urban", "direct urban"),
        ("Is there a river?", "presence", "is there"),
        ("Can you see any buildings?", "presence", "visible/can you see"),
        ("How many buildings are there?", "count", "how many"),
        ("What is the amount of roads next to water?", "count", "relational count"),
        ("Are there fewer buildings than roads?", "comp", "less/fewer"),
        ("Is the number of roads equal to buildings?", "comp", "equal/same"),
    ],
)
def test_wording_classification(question: str, question_type: str, pattern: str) -> None:
    assert analysis.classify_wording(question, question_type) == pattern


def test_failure_taxonomy_falls_back_to_unknown_without_evidence() -> None:
    row = _row("unmapped", "yes", "no", False)
    category, evidence = analysis.classify_failure(row)
    assert category == "unknown"
    assert "do not support" in evidence
