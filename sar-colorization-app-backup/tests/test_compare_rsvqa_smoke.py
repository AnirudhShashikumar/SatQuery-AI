from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_rsvqa_smoke.py"
SPEC = importlib.util.spec_from_file_location("compare_rsvqa_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(question_id: str, question_type: str, label: str, prediction: str = "") -> dict[str, str]:
    return {
        "question_id": question_id,
        "image_id": question_id,
        "question_type": question_type,
        "question": f"question {question_id}",
        "normalized_ground_truth": label,
        "normalized_prediction": prediction,
        "correct": str(prediction == label).lower(),
        "endpoint_error": "false",
        "status": "success" if prediction else "insufficient_evidence",
    }


def test_compare_reports_fixes_regressions_and_distribution() -> None:
    v1 = [
        _row("1", "count", "2", "0"),
        _row("2", "count", "0", "0"),
        _row("3", "presence", "yes", "yes"),
        _row("4", "presence", "no", ""),
        _row("5", "comp", "no", "no"),
        _row("6", "rural_urban", "urban", "rural"),
    ]
    v2 = [
        _row("1", "count", "2", "2"),
        _row("2", "count", "0", ""),
        _row("3", "presence", "yes", "no"),
        _row("4", "presence", "no", "no"),
        _row("5", "comp", "no", "no"),
        _row("6", "rural_urban", "urban", "urban"),
    ]
    summary, changes = MODULE.compare(v1, v2)
    assert summary["fixed_questions"] == 3
    assert summary["regressions"] == 2
    assert summary["count_prediction_distribution"]["v1"] == {"0": 2}
    assert summary["count_prediction_distribution"]["v2"] == {"2": 1, "null": 1}
    assert {row["change"] for row in changes} >= {"fixed", "regression", "unchanged_correct"}


def test_compare_rejects_different_question_sets() -> None:
    try:
        MODULE.compare([_row("1", "count", "0", "0")], [_row("2", "count", "0", "0")])
    except ValueError as error:
        assert "Question sets differ" in str(error)
    else:
        raise AssertionError("Expected mismatched runs to be rejected")
