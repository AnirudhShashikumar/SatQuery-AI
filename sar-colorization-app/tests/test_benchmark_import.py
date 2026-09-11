from __future__ import annotations

import json
from pathlib import Path

from scripts.import_existing_benchmarks import build_records, build_suite


ROOT = Path(__file__).resolve().parents[1]


def by_id(records: list[dict], benchmark_id: str) -> dict:
    return next(record for record in records if record["benchmark_id"] == benchmark_id)


def test_import_preserves_rsvqa_official_test_values() -> None:
    records = build_records(ROOT)
    record = by_id(records, "rsvqa.specialist-v1.official-test.2026-08-29")
    source = json.loads((ROOT / "artifacts/rsvqa_full_specialist_v1/results.json").read_text())["summary"]
    assert record["evaluation"]["status"] == "verified_test"
    assert record["evaluation"]["sample_count"] == source["total_records"] == 10004
    assert record["metrics"][0]["value"] == source["exact_match_accuracy"]
    assert record["performance"]["p95_latency_ms"] == source["p95_latency_ms"]


def test_import_preserves_grounding_validation_values_and_breakdowns() -> None:
    records = build_records(ROOT)
    record = by_id(records, "grounding.specialist-v1-1.vrsbench-validation.step600")
    source = json.loads((ROOT / "models/grounding_specialist_v1_1/validation_results.json").read_text())
    assert record["evaluation"]["status"] == "verified_validation"
    assert record["evaluation"]["sample_count"] == 16146
    assert next(metric for metric in record["metrics"] if metric["id"] == "accuracy_at_050")["value"] == source["overall"]["accuracy_at_050"]
    assert len(record["breakdowns"]) == len(source["per_class"]) == 26


def test_smoke_and_operational_sources_are_not_promoted() -> None:
    records = build_records(ROOT)
    baseline = by_id(records, "grounding.dino-tiny.vrsbench-smoke100.baseline")
    changerex = by_id(records, "changerex.r18.hanford-mps-operational")
    pix2pix = by_id(records, "sar-translation.pix2pix.single-image-operational")
    assert baseline["evaluation"]["status"] == "smoke_only"
    assert changerex["evaluation"]["status"] == "operational_only"
    assert not any(metric["id"] in {"iou", "f1", "accuracy"} for metric in changerex["metrics"])
    assert pix2pix["metrics"] == []


def test_suite_prefers_strongest_available_result_and_records_gaps() -> None:
    records = build_records(ROOT)
    suite = build_suite(ROOT, records)
    assert suite["preferred_record_by_specialist"]["rsvqa"].endswith("official-test.2026-08-29")
    assert suite["preferred_record_by_specialist"]["grounding"].endswith("validation.step600")
    assert suite["coverage"] == {
        "verified_test": 1,
        "verified_validation": 1,
        "partial_validation": 0,
        "smoke_only": 0,
        "operational_only": 2,
        "external_reported": 0,
        "unavailable": 3,
    }
    assert any("caption" in gap.lower() for gap in suite["known_gaps"])


def test_import_is_deterministic() -> None:
    first = build_records(ROOT)
    second = build_records(ROOT)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
