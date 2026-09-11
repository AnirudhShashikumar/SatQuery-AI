from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.benchmark_data import (
    BenchmarkValidationError,
    discover_records,
    read_json,
    validate_demo_manifest,
    validate_result,
    validate_suite,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_canonical_benchmark_records_and_suite_validate() -> None:
    records = discover_records(ROOT)
    assert len(records) == 9
    suite = read_json(ROOT / "benchmarks/suites/satquery_benchmark_suite.json")
    validate_suite(suite, records)
    assert set(suite["benchmark_record_ids"]) == {record["benchmark_id"] for record in records}


def test_invalid_schema_is_rejected() -> None:
    record = discover_records(ROOT)[0]
    invalid = copy.deepcopy(record)
    invalid["evaluation"]["status"] = "accurate"
    with pytest.raises(BenchmarkValidationError, match="invalid evaluation status"):
        validate_result(invalid)


def test_missing_metric_stays_missing_and_zero_is_not_null() -> None:
    caption = read_json(ROOT / "benchmarks/results/captioning/captioning.unavailable.v1.json")
    assert caption["metrics"] == []
    measured = copy.deepcopy(next(record for record in discover_records(ROOT) if record["metrics"]))
    measured["metrics"][0]["value"] = 0
    validate_result(measured)
    assert measured["metrics"][0]["value"] == 0


def test_private_paths_are_rejected() -> None:
    record = copy.deepcopy(discover_records(ROOT)[0])
    record["provenance"]["source_artifacts"] = ["/Users/example/private.json"]
    with pytest.raises(BenchmarkValidationError, match="absolute paths"):
        validate_result(record)


def test_demo_manifest_has_attribution_and_expected_result_classes() -> None:
    manifest = read_json(ROOT / "benchmarks/demos/demo_gallery.json")
    validate_demo_manifest(manifest)
    assert len(manifest["cases"]) == 3
    assert {case["expected_result_classification"] for case in manifest["cases"]} == {
        "benchmark_ground_truth", "curated_expected_behavior"
    }
    assert all(case["attribution"] and case["license"] for case in manifest["cases"])


def test_json_schema_files_are_valid_json_and_pin_v1() -> None:
    for path in sorted((ROOT / "benchmarks/schema").glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["properties"]["schema_version"]["const"] == "1.0.0"
