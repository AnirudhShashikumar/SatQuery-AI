"""Standalone benchmark contracts shared by import and validation scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "verified_test",
    "verified_validation",
    "partial_validation",
    "smoke_only",
    "operational_only",
    "external_reported",
    "unavailable",
    "not_applicable",
}
ALLOWED_UNITS = {
    "ratio", "percentage", "milliseconds", "seconds", "megabytes",
    "count", "samples_per_second", "db", "score",
}
REQUIRED_RESULT_KEYS = {
    "schema_version", "benchmark_id", "specialist_id", "display_name", "task",
    "model", "evaluation", "metrics", "performance", "artifacts", "limitations",
    "notes", "provenance",
}


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark artifact violates the reporting contract."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkValidationError(message)


def _iso8601(value: str | None, field: str) -> None:
    if value is None:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BenchmarkValidationError(f"{field} must be ISO 8601") from exc


def validate_result(record: dict[str, Any]) -> None:
    missing = REQUIRED_RESULT_KEYS - record.keys()
    _require(not missing, f"missing benchmark keys: {sorted(missing)}")
    _require(record["schema_version"] == "1.0.0", "unsupported schema_version")
    _require(isinstance(record["benchmark_id"], str) and bool(record["benchmark_id"]), "benchmark_id is required")
    evaluation = record["evaluation"]
    _require(evaluation.get("status") in ALLOWED_STATUSES, "invalid evaluation status")
    count = evaluation.get("sample_count")
    _require(count is None or isinstance(count, int) and count >= 0, "sample_count must be null or a non-negative integer")
    _iso8601(evaluation.get("timestamp"), "evaluation.timestamp")
    for metric in record["metrics"]:
        _require(metric.get("unit") in ALLOWED_UNITS, f"invalid metric unit: {metric.get('unit')}")
        _require(metric.get("value") is None or isinstance(metric.get("value"), (int, float)), "metric value must be numeric or null")
        _require(isinstance(metric.get("primary"), bool), "metric primary must be boolean")
    for breakdown in record.get("breakdowns", []):
        _require(isinstance(breakdown.get("metrics"), list) and breakdown["metrics"], "breakdown metrics are required")
    model = record["model"]
    fingerprint = model.get("checkpoint_sha256")
    _require(fingerprint is None or isinstance(fingerprint, str) and len(fingerprint) == 64, "checkpoint_sha256 must be null or 64 hex characters")
    provenance = record["provenance"]
    _require(provenance.get("result_origin") in {"satquery_reproduced", "external_reported", "unmeasured"}, "invalid result_origin")
    for source in provenance.get("source_artifacts", []):
        _require(not source.startswith("/"), "source artifact references must not expose absolute paths")
        _require("/Users/" not in source and "/home/" not in source, "source artifact contains a private path")


def validate_suite(suite: dict[str, Any], records: list[dict[str, Any]]) -> None:
    _require(suite.get("schema_version") == "1.0.0", "unsupported suite schema")
    _iso8601(suite.get("generated_at"), "suite.generated_at")
    ids = [record["benchmark_id"] for record in records]
    _require(len(ids) == len(set(ids)), "duplicate benchmark IDs")
    _require(set(suite.get("benchmark_record_ids", [])) == set(ids), "suite record IDs do not match result files")
    preferred = suite.get("preferred_record_by_specialist", {})
    _require(all(value in ids for value in preferred.values()), "preferred record references an unknown benchmark")


def validate_demo_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("schema_version") == "1.0.0", "unsupported demo schema")
    _iso8601(manifest.get("generated_at"), "demo.generated_at")
    ids: list[str] = []
    for case in manifest.get("cases", []):
        ids.append(case.get("demo_id", ""))
        _require(case.get("expected_result_classification") in {
            "benchmark_ground_truth", "curated_expected_behavior", "illustrative_only"
        }, "invalid demo expected-result classification")
        _require(case.get("readiness") in {"ready", "guided_only", "unavailable"}, "invalid demo readiness")
        _require(str(case.get("primary_image", "")).startswith("/demos/"), "demo image must be a public demo asset")
        _require(bool(case.get("attribution")) and bool(case.get("license")), "demo attribution and license are required")
    _require(len(ids) == len(set(ids)), "duplicate demo IDs")


def discover_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "benchmarks" / "results").glob("*/*.json")):
        record = read_json(path)
        validate_result(record)
        records.append(record)
    return records
