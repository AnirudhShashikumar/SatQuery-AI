#!/usr/bin/env python3
"""Rebuild the suite manifest from deterministic imported benchmark records."""

from __future__ import annotations

from pathlib import Path

try:
    from benchmark_data import write_json
    from import_existing_benchmarks import build_records, build_suite
except ModuleNotFoundError:  # Imported as scripts.build_benchmark_suite in tests.
    from scripts.benchmark_data import write_json
    from scripts.import_existing_benchmarks import build_records, build_suite


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    records = build_records(root)
    write_json(root / "benchmarks" / "suites" / "satquery_benchmark_suite.json", build_suite(root, records))
    print(f"Built suite with {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
