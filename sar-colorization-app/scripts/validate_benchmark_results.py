#!/usr/bin/env python3
"""Validate all canonical benchmark and demo artifacts without third-party packages."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from benchmark_data import discover_records, read_json, validate_demo_manifest, validate_suite
except ModuleNotFoundError:  # Imported as scripts.validate_benchmark_results in tests.
    from scripts.benchmark_data import discover_records, read_json, validate_demo_manifest, validate_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    records = discover_records(root)
    suite = read_json(root / "benchmarks" / "suites" / "satquery_benchmark_suite.json")
    validate_suite(suite, records)
    validate_demo_manifest(read_json(root / "benchmarks" / "demos" / "demo_gallery.json"))
    print(f"Validated {len(records)} benchmark records and the demo gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
