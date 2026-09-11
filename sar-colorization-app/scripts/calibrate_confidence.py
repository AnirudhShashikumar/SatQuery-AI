#!/usr/bin/env python3
"""Optional post-hoc reliability analysis for benchmark confidence/correctness pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from evaluation_common import write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import write_csv, write_json


def calibration_metrics(rows: list[dict[str, Any]], bins: int = 10, minimum_samples: int = 10) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = []
    for row in rows:
        try:
            confidence = float(row["confidence"])
            raw_correct = row["correct"]
            correct = raw_correct if isinstance(raw_correct, bool) else str(raw_correct).strip().lower() in {"1", "true", "yes"}
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= confidence <= 1.0:
            values.append((confidence, float(correct)))
    if len(values) < minimum_samples:
        return {"status": "not calibrated / insufficient evaluation evidence", "samples": len(values), "ece": None, "brier_score": None, "bins": []}
    reliability, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [(confidence, correct) for confidence, correct in values if lower <= confidence <= upper and (index == bins - 1 or confidence < upper)]
        if not members:
            reliability.append({"lower": lower, "upper": upper, "samples": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_confidence = sum(value[0] for value in members) / len(members)
        accuracy = sum(value[1] for value in members) / len(members)
        ece += len(members) / len(values) * abs(mean_confidence - accuracy)
        reliability.append({"lower": lower, "upper": upper, "samples": len(members), "mean_confidence": mean_confidence, "accuracy": accuracy})
    brier = sum((confidence - correct) ** 2 for confidence, correct in values) / len(values)
    return {"status": "calibration diagnostics computed", "samples": len(values), "ece": ece, "brier_score": brier, "bins": reliability, "disclosure": "Diagnostics do not convert runtime confidence into a calibrated probability."}


def main(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Calibration input is unavailable: {source}")
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
    result = calibration_metrics(rows, bins=args.bins, minimum_samples=args.minimum_samples)
    output = args.output_dir.expanduser().resolve()
    write_json(output / "calibration.json", result)
    write_csv(output / "reliability_bins.csv", result["bins"])
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/confidence_calibration"))
    value.add_argument("--bins", type=int, default=10)
    value.add_argument("--minimum-samples", type=int, default=10)
    return value


if __name__ == "__main__":
    main(parser().parse_args())
