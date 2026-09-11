#!/usr/bin/env python3
"""Controlled same-sample Optical-only, SAR-only, and Optical+SAR evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from evaluate_cdvqa import normalize_answer
    from evaluation_common import benchmark_record, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluate_cdvqa import normalize_answer
    from scripts.evaluation_common import benchmark_record, write_csv, write_json


SUPPORTED_TASKS = {"water_presence", "built_up_presence", "object_or_region_localization", "cross_modal_agreement", "complementary_evidence"}


def fusion_gain(rows: list[dict]) -> float | None:
    labelled = [row for row in rows if row.get("reference_answer") not in {None, ""}]
    if len(labelled) != len(rows) or not rows:
        return None
    fused = sum(bool(row["fused_correct"]) for row in rows) / len(rows)
    optical = sum(bool(row["optical_correct"]) for row in rows) / len(rows)
    sar = sum(bool(row["sar_correct"]) for row in rows) / len(rows)
    return fused - max(optical, sar)


def evaluate(args: argparse.Namespace) -> dict:
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Controlled optical-SAR manifest is unavailable: {manifest}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Controlled optical-SAR manifest is empty")
    root = args.dataset_root.expanduser().resolve()
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    rows, failures = [], []
    for index, record in enumerate(records):
        task = record.get("task")
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task in record {index}: {task}")
        optical, sar = root / record["optical_image"], root / record["sar_image"]
        if not optical.is_file() or not sar.is_file():
            raise FileNotFoundError(f"Missing paired input for record {index}")
        question, reference = record["question"], record.get("answer")
        try:
            def single(path: Path, modality: str):
                return client.post("/api/agent/query", data={"query": question, "input_mode": "single", "primary_modality": modality, "use_cache": "false"}, files={"primary_image": (path.name, path.read_bytes())}).json()
            optical_result, sar_result = single(optical, "optical"), single(sar, "sar")
            fused_result = client.post("/api/agent/query", data={"query": question, "input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar", "use_cache": "false"}, files={"primary_image": (optical.name, optical.read_bytes()), "secondary_image": (sar.name, sar.read_bytes())}).json()
            answers = [optical_result.get("answer"), sar_result.get("answer"), fused_result.get("answer")]
            correct = [normalize_answer(value) == normalize_answer(reference) if reference is not None else None for value in answers]
            rows.append({
                "sample_id": record.get("id", index), "task": task, "question": question, "reference_answer": reference,
                "optical_answer": answers[0], "sar_answer": answers[1], "fused_answer": answers[2],
                "optical_correct": correct[0], "sar_correct": correct[1], "fused_correct": correct[2],
                "optical_specialists": ";".join((optical_result.get("execution") or {}).get("selected_tools", [])),
                "sar_specialists": ";".join((sar_result.get("execution") or {}).get("selected_tools", [])),
                "fused_specialists": ";".join((fused_result.get("execution") or {}).get("selected_tools", [])),
                "fused_fallback": bool((fused_result.get("change_engine") or {}).get("fallback_used")),
                "evidence_sources": ";".join(sorted({fact.get("source", "") for fact in ((fused_result.get("cross_modal_analysis") or {}).get("evidence_facts") or [])})),
            })
        except Exception as error:
            failures.append({"sample_id": record.get("id", index), "error": f"{type(error).__name__}: {error}"[:500]})
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "failures.csv", failures, ("sample_id", "error"))
    gain = fusion_gain(rows) if not failures else None
    summary = {"samples": len(records), "completed": len(rows), "failures": len(failures), "fusion_gain": gain, "status": "verified_test" if not failures and gain is not None else "operational_only" if not failures else "unavailable"}
    if gain is not None:
        for path in ("optical", "sar", "fused"):
            summary[f"{path}_accuracy"] = sum(bool(row[f"{path}_correct"]) for row in rows) / len(rows)
    write_json(output / "summary.json", summary)
    metrics = [
        {"name": name, "value": summary.get(name), "unit": "ratio", "primary": name == "fusion_gain", "definition": "Same-sample normalized exact-answer comparison."}
        for name in ("optical_accuracy", "sar_accuracy", "fused_accuracy", "fusion_gain")
    ]
    write_json(output / "benchmark_record.json", benchmark_record(
        benchmark_id="optical-sar.controlled.v1", specialist_id="cross_modal_optical_sar_analyzer",
        display_name="Controlled Optical + SAR comparison", task="cross_modal_analysis",
        model_name="SatQuery source-aware evidence fusion", model_version="1.1",
        checkpoint=None, checkpoint_sha256=None, dataset=manifest.stem, split="controlled",
        sample_count=len(rows) if not failures else None, status=summary["status"], metrics=metrics,
        performance={}, artifacts=["predictions.csv", "failures.csv", "summary.json", "report.md"],
        limitations=["Fusion gain is emitted only for the same labelled samples and exact-answer metric.", "No generic fusion accuracy is inferred."],
    ))
    (output / "report.md").write_text("# Optical + SAR controlled evaluation\n\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/optical_sar_controlled"))
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
