#!/usr/bin/env python3
"""Normalize existing SatQuery artifacts without running inference or inferring values."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from benchmark_data import validate_result, write_json
except ModuleNotFoundError:  # Imported as scripts.import_existing_benchmarks in tests.
    from scripts.benchmark_data import validate_result, write_json


SCHEMA_VERSION = "1.0.0"
SUITE_GENERATED_AT = "2026-09-01T14:15:53Z"


def load(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def metric(metric_id: str, label: str, value: float | int | None, unit: str, description: str, *, primary: bool = False, higher: bool | None = True) -> dict[str, Any]:
    return {"id": metric_id, "label": label, "value": value, "unit": unit, "higher_is_better": higher, "description": description, "primary": primary}


def environment(*, device: str | None = None, hardware: str | None = None, os_name: str | None = None, python: str | None = None, torch: str | None = None, input_size: str | None = None) -> dict[str, Any]:
    return {"device": device, "hardware": hardware, "os": os_name, "python": python, "torch": torch, "input_size": input_size}


def performance(**values: Any) -> dict[str, Any]:
    return {key: values.get(key) for key in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms", "throughput_samples_per_second", "peak_memory_mb", "model_load_ms", "warm_reuse")}


def artifacts(*, report: str | None = None, predictions: str | None = None, per_class_metrics: str | None = None, confusion_matrix: str | None = None, visual_examples: list[str] | None = None) -> dict[str, Any]:
    return {"report": report, "predictions": predictions, "per_class_metrics": per_class_metrics, "confusion_matrix": confusion_matrix, "visual_examples": visual_examples or []}


def provenance(sources: list[str], *, verified: bool, origin: str = "satquery_reproduced") -> dict[str, Any]:
    return {"generated_by": "scripts/import_existing_benchmarks.py", "source_artifacts": sources, "verified": verified, "result_origin": origin}


def unavailable(specialist_id: str, display_name: str, task: str, model_name: str, architecture: str, note: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": f"{specialist_id}.unavailable.v1",
        "specialist_id": specialist_id,
        "display_name": display_name,
        "task": task,
        "model": {"name": model_name, "version": "production", "architecture": architecture, "checkpoint": None, "checkpoint_sha256": None, "training_dataset": None, "adaptation": None, "license": None, "source": None},
        "evaluation": {"status": "unavailable", "dataset": None, "split": None, "sample_count": None, "protocol": "No valid quality benchmark artifact was found during the repository audit.", "timestamp": None, "environment": environment()},
        "metrics": [], "performance": performance(), "breakdowns": [], "artifacts": artifacts(),
        "limitations": ["Operational integration does not establish benchmark quality."],
        "notes": [note, "Smoke tests, if present, are not promoted to accuracy measurements."],
        "provenance": provenance(["docs/BENCHMARK_EVALUATION_MODULE_AUDIT.md"], verified=False, origin="unmeasured"),
    }


def build_records(root: Path) -> list[dict[str, Any]]:
    rsvqa = load(root, "artifacts/rsvqa_full_specialist_v1/results.json")
    rsvqa_state = load(root, "artifacts/rsvqa_full_specialist_v1/run_state.json")
    rsvqa_card = load(root, "models/rsvqa_specialist_v1/model_card.json")
    rsvqa_sha = load(root, "models/rsvqa_specialist_v1/sha256.json")["rsvqa_specialist_v1_head.pt"]
    summary = rsvqa["summary"]
    rsvqa_breakdowns = []
    for task_id, values in summary["by_type"].items():
        rsvqa_breakdowns.append({"dimension": "question_family", "label": task_id, "sample_count": values["samples"], "metrics": [metric("accuracy", "Exact-match accuracy", values["accuracy"], "ratio", "Exact normalized answer match for this question family.", primary=True)]})
    rsvqa_test = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": "rsvqa.specialist-v1.official-test.2026-08-29", "specialist_id": "rsvqa", "display_name": "RSVQA Specialist", "task": "Remote-sensing visual question answering",
        "model": {"name": rsvqa_card["model_name"], "version": rsvqa_card["version"], "architecture": "Frozen OpenCLIP ViT-L-14 + SatQuery Vision Encoder v1 adapter + four task heads", "checkpoint": "rsvqa_specialist_v1_head.pt", "checkpoint_sha256": rsvqa_sha, "training_dataset": "RSVQA-LR official training split", "adaptation": "Presence, comparison, count, and rural/urban task heads", "license": None, "source": "Repository model bundle"},
        "evaluation": {"status": "verified_test", "dataset": "RSVQA-LR", "split": "official test", "sample_count": summary["total_records"], "protocol": "Complete official test manifest through the production RSVQA endpoint; exact normalized answer match, with overflow-aware count accuracy reported separately.", "timestamp": rsvqa_state["completed_at"], "environment": environment(device="local runtime (device not captured)", hardware=None, input_size="Rendered RGB from official test imagery")},
        "metrics": [
            metric("exact_match_accuracy", "Exact-match accuracy", summary["exact_match_accuracy"], "ratio", "Exact normalized answer match over all official test questions.", primary=True),
            metric("overflow_aware_accuracy", "Overflow-aware accuracy", summary["overflow_aware_accuracy"], "ratio", "Count answers above 201 are treated as the exported 201+ label."),
            metric("macro_task_accuracy", "Macro task accuracy", summary["macro_task_accuracy"], "ratio", "Unweighted mean of the four task-family accuracies."),
            metric("coverage", "Answer coverage", summary["coverage"], "ratio", "Fraction of questions receiving a non-null specialist response."),
        ],
        "performance": performance(mean_latency_ms=summary["average_latency_ms"], median_latency_ms=summary["median_latency_ms"], p95_latency_ms=summary["p95_latency_ms"], throughput_samples_per_second=summary["throughput_questions_per_second"], warm_reuse=True),
        "breakdowns": rsvqa_breakdowns,
        "artifacts": artifacts(report="artifacts/rsvqa_full_specialist_v1/benchmark_report.md", predictions="artifacts/rsvqa_full_specialist_v1/predictions.csv", per_class_metrics="artifacts/rsvqa_full_specialist_v1/metrics_by_type.csv", confusion_matrix="artifacts/rsvqa_full_specialist_v1/confusion_comparison.csv", visual_examples=["artifacts/rsvqa_full_specialist_v1/report_assets/per_task_accuracy.png"]),
        "limitations": rsvqa_card["limitations"],
        "notes": ["This is a reproduced official test-split result, not the earlier Smoke-50 snapshot.", "Softmax confidence is not a calibrated probability."],
        "provenance": provenance(["artifacts/rsvqa_full_specialist_v1/results.json", "artifacts/rsvqa_full_specialist_v1/run_state.json", "models/rsvqa_specialist_v1/model_card.json", "models/rsvqa_specialist_v1/sha256.json"], verified=True),
    }

    rsvqa_validation_source = load(root, "models/rsvqa_specialist_v1/validation_metrics.json")
    rv = rsvqa_validation_source["final_validation"]
    rsvqa_validation = {
        **{key: value for key, value in rsvqa_test.items() if key not in {"benchmark_id", "evaluation", "metrics", "performance", "breakdowns", "artifacts", "notes", "provenance"}},
        "benchmark_id": "rsvqa.specialist-v1.validation.epoch8",
        "evaluation": {"status": "verified_validation", "dataset": "RSVQA-LR", "split": "official validation", "sample_count": rv["samples"], "protocol": "Saved epoch-8 validation metrics from specialist training; not a test-set result.", "timestamp": "2026-08-29T08:38:04Z", "environment": environment(device="training validation device (not captured)", input_size="224×224 RGB")},
        "metrics": [metric("overall_accuracy", "Overall accuracy", rv["overall_accuracy"], "ratio", "Validation accuracy at the selected epoch-8 checkpoint.", primary=True), metric("macro_task_accuracy", "Macro task accuracy", rv["macro_accuracy"], "ratio", "Unweighted validation task-family mean.")],
        "performance": performance(peak_memory_mb=rsvqa_validation_source["peak_gpu_memory_gib"] * 1024),
        "breakdowns": [{"dimension": "question_family", "label": label, "sample_count": None, "metrics": [metric("accuracy", "Accuracy", value, "ratio", "Validation accuracy for this task family.", primary=True)]} for label, value in rv["per_task_accuracy"].items()],
        "artifacts": artifacts(report="models/rsvqa_specialist_v1/model_card.json", predictions="models/rsvqa_specialist_v1/validation_predictions.csv", per_class_metrics="models/rsvqa_specialist_v1/validation_metrics.json"),
        "notes": ["Validation result retained as historical context; the official test result is preferred."],
        "provenance": provenance(["models/rsvqa_specialist_v1/validation_metrics.json", "models/rsvqa_specialist_v1/model_card.json"], verified=True),
    }

    grounding = load(root, "models/grounding_specialist_v1_1/validation_results.json")
    grounding_card = load(root, "models/grounding_specialist_v1_1/model_card.json")
    grounding_sha = load(root, "models/grounding_specialist_v1_1/sha256.json")["grounding_specialist_v1_1_head.pt"]
    go = grounding["overall"]
    grounding_validation = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": "grounding.specialist-v1-1.vrsbench-validation.step600", "specialist_id": "grounding", "display_name": "Grounding DINO + Grounding Specialist", "task": "Referring-expression grounding",
        "model": {"name": grounding["model"], "version": "1.1", "architecture": "Grounding DINO Tiny backbone + SatQuery learned proposal-selection head", "checkpoint": "grounding_specialist_v1_1_head.pt", "checkpoint_sha256": grounding_sha, "training_dataset": "VRSBench grounding_v2 training split", "adaptation": "Training recipe v1.1, balanced sampling, area and query-entropy regularization, step 600", "license": "Apache-2.0 base model; specialist bundle repository terms", "source": "IDEA-Research/grounding-dino-tiny + repository specialist bundle"},
        "evaluation": {"status": "verified_validation", "dataset": "VRSBench grounding_v2", "split": "validation", "sample_count": go["samples"], "protocol": "Full saved validation run using highest specialist query logit; one predicted box per sample.", "timestamp": "2026-08-30T06:20:42Z", "environment": environment(device="GPU (model artifact does not identify model)", hardware=None, input_size="Grounding DINO processor input")},
        "metrics": [metric("mean_iou", "Mean IoU", go["mean_iou"], "ratio", "Mean intersection-over-union with one ground-truth box.", primary=True), metric("median_iou", "Median IoU", go["median_iou"], "ratio", "Median intersection-over-union."), metric("accuracy_at_025", "Accuracy @ IoU 0.25", go["accuracy_at_025"], "ratio", "Fraction with IoU at least 0.25."), metric("accuracy_at_050", "Accuracy @ IoU 0.50", go["accuracy_at_050"], "ratio", "Fraction with IoU at least 0.50.", primary=True), metric("accuracy_at_075", "Accuracy @ IoU 0.75", go["accuracy_at_075"], "ratio", "Fraction with IoU at least 0.75."), metric("top_1_query_usage", "Top-1 query usage", go["top_1_query_usage"], "ratio", "Share of samples assigned to the most-used query; lower is more diverse.", higher=False), metric("top_5_query_coverage", "Top-5 query coverage", go["top_5_query_coverage"], "ratio", "Share assigned to the five most-used queries; descriptive only.", higher=None)],
        "performance": performance(mean_latency_ms=go["average_latency_ms"], median_latency_ms=go["median_latency_ms"], p95_latency_ms=go["p95_latency_ms"], throughput_samples_per_second=go["throughput_samples_per_second"], peak_memory_mb=go["peak_gpu_memory_gib"] * 1024, warm_reuse=True),
        "breakdowns": [{"dimension": "object_class", "label": label, "sample_count": values["samples"], "metrics": [metric("mean_iou", "Mean IoU", values["mean_iou"], "ratio", "Per-class mean IoU."), metric("accuracy_at_050", "Accuracy @ IoU 0.50", values["accuracy_at_050"], "ratio", "Per-class IoU@0.50 accuracy.", primary=True)]} for label, values in grounding["per_class"].items()],
        "artifacts": artifacts(report="models/grounding_specialist_v1_1/model_card.json", per_class_metrics="models/grounding_specialist_v1_1/validation_results.json"),
        "limitations": grounding_card["limitations"], "notes": ["This full validation result is preferred over the Smoke-100 pilot.", "Confidence values are not presented as calibrated probabilities."],
        "provenance": provenance(["models/grounding_specialist_v1_1/validation_results.json", "models/grounding_specialist_v1_1/model_card.json", "models/grounding_specialist_v1_1/sha256.json"], verified=True),
    }

    baseline = load(root, "artifacts/vrsbench_grounding_smoke_baseline/results.json")
    baseline_state = load(root, "artifacts/vrsbench_grounding_smoke_baseline/run_state.json")
    bs = baseline["summary"]
    grounding_smoke = {
        **{key: value for key, value in grounding_validation.items() if key not in {"benchmark_id", "model", "evaluation", "metrics", "performance", "breakdowns", "artifacts", "limitations", "notes", "provenance"}},
        "benchmark_id": "grounding.dino-tiny.vrsbench-smoke100.baseline", "model": {**grounding_validation["model"], "name": "IDEA-Research/grounding-dino-tiny", "version": "production baseline", "architecture": "Grounding DINO Tiny", "checkpoint": "IDEA-Research/grounding-dino-tiny", "checkpoint_sha256": None, "training_dataset": None, "adaptation": None},
        "evaluation": {"status": "smoke_only", "dataset": "VRSBench", "split": "prepared Grounding Smoke-100 subset", "sample_count": bs["samples"], "protocol": "Best-IoU benchmark matching on a curated 100-record smoke subset using unchanged production thresholds.", "timestamp": baseline_state["completed_at"], "environment": environment(device="CPU", hardware="Local Apple Silicon Mac", input_size="Grounding DINO processor input")},
        "metrics": [metric("mean_iou", "Mean IoU", bs["mean_iou_all"], "ratio", "Smoke-subset mean IoU with null predictions counted as zero.", primary=True), metric("accuracy_at_050", "Accuracy @ IoU 0.50", bs["accuracy_at_0.50"], "ratio", "Smoke-subset accuracy; not a full benchmark claim.", primary=True), metric("null_rate", "Null-prediction rate", bs["null_prediction_rate"], "ratio", "Share with no accepted production prediction.", higher=False)],
        "performance": performance(mean_latency_ms=bs["average_latency_ms"], median_latency_ms=bs["median_latency_ms"], p95_latency_ms=bs["p95_latency_ms"], throughput_samples_per_second=bs["throughput_samples_per_second"], warm_reuse=True), "breakdowns": [],
        "artifacts": artifacts(report="artifacts/vrsbench_grounding_smoke_baseline/benchmark_report.md", predictions="artifacts/vrsbench_grounding_smoke_baseline/predictions.csv", per_class_metrics="artifacts/vrsbench_grounding_smoke_baseline/metrics_by_class.csv", visual_examples=["artifacts/vrsbench_grounding_smoke_baseline/report_assets/iou_distribution.png"]),
        "limitations": ["This 100-record smoke subset proves operational behavior and supports diagnosis, but is not the full validation split."], "notes": ["Production thresholds were unchanged.", "Best-IoU matching is for analysis and is not production selection."],
        "provenance": provenance(["artifacts/vrsbench_grounding_smoke_baseline/results.json", "artifacts/vrsbench_grounding_smoke_baseline/run_state.json"], verified=True),
    }

    changer = load(root, "artifacts/changerex_local_smoke/mps/result.json")
    changer_env = load(root, "artifacts/changerex_local_smoke/mps/environment.json")
    cp = changer["checkpoint_provenance"]
    cb = changer["benchmark"]
    changer_record = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": "changerex.r18.hanford-mps-operational", "specialist_id": "changerex", "display_name": "ChangerEx", "task": "Bi-temporal change detection",
        "model": {"name": cp["model_name"], "version": "Open-CD v1.1.0 extraction", "architecture": "ChangerEx + IA-ResNetV1c-18", "checkpoint": cp["checkpoint_filename"], "checkpoint_sha256": cp["checkpoint_sha256"], "training_dataset": cp["training_dataset"], "adaptation": "Exact pure-PyTorch extraction; no SatQuery retraining", "license": "Open-CD and checkpoint source terms", "source": cp["checkpoint_url"]},
        "evaluation": {"status": "operational_only", "dataset": "NASA/USGS Hanford image pair", "split": "single operational smoke pair; no ground-truth mask", "sample_count": 1, "protocol": "Five warm repeated inferences plus CPU/MPS parity. No accuracy metric is computed because no reference mask exists.", "timestamp": "2026-09-01T07:44:33Z", "environment": environment(device="MPS float32", hardware="Apple Silicon Mac", os_name=f"macOS {changer_env['macos_version']}", python=changer_env["python"].splitlines()[0], torch=changer_env["torch"], input_size="1024×768 resized/padded pair")},
        "metrics": [metric("deterministic_repeatability", "Repeatability", 1.0 if cb["deterministic_repeatability"] else 0.0, "ratio", "Agreement across repeated probability-map inference; operational, not accuracy.", primary=True)],
        "performance": performance(mean_latency_ms=cb["mean_seconds"] * 1000, median_latency_ms=cb["median_seconds"] * 1000, p95_latency_ms=cb["p95_seconds"] * 1000, peak_memory_mb=changer["runtime"]["peak_memory_mb"], model_load_ms=changer["runtime"]["load_seconds"] * 1000, warm_reuse=True), "breakdowns": [],
        "artifacts": artifacts(report="artifacts/changerex_local_smoke/mps/benchmark_report.md", visual_examples=["artifacts/changerex_local_smoke/mps/overlay.png", "artifacts/changerex_local_smoke/mps/probability_map.png"]),
        "limitations": changer["limitations"] + ["This smoke pair has no pixel-level ground truth and cannot establish precision, recall, F1, IoU, or accuracy."], "notes": ["MPS/CPU parity passed; parity is implementation evidence, not task-quality evidence."],
        "provenance": provenance(["artifacts/changerex_local_smoke/mps/result.json", "artifacts/changerex_local_smoke/mps/environment.json", "artifacts/changerex_local_smoke/mps_vs_cpu_parity.json", "artifacts/changerex_local_smoke/source_manifest.json"], verified=True),
    }

    sar = load(root, "artifacts/single_sar_translation_smoke/comparison.json")
    pix2pix = {
        "schema_version": SCHEMA_VERSION, "benchmark_id": "sar-translation.pix2pix.single-image-operational", "specialist_id": "sar_translation_pix2pix", "display_name": "SAR Translation · Pix2Pix", "task": "SAR-to-optical-like translation",
        "model": {"name": "Pix2Pix", "version": "production checkpoint", "architecture": "Conditional GAN generator", "checkpoint": "pix2pix_generator.pth", "checkpoint_sha256": None, "training_dataset": None, "adaptation": "Production SAR translation pipeline; Color Corrector is an optional pipeline component", "license": None, "source": "Repository production model"},
        "evaluation": {"status": "operational_only", "dataset": "Repository single-SAR smoke input", "split": "3 operational requests; no paired ground truth", "sample_count": sar["completed_requests"], "protocol": "Production single-image SAR workflow smoke with shared-model reuse. No paired reference metrics were computed.", "timestamp": "2026-08-30T15:26:29Z", "environment": environment(device="local runtime", hardware="Apple Silicon Mac")},
        "metrics": [], "performance": performance(mean_latency_ms=sar["translation_runtime_ms"], peak_memory_mb=sar["peak_memory_mb"], warm_reuse=True), "breakdowns": [],
        "artifacts": artifacts(report="artifacts/single_sar_translation_smoke/benchmark_report.md", visual_examples=["artifacts/single_sar_translation_smoke/generated_preview.png"]),
        "limitations": ["The operational smoke has no paired optical ground truth, so PSNR, SSIM, LPIPS, L1, and FID are unavailable.", "Generated optical-like imagery must not be treated as observation."], "notes": ["Color Corrector is represented only as a pipeline component and has no standalone benchmark."],
        "provenance": provenance(["artifacts/single_sar_translation_smoke/comparison.json", "artifacts/single_sar_translation_smoke/benchmark_report.md"], verified=True),
    }

    records = [
        rsvqa_test, rsvqa_validation, grounding_validation, grounding_smoke, changer_record, pix2pix,
        unavailable("captioning", "Remote-Sensing Captioner", "Image captioning", "Production remote-sensing captioner", "Vision encoder-decoder captioner", "No BLEU, METEOR, ROUGE-L, CIDEr, or SPICE artifact was found."),
        unavailable("sar_translation_sarfusionformer", "SAR Translation · SARFusionFormer", "SAR-to-optical-like translation", "SARFusionFormer", "Transformer-based SAR fusion model", "Existing files prove workflow availability only; no verified paired quality result was found."),
        unavailable("cross_modal", "Optical + SAR Fusion", "Cross-modal evidence fusion", "SatQuery cross-modal evidence pipeline", "Optical and SAR specialist evidence fusion", "No controlled, ground-truthed fusion-quality evaluation was found."),
    ]
    for record in records:
        validate_result(record)
    return records


def source_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def output_path(root: Path, record: dict[str, Any]) -> Path:
    groups = {
        "rsvqa": "rsvqa", "captioning": "captioning", "grounding": "grounding",
        "changerex": "changerex", "sar_translation_pix2pix": "sar_translation",
        "sar_translation_sarfusionformer": "sar_translation", "cross_modal": "cross_modal",
    }
    return root / "benchmarks" / "results" / groups[record["specialist_id"]] / f"{record['benchmark_id']}.json"


def build_suite(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = {
        "rsvqa": "rsvqa.specialist-v1.official-test.2026-08-29",
        "grounding": "grounding.specialist-v1-1.vrsbench-validation.step600",
        "changerex": "changerex.r18.hanford-mps-operational",
        "sar_translation_pix2pix": "sar-translation.pix2pix.single-image-operational",
        "captioning": "captioning.unavailable.v1",
        "sar_translation_sarfusionformer": "sar_translation_sarfusionformer.unavailable.v1",
        "cross_modal": "cross_modal.unavailable.v1",
    }
    preferred_records = {key: next(record for record in records if record["benchmark_id"] == value) for key, value in preferred.items()}
    coverage = {key: 0 for key in ("verified_test", "verified_validation", "partial_validation", "smoke_only", "operational_only", "external_reported", "unavailable")}
    for record in preferred_records.values():
        coverage[record["evaluation"]["status"]] += 1
    return {
        "schema_version": SCHEMA_VERSION, "suite_version": "1.0.0", "name": "SatQuery AI Evidence-Based Benchmark Suite", "generated_at": SUITE_GENERATED_AT, "source_commit": source_commit(root),
        "specialists": list(preferred.keys()), "benchmark_record_ids": sorted(record["benchmark_id"] for record in records), "preferred_record_by_specialist": preferred, "coverage": coverage,
        "known_gaps": ["No verified caption quality benchmark.", "No paired SAR translation quality benchmark for Pix2Pix or SARFusionFormer.", "No controlled ground-truthed Optical + SAR fusion benchmark.", "ChangerEx has operational parity and runtime evidence but no SatQuery-reproduced LEVIR-CD quality benchmark."],
        "reproducibility_notes": ["All displayed numbers are copied from named repository artifacts.", "Smoke and operational records are never promoted to test accuracy.", "Missing values remain null; unavailable records contain no fabricated metrics."],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    records = build_records(root)
    for record in records:
        write_json(output_path(root, record), record)
    write_json(root / "benchmarks" / "suites" / "satquery_benchmark_suite.json", build_suite(root, records))
    digest = hashlib.sha256("\n".join(sorted(record["benchmark_id"] for record in records)).encode()).hexdigest()[:12]
    print(f"Imported {len(records)} benchmark records (manifest {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
