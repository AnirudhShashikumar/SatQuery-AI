#!/usr/bin/env python3
"""Evaluate the existing pure-PyTorch ChangerEx on a complete LEVIR-CD split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from evaluation_common import benchmark_record, binary_confusion, binary_metrics, environment_record, sha256_file, summarize_latencies, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import benchmark_record, binary_confusion, binary_metrics, environment_record, sha256_file, summarize_latencies, write_csv, write_json


EXPECTED_CHECKPOINT = "ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth"
EXPECTED_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"


def discover_split(root: Path, split: str) -> list[tuple[str, Path, Path, Path]]:
    candidates = [root / split, root / "LEVIR-CD" / split, root]
    for base in candidates:
        a_dir, b_dir = base / "A", base / "B"
        label_dir = base / "label"
        if not label_dir.is_dir():
            label_dir = base / "labels"
        if all(path.is_dir() for path in (a_dir, b_dir, label_dir)):
            samples = []
            for before in sorted(path for path in a_dir.iterdir() if path.is_file()):
                after, label = b_dir / before.name, label_dir / before.name
                if not after.is_file() or not label.is_file():
                    raise RuntimeError(f"Incomplete LEVIR-CD split; missing paired file for {before.name}")
                samples.append((before.stem, before, after, label))
            if not samples:
                raise RuntimeError(f"LEVIR-CD {split} split contains no samples")
            return samples
    raise FileNotFoundError(f"LEVIR-CD split structure not found below {root}; expected {split}/A, {split}/B and {split}/label")


def evaluate(args: argparse.Namespace) -> dict:
    root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LEVIR-CD dataset is unavailable: {root}")
    if checkpoint.name != EXPECTED_CHECKPOINT or not checkpoint.is_file():
        raise FileNotFoundError(f"Official ChangerEx checkpoint is unavailable: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    verification = {"filename": checkpoint.name, "expected_sha256": EXPECTED_SHA256, "actual_sha256": actual_sha, "verified": actual_sha == EXPECTED_SHA256}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "checkpoint_verification.json", verification)
    if not verification["verified"]:
        raise RuntimeError("ChangerEx checkpoint SHA-256 mismatch; benchmark aborted before inference")
    samples = discover_split(root, args.split)

    from changerex_local.inference import predict_change
    from changerex_local.lifecycle import configure_lifecycle

    configure_lifecycle(checkpoint, device=args.device, allow_device_fallback=args.allow_device_fallback)
    global_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    rows, latencies, failures = [], [], []
    qualitative = output / "qualitative_examples"
    qualitative.mkdir(exist_ok=True)
    started = time.perf_counter()
    cold_load_ms = None
    peak_memory = None
    for index, (sample_id, before, after, label_path) in enumerate(samples):
        sample_started = time.perf_counter()
        try:
            result = predict_change(before, after, device=args.device, threshold=args.threshold, allow_device_fallback=args.allow_device_fallback)
            target = (np.asarray(Image.open(label_path).convert("L")) > 0).astype(np.uint8)
            counts = binary_confusion(result.binary_mask, target)
            metrics = binary_metrics(counts)
            for key in global_counts:
                global_counts[key] += counts[key]
            latency = (time.perf_counter() - sample_started) * 1000.0
            latencies.append(latency)
            cold_load_ms = cold_load_ms if cold_load_ms is not None else result.runtime.load_seconds * 1000.0
            if result.runtime.peak_memory_mb is not None:
                peak_memory = max(peak_memory or 0.0, result.runtime.peak_memory_mb)
            rows.append({
                "sample_id": sample_id, **counts, **metrics,
                "predicted_changed_percent": result.changed_percentage,
                "ground_truth_changed_percent": float(target.mean() * 100.0),
                "latency_ms": latency,
                "device": result.selected_device,
            })
            if index < args.qualitative_count:
                Image.fromarray(result.binary_mask * 255).save(qualitative / f"{sample_id}_prediction.png")
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"[:500]})
    total_seconds = time.perf_counter() - started
    if failures:
        write_csv(output / "failures.csv", failures)
        raise RuntimeError(f"LEVIR-CD evaluation incomplete: {len(failures)} of {len(samples)} samples failed; no verified result emitted")

    metrics = binary_metrics(global_counts)
    summary = {
        "dataset": "LEVIR-CD", "split": args.split, "samples": len(samples),
        "checkpoint_sha256": actual_sha, "threshold": args.threshold,
        "global_confusion": global_counts, "global_metrics": metrics,
        "performance": {**summarize_latencies(latencies), "cold_load_ms": cold_load_ms, "first_inference_ms": latencies[0] if latencies else None, "peak_memory_mb": peak_memory, "total_seconds": total_seconds},
        "status": "verified_test" if args.split == "test" else "verified_validation",
    }
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "per_image_metrics.csv", rows)
    write_json(output / "results.json", summary)
    write_json(output / "environment.json", environment_record())
    metric_rows = [
        {"name": name, "value": value, "unit": "ratio", "primary": name in {"f1", "iou"}, "definition": "Global pixel-confusion metric."}
        for name, value in metrics.items()
    ]
    record = benchmark_record(
        benchmark_id=f"changerex.r18.levircd-{args.split}", specialist_id="changerex_change_detector",
        display_name="ChangerEx ResNet-18 LEVIR-CD evaluation", task="change_detection",
        model_name="ChangerEx ResNet-18", model_version="Open-CD official pure-PyTorch extraction",
        checkpoint=EXPECTED_CHECKPOINT, checkpoint_sha256=actual_sha,
        dataset="LEVIR-CD", split=args.split, sample_count=len(samples), status=summary["status"],
        metrics=metric_rows, performance=summary["performance"],
        artifacts=["results.json", "predictions.csv", "per_image_metrics.csv", "environment.json", "checkpoint_verification.json", "benchmark_report.md"],
        limitations=["LEVIR-CD primarily represents building change.", "Model output is not ground truth outside this labelled evaluation."],
    )
    write_json(output / "benchmark_record.json", record)
    report = [
        "# ChangerEx LEVIR-CD benchmark", "", f"Status: **{summary['status']}**", f"Samples: **{len(samples)}**", "",
        "## Global pixel metrics", "",
        *(f"- {name}: {value:.6f}" for name, value in metrics.items()), "",
        "The checkpoint was SHA-256 verified before inference. All discovered split samples completed; deterministic-overlap evidence is not used as accuracy.",
    ]
    (output / "benchmark_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    value.add_argument("--checkpoint", type=Path, default=root / "models" / "changerex" / EXPECTED_CHECKPOINT)
    value.add_argument("--output-dir", type=Path, default=root / "artifacts" / "changerex_levircd_test")
    value.add_argument("--threshold", type=float, default=0.5)
    value.add_argument("--qualitative-count", type=int, default=12)
    value.add_argument("--allow-device-fallback", action="store_true")
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
