#!/usr/bin/env python3
"""Evaluate existing SAR translation endpoints on one held-out paired manifest."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

try:
    from evaluation_common import benchmark_record, summarize_latencies, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import benchmark_record, summarize_latencies, write_csv, write_json


def lab_prediction_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert normalized ChW/HwC LAB model output through production conversion."""
    import torch
    from sarfusionformer import lab_to_rgb
    value = np.asarray(lab, dtype=np.float32)
    if value.ndim == 3 and value.shape[-1] == 3:
        value = np.moveaxis(value, -1, 0)
    if value.shape[0] != 3:
        raise ValueError("LAB prediction must have three channels")
    tensor = torch.from_numpy(value).unsqueeze(0)
    return np.moveaxis(lab_to_rgb(tensor)[0].detach().cpu().numpy(), 0, -1)


def rgb_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, truth = np.asarray(prediction, dtype=np.float32), np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("RGB prediction and target must have the same HxWx3 shape")
    if pred.max() > 1.0:
        pred = pred / 255.0
    if truth.max() > 1.0:
        truth = truth / 255.0
    pred, truth = np.clip(pred, 0, 1), np.clip(truth, 0, 1)
    mse = float(np.mean((pred - truth) ** 2))
    return {
        "psnr": float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse),
        "ssim": float(structural_similarity(truth, pred, channel_axis=2, data_range=1.0)),
        "mae": float(np.mean(np.abs(pred - truth))),
    }


def evaluate(args: argparse.Namespace) -> dict:
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"SAR translation manifest is unavailable: {manifest}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("SAR translation manifest is empty")
    root, output = args.dataset_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if any(value not in {"pix2pix", "sarfusionformer", "sarfusionformer_color_corrected"} for value in models):
        raise ValueError("models must contain pix2pix, sarfusionformer, or sarfusionformer_color_corrected")
    rows, failures, latencies = [], [], []
    for index, record in enumerate(records):
        target_path = root / record["optical_target"]
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing optical target for record {index}: {target_path}")
        target = np.asarray(Image.open(target_path).convert("RGB"))
        for model in models:
            started = time.perf_counter()
            try:
                if model == "pix2pix":
                    source = root / record["sar_file"]
                    response = client.post("/api/pix2pix/infer", files={"file": (source.name, source.read_bytes()), "ground_truth": (target_path.name, target_path.read_bytes())})
                    payload = response.json()
                    prediction = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload["output"]))).convert("RGB"))
                else:
                    vv, vh = root / record["vv_file"], root / record["vh_file"]
                    response = client.post(
                        "/api/sarfusionformer/infer",
                        data={"apply_color_correction": str(model.endswith("color_corrected")).lower()},
                        files={"vv_file": (vv.name, vv.read_bytes()), "vh_file": (vh.name, vh.read_bytes()), "ground_truth": (target_path.name, target_path.read_bytes())},
                    )
                    payload = response.json()
                    key = "corrected_output" if model.endswith("color_corrected") else "output"
                    prediction = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload[key]))).convert("RGB"))
                if prediction.shape != target.shape:
                    prediction = np.asarray(Image.fromarray(prediction).resize((target.shape[1], target.shape[0]), Image.Resampling.BICUBIC))
                measured = rgb_metrics(prediction, target)
                latency = (time.perf_counter() - started) * 1000.0
                latencies.append(latency)
                rows.append({"sample_id": record.get("id", index), "model": model, **measured, "lpips": None, "latency_ms": latency})
            except Exception as error:
                failures.append({"sample_id": record.get("id", index), "model": model, "error": f"{type(error).__name__}: {error}"[:500]})
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "failures.csv", failures, ("sample_id", "model", "error"))
    summary = {"samples": len(records), "models": {}, "failures": len(failures), "color_space_rule": "SARFusionFormer normalized LAB is converted to sRGB before every metric."}
    for model in models:
        items = [row for row in rows if row["model"] == model]
        summary["models"][model] = {
            name: (sum(float(item[name]) for item in items if math.isfinite(float(item[name]))) / len(items) if items else None)
            for name in ("psnr", "ssim", "mae")
        }
        summary["models"][model]["performance"] = summarize_latencies([item["latency_ms"] for item in items])
        summary["models"][model]["lpips"] = None
    write_json(output / "summary.json", summary)
    metric_rows = []
    for model, values in summary["models"].items():
        for name in ("psnr", "ssim", "mae", "lpips"):
            metric_rows.append({"name": f"{model}_{name}", "value": values[name], "unit": "db" if name == "psnr" else "score", "primary": name in {"ssim", "mae"}, "definition": "Mean paired RGB-space metric; LAB outputs are converted to sRGB first."})
    status = "verified_test" if not failures and rows else "unavailable"
    write_json(output / "benchmark_record.json", benchmark_record(
        benchmark_id="sar-translation.paired.v1", specialist_id="sar_translation_comparison",
        display_name="Paired SAR translation comparison", task="sar_translation",
        model_name="Pix2Pix / SARFusionFormer / optional color corrector", model_version="production",
        checkpoint=None, checkpoint_sha256=None, dataset=manifest.stem, split="held_out",
        sample_count=len(records) if status == "verified_test" else None, status=status,
        metrics=metric_rows, performance=summarize_latencies(latencies),
        artifacts=["per_image_metrics.csv", "failures.csv", "summary.json", "report.md"],
        limitations=["LPIPS remains null when no approved implementation is installed.", "Outputs are optical-like predictions, not optical ground truth."],
    ))
    (output / "report.md").write_text("# SAR translation evaluation\n\n" + json.dumps(summary, indent=2) + "\n\nOutputs are optical-like predictions, not reconstructed optical truth. LPIPS is unavailable unless a separately approved implementation is connected.\n", encoding="utf-8")
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--models", default="pix2pix,sarfusionformer,sarfusionformer_color_corrected")
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/sar_translation_test"))
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
