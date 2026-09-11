#!/usr/bin/env python3
"""Repeatable local runtime benchmark; this tool does not measure accuracy."""

from __future__ import annotations

import argparse
import os
import platform
import resource
import time
from pathlib import Path

try:
    from evaluation_common import environment_record, summarize_latencies, write_csv, write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import environment_record, summarize_latencies, write_csv, write_json


def benchmark(args: argparse.Namespace) -> dict:
    if args.warm_iterations < 1:
        raise ValueError("--warm-iterations must be a positive integer")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    root = args.root.expanduser().resolve()
    samples = root / "satquery_agent" / "demo_samples"
    workloads = {
        "single_image": ("Describe the land-cover and major objects visible in this image.", "single-optical.png", None, {"input_mode": "single", "primary_modality": "optical"}),
        "grounding": ("Locate the buildings.", "single-optical.png", None, {"input_mode": "single", "primary_modality": "optical"}),
        "sar_single_image": ("Describe the radar scene.", "cross-sar.tif", None, {"input_mode": "single", "primary_modality": "sar"}),
        "optical_sar": ("Where do both modalities agree?", "cross-optical.tif", "cross-sar.tif", {"input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar"}),
        "bitemporal": ("What changed between these two dates?", "change-before.png", "change-after.png", {"input_mode": "bi_temporal", "primary_modality": "optical", "secondary_modality": "optical", "primary_date": "2020-01-01", "secondary_date": "2021-01-01"}),
    }
    rows = []
    for workflow, (query, first_name, second_name, fields) in workloads.items():
        first, second = samples / first_name, samples / second_name if second_name else None
        for iteration in range(args.warm_iterations + 1):
            files = {"primary_image": (first.name, first.read_bytes())}
            if second:
                files["secondary_image"] = (second.name, second.read_bytes())
            started = time.perf_counter()
            rss_before = _peak_rss_mb()
            response = client.post("/api/agent/query", data={"query": query, "use_cache": "false", **fields}, files=files)
            elapsed = (time.perf_counter() - started) * 1000.0
            rss_after = _peak_rss_mb()
            payload = response.json()
            steps = (payload.get("execution") or {}).get("steps") or []
            load_ms = sum(step.get("duration_ms", 0) for step in steps if "load" in step.get("tool", ""))
            specialist_ms = sum(step.get("duration_ms", 0) for step in steps if any(term in step.get("tool", "") for term in ("inference", "analyzer", "fusion", "caption", "ground")))
            rows.append({"workflow": workflow, "iteration": iteration, "phase": "cold_or_first" if iteration == 0 else "warm", "http_status": response.status_code, "end_to_end_ms": elapsed, "model_load_ms": load_ms, "specialist_ms": specialist_ms, "peak_rss_before_mb": rss_before, "peak_rss_after_mb": rss_after, "peak_rss_delta_mb": max(0.0, rss_after - rss_before), "result_status": payload.get("result_status")})
    summaries = {}
    for workflow in workloads:
        items = [row for row in rows if row["workflow"] == workflow]
        warm = [row["end_to_end_ms"] for row in items if row["phase"] == "warm"]
        summaries[workflow] = {"cold_or_first_request_ms": items[0]["end_to_end_ms"], "reported_model_load_ms": items[0]["model_load_ms"], "warm_request": summarize_latencies(warm), "reported_specialist_warm_mean_ms": sum(row["specialist_ms"] for row in items[1:]) / max(len(items) - 1, 1), "peak_rss_after_mb": max(row["peak_rss_after_mb"] for row in items), "peak_rss_delta_mb": max(row["peak_rss_delta_mb"] for row in items)}
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"classification": "runtime benchmark only; no accuracy measured", "environment": environment_record(), "warm_iterations": args.warm_iterations, "workflows": summaries}
    write_json(output / "performance.json", result)
    write_csv(output / "performance.csv", rows)
    (output / "performance_report.md").write_text("# SatQuery local performance benchmark\n\nThis run measures load, first-request, warm specialist, and end-to-end latency. It is not an accuracy benchmark.\n\n" + "\n".join(f"- {name}: first {values['cold_or_first_request_ms']:.1f} ms; warm mean {values['warm_request']['mean_ms']:.1f} ms" for name, values in summaries.items()) + "\n", encoding="utf-8")
    return result


def _peak_rss_mb() -> float:
    """Best-effort process peak RSS; macOS reports bytes, Linux KiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return value / divisor


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/performance"))
    value.add_argument("--warm-iterations", type=int, default=5)
    return value


if __name__ == "__main__":
    benchmark(parser().parse_args())
