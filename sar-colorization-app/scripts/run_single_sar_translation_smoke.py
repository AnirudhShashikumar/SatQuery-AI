#!/usr/bin/env python3
"""Exercise native and optional translated-evidence single-image SAR paths."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True, help="Approved local SAR PNG/JPEG/TIFF sample")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/single_sar_translation_smoke"))
    parser.add_argument("--query", default="Describe the major radar-backscatter patterns.")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _peak_memory_mb() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most BSD-based CI images report KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(maximum / divisor, 2)


def _request(client: TestClient, sample: Path, query: str, *, enabled: bool, model: str, timeout: float, force_preview: bool = False) -> tuple[dict[str, Any], float]:
    os.environ["SATQUERY_SAR_TRANSLATION_ENABLED"] = "1" if enabled else "0"
    os.environ["SATQUERY_SAR_TRANSLATION_MODEL"] = model
    data = sample.read_bytes()
    mime = mimetypes.guess_type(sample.name)[0] or "application/octet-stream"
    inspect = client.post("/api/agent/inspect", files={"image": (sample.name, data, mime)})
    inspect.raise_for_status()
    metadata = inspect.json()["metadata"]
    bands = int(metadata["band_count"])
    modality = "sar_preview" if force_preview else "sar_vv_vh" if bands == 2 else "sar_preview" if metadata["representation"] == "display_preview" else "sar_vv"
    started = time.perf_counter()
    response = client.post(
        "/api/agent/query",
        data={
            "query": query,
            "input_mode": "single",
            "primary_modality": "sar",
            "primary_image_modality": modality,
            "use_cache": "false",
            "force_rerun": "true",
        },
        files={"primary_image": (sample.name, data, mime)},
        timeout=max(1.0, timeout),
    )
    response.raise_for_status()
    return response.json(), (time.perf_counter() - started) * 1000


def main() -> int:
    args = parse_args()
    if not args.sample.is_file():
        raise SystemExit(f"Sample not found: {args.sample}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    original = {name: os.environ.get(name) for name in ("SATQUERY_SAR_TRANSLATION_ENABLED", "SATQUERY_SAR_TRANSLATION_MODEL")}
    try:
        native, native_runtime = _request(client, args.sample, args.query, enabled=False, model="sarfusionformer", timeout=args.timeout)
        translated, translated_runtime = _request(client, args.sample, args.query, enabled=True, model="pix2pix", timeout=args.timeout)
        # Treat the same local sample as an explicitly confirmed display representation so
        # SARFusionFormer's verified-pair contract rejects it and the compatible Pix2Pix path is exercised.
        fallback, fallback_runtime = _request(client, args.sample, args.query, enabled=True, model="sarfusionformer", timeout=args.timeout, force_preview=True)
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    translated_branch = translated.get("sar_translated_optical_analysis") or {}
    fallback_branch = fallback.get("sar_translated_optical_analysis") or {}
    translation_health = client.get("/api/agent/health").json().get("specialists", {}).get("sar_translation_service", {})
    comparison = {
        "completed_requests": sum(item.get("status") in {"success", "partial"} for item in (native, translated, fallback)),
        "native_runtime_ms": round(native_runtime, 2),
        "translation_enabled_total_runtime_ms": round(translated_runtime, 2),
        "fallback_total_runtime_ms": round(fallback_runtime, 2),
        "translation_runtime_ms": translated_branch.get("runtime_ms"),
        "translator_model_used": translated_branch.get("model"),
        "fallback_model_used": fallback_branch.get("model"),
        "fallback_used": fallback_branch.get("fallback_used"),
        "optical_specialists_executed": translated_branch.get("optical_specialists_executed") or [],
        "agreement_state": translated_branch.get("agreement"),
        "model_load_count": translation_health.get("load_count"),
        "model_reuse_count": translation_health.get("reuse_count"),
        "failures": [
            {"run": name, "result_status": item.get("result_status")}
            for name, item in (("native", native), ("translated", translated), ("fallback", fallback))
            if item.get("status") == "failed"
        ],
        "peak_memory_mb": _peak_memory_mb(),
    }
    _write_json(output / "native_result.json", native)
    _write_json(output / "translated_result.json", translated)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "warnings.json", {
        "native": native.get("warnings") or [],
        "translated": translated.get("warnings") or [],
        "fallback": fallback.get("warnings") or [],
    })
    with (output / "execution_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "index", "tool", "status", "duration_ms"])
        writer.writeheader()
        for run, payload in (("native", native), ("translated", translated), ("fallback", fallback)):
            for index, step in enumerate((payload.get("execution") or {}).get("steps") or [], start=1):
                writer.writerow({"run": run, "index": index, "tool": step.get("tool"), "status": step.get("status"), "duration_ms": step.get("duration_ms")})
    preview_url = translated_branch.get("generated_preview_url")
    if preview_url:
        preview = client.get(preview_url)
        if preview.status_code == 200:
            (output / "generated_preview.png").write_bytes(preview.content)
    report = """# Single-Image SAR Translation Smoke Report

- Completed requests: {completed_requests}/3
- Native runtime: {native_runtime_ms} ms
- Translation-enabled total runtime: {translation_enabled_total_runtime_ms} ms
- Translation runtime: {translation_runtime_ms} ms
- Translator: {translator_model_used}
- Fallback run translator: {fallback_model_used}
- Fallback used: {fallback_used}
- Optical specialists: {optical_specialists_executed}
- Agreement: {agreement_state}
- Translation model loads/reuses: {model_load_count}/{model_reuse_count}
- Failures: {failures}
- Peak process memory: {peak_memory_mb} MB

Generated imagery is learned supporting evidence, not an observed optical measurement.
""".format(**comparison)
    (output / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["completed_requests"] == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
