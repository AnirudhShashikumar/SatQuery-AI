#!/usr/bin/env python3
"""Verify the real TTP-primary production path through existing SatQuery APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE = PROJECT_ROOT / "satquery_agent" / "demo_samples" / "change-before.png"
DEFAULT_AFTER = PROJECT_ROOT / "satquery_agent" / "demo_samples" / "change-after.png"


def _get_json(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _health(session: requests.Session, base_url: str, timeout: float) -> dict[str, Any]:
    payload = _get_json(session, f"{base_url}/api/agent/health", timeout)
    specialist = (payload.get("specialists") or {}).get("ttp_change_detector")
    if not isinstance(specialist, dict):
        raise RuntimeError("Agent health did not contain ttp_change_detector lifecycle state.")
    return specialist


def _submit_pair(
    session: requests.Session,
    base_url: str,
    before: Path,
    after: Path,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with before.open("rb") as earlier, after.open("rb") as later:
        response = session.post(
            f"{base_url}/api/agent/query",
            data={
                "query": "What changed?",
                "input_mode": "bi_temporal",
                "primary_modality": "optical",
                "secondary_modality": "optical",
                "primary_date": "2024-01-01",
                "secondary_date": "2025-01-01",
            },
            files={
                "primary_image": (before.name, earlier, "application/octet-stream"),
                "secondary_image": (after.name, later, "application/octet-stream"),
            },
            timeout=timeout,
        )
    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Change endpoint did not return a JSON object.")
    return payload, wall_ms


def _validate_and_fetch_mask(
    session: requests.Session,
    base_url: str,
    payload: dict[str, Any],
    destination: Path,
    timeout: float,
) -> dict[str, Any]:
    change = payload.get("change_analysis") or payload
    engine = payload.get("change_engine") or change.get("change_engine") or {}
    ttp = payload.get("ttp_result") or change.get("ttp_result") or {}
    statistics = change.get("statistics") or {}
    execution = payload.get("execution") or change.get("execution") or {}
    if payload.get("status") != "success":
        raise RuntimeError(f"Change response status was {payload.get('status')!r}.")
    if engine.get("mode") not in {"hybrid", "ttp"} or engine.get("primary_tool") != "ttp_change_detector":
        raise RuntimeError(f"TTP did not execute as primary; engine={engine!r}")
    if engine.get("fallback_used"):
        raise RuntimeError(f"TTP fell back: {engine.get('fallback_reason')}")
    if ttp.get("status") != "success" or ttp.get("checkpoint") != "epoch_260.pth":
        raise RuntimeError("TTP result did not confirm the production checkpoint.")
    if not isinstance(statistics.get("changed_pixels"), int) or not isinstance(statistics.get("number_of_regions"), int):
        raise RuntimeError("Learned-mask statistics were absent or invalid.")
    steps = execution.get("steps") or []
    inference = next((step for step in steps if step.get("tool") == "ttp_inference"), None)
    if not isinstance(inference, dict) or not (inference.get("parameters") or {}).get("mask_generated"):
        raise RuntimeError("Execution summary did not confirm TTP mask generation.")
    mask_url = (change.get("previews") or {}).get("ttp_raw_mask")
    if not isinstance(mask_url, str):
        raise RuntimeError("TTP raw mask preview was absent.")
    response = session.get(urljoin(base_url + "/", mask_url.lstrip("/")), timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    with Image.open(destination) as image:
        values = np.asarray(image.convert("L"))
    if not set(np.unique(values).tolist()).issubset({0, 1, 255}):
        raise RuntimeError("TTP raw mask artifact was not binary.")
    if int((values > 0).sum()) != int(statistics["changed_pixels"]):
        raise RuntimeError("TTP mask and primary changed-pixel statistics disagreed.")
    parameters = inference.get("parameters") or {}
    return {
        "request_id": payload.get("request_id"),
        "runtime_ms": ttp.get("runtime_ms"),
        "model_load_ms": ttp.get("model_load_ms"),
        "reused_model": ttp.get("reused_model"),
        "device": ttp.get("device"),
        "changed_pixels": statistics.get("changed_pixels"),
        "changed_percentage": statistics.get("percentage_changed"),
        "region_count": statistics.get("number_of_regions"),
        "largest_region_pixels": statistics.get("largest_connected_region"),
        "gpu_allocated_mb": parameters.get("gpu_allocated_mb"),
        "gpu_reserved_mb": parameters.get("gpu_reserved_mb"),
        "gpu_peak_mb": parameters.get("gpu_peak_mb"),
        "mask_sha256": hashlib.sha256(response.content).hexdigest(),
        "execution_tools": [step.get("tool") for step in steps],
    }


def _generate_report(
    session: requests.Session,
    base_url: str,
    request_id: str,
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    response = session.post(
        f"{base_url}/api/agent/report",
        json={"request_id": request_id, "formats": ["json"]},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    artifact = (payload.get("artifacts") or [None])[0]
    if not isinstance(artifact, dict) or not isinstance(artifact.get("url"), str):
        raise RuntimeError("Report endpoint did not return a JSON report artifact.")
    download = session.get(urljoin(base_url + "/", artifact["url"].lstrip("/")), timeout=timeout)
    download.raise_for_status()
    path = output_dir / "ttp_change_report.json"
    path.write_bytes(download.content)
    report = json.loads(download.content)
    serialized = json.dumps(report)
    if "epoch_260.pth" not in serialized or "ttp_change_detector" not in serialized:
        raise RuntimeError("Generated report omitted TTP checkpoint or execution provenance.")
    return {"path": str(path), "size_bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.api_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    started = time.perf_counter()
    initial_health = _health(session, base_url, args.timeout)
    first_payload, first_wall = _submit_pair(session, base_url, args.before, args.after, args.timeout)
    first = _validate_and_fetch_mask(session, base_url, first_payload, args.output_dir / "ttp_mask_first.png", args.timeout)
    health_after_first = _health(session, base_url, args.timeout)
    second_payload, second_wall = _submit_pair(session, base_url, args.before, args.after, args.timeout)
    second = _validate_and_fetch_mask(session, base_url, second_payload, args.output_dir / "ttp_mask_second.png", args.timeout)
    final_health = _health(session, base_url, args.timeout)
    report = _generate_report(session, base_url, str(second["request_id"]), args.output_dir, args.timeout)

    if final_health.get("status") != "ready" or final_health.get("loaded") is not True:
        raise RuntimeError(f"TTP health was not ready after inference: {final_health!r}")
    if int(final_health.get("load_count", -1)) != 1:
        raise RuntimeError(f"Expected exactly one model load, received {final_health.get('load_count')!r}.")
    if second.get("reused_model") is not True:
        raise RuntimeError("Second TTP inference did not report singleton reuse.")
    if int(final_health.get("reuse_count", 0)) < 1:
        raise RuntimeError("TTP health did not record model reuse.")
    if second.get("gpu_peak_mb") is None:
        raise RuntimeError("TTP execution trace did not contain GPU memory metrics.")

    summary = {
        "status": "passed",
        "api_url": base_url,
        "before": str(args.before),
        "after": str(args.after),
        "initial_health": initial_health,
        "health_after_first": health_after_first,
        "final_health": final_health,
        "first": {**first, "wall_runtime_ms": first_wall},
        "second": {**second, "wall_runtime_ms": second_wall},
        "report": report,
        "total_runtime_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    (args.output_dir / "smoke_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "smoke_report.md").write_text(
        "# TTP production smoke\n\n"
        f"- Status: **passed**\n"
        f"- Checkpoint: `{final_health.get('checkpoint')}`\n"
        f"- Device: `{final_health.get('device')}`\n"
        f"- Load count: `{final_health.get('load_count')}`\n"
        f"- Reuse count: `{final_health.get('reuse_count')}`\n"
        f"- First TTP runtime: `{first.get('runtime_ms')} ms`\n"
        f"- Second TTP runtime: `{second.get('runtime_ms')} ms`\n"
        f"- GPU peak memory: `{second.get('gpu_peak_mb')} MB`\n"
        f"- Changed pixels: `{second.get('changed_pixels')}`\n"
        f"- Regions: `{second.get('region_count')}`\n"
        f"- Report: `{report['path']}`\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8010")
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "ttp_production_smoke")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        summary = run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure: dict[str, Any] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "total_runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        try:
            failure["ttp_health"] = _health(requests.Session(), args.api_url.rstrip("/"), args.timeout)
        except Exception as health_error:
            failure["health_error"] = f"{type(health_error).__name__}: {health_error}"
        (args.output_dir / "smoke_results.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (args.output_dir / "smoke_report.md").write_text(
            "# TTP production smoke\n\n"
            "- Status: **failed**\n"
            f"- Error: `{failure['error']}`\n\n"
            f"- Runtime before failure: `{failure['total_runtime_ms']} ms`\n"
            f"- TTP health: `{json.dumps(failure.get('ttp_health'))}`\n\n"
            "The runner does not count deterministic fallback as a learned-engine pass.\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
