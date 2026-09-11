#!/usr/bin/env python3
"""Run the checked-in samples through SatQuery's production API in-process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from evaluation_common import write_json
except ModuleNotFoundError:
    from scripts.evaluation_common import write_json


def run(root: Path) -> dict:
    # This is an offline-readiness check. Missing optional Hugging Face assets
    # must trigger the application's documented local fallback immediately,
    # rather than causing network retries that disguise offline behavior.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    samples = root / "satquery_agent" / "demo_samples"
    cases = [
        ("single", "Describe the land-cover and major objects visible in this image.", "single-optical.png", None, {"input_mode": "single", "primary_modality": "optical"}),
        ("optical_sar", "Use the optical and SAR images together to identify built-up and water-covered regions.", "cross-optical.tif", "cross-sar.tif", {"input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar"}),
        ("bitemporal", "What changed between these two dates, and where did the change occur?", "change-before.png", "change-after.png", {"input_mode": "bi_temporal", "primary_modality": "optical", "secondary_modality": "optical", "primary_date": "2020-01-01", "secondary_date": "2021-01-01"}),
    ]
    rows = []
    first_request = None
    for name, query, primary_name, secondary_name, fields in cases:
        primary = samples / primary_name
        files = {"primary_image": (primary.name, primary.read_bytes())}
        if secondary_name:
            secondary = samples / secondary_name
            files["secondary_image"] = (secondary.name, secondary.read_bytes())
        response = client.post("/api/agent/query", data={"query": query, "use_cache": "false", **fields}, files=files)
        payload = response.json()
        first_request = first_request or payload.get("request_id")
        rows.append({
            "workflow": name, "http_status": response.status_code, "status": payload.get("status"),
            "result_status": payload.get("result_status"), "answer_present": bool(payload.get("answer")),
            "evidence_count": len(payload.get("evidence") or []), "trace_steps": len((payload.get("execution") or {}).get("steps") or []),
            "selected_tools": (payload.get("execution") or {}).get("selected_tools", []), "warnings": payload.get("warnings") or [],
        })
    report_formats = []
    if first_request:
        report = client.post("/api/agent/report", json={"request_id": first_request, "formats": ["json"]})
        if report.status_code == 200:
            report_formats = [item["format"] for item in report.json().get("artifacts", [])]
    health = client.get("/api/agent/health").json()
    routes = {route.path for route in app.routes}
    required_routes = {"/api/agent/query", "/api/agent/health", "/api/agent/report", "/api/agent/compliance"}
    ok = all(row["http_status"] == 200 and row["trace_steps"] > 0 for row in rows) and "json" in report_formats and required_routes <= routes
    return {"status": "passed" if ok else "failed", "workflows": rows, "report_formats": report_formats, "specialist_health": health.get("specialists", {}), "required_routes_present": required_routes <= routes, "required_routes": sorted(required_routes)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/offline_smoke.json"))
    args = parser.parse_args()
    result = run(args.root.expanduser().resolve())
    write_json(args.output, result)
    print(f"SatQuery workflow smoke: {result['status']}")
    raise SystemExit(0 if result["status"] == "passed" else 1)
