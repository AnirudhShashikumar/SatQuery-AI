#!/usr/bin/env python3
"""Prove automatic SAR detection, Pix2Pix translation, and query-preserving optical dispatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--query", default="Describe this image.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/single_sar_pix2pix_auto_smoke"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    sample = args.sample.expanduser().resolve()
    if not sample.is_file():
        raise SystemExit(f"Sample does not exist: {sample}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # This smoke intentionally disables the optional general SAR translation branch.
    # Automatic preview routing must still activate the shared Pix2Pix instance.
    os.environ["SATQUERY_SAR_TRANSLATION_ENABLED"] = "0"
    os.environ["SATQUERY_SAR_TRANSLATION_OPTICAL_SPECIALISTS_ENABLED"] = "1"
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from fastapi.testclient import TestClient

    from backend import app

    client = TestClient(app)
    content = sample.read_bytes()
    mime = "image/jpeg" if sample.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    health_before = client.get("/api/agent/health").json()["specialists"]["sar_translation_service"]
    inspection_response = client.post(
        "/api/agent/inspect",
        files={"image": (sample.name, content, mime)},
    )
    inspection_response.raise_for_status()
    inspection = inspection_response.json()
    query_response = client.post(
        "/api/agent/query",
        data={
            "query": args.query,
            "input_mode": "single",
            "primary_modality": "optical",
            "primary_image_modality": "auto",
            "use_cache": "false",
        },
        files={"primary_image": (sample.name, content, mime)},
    )
    query_response.raise_for_status()
    response = query_response.json()
    health_after = client.get("/api/agent/health").json()["specialists"]["sar_translation_service"]

    metadata = response["primary_image_metadata"]
    translated = response.get("sar_translated_optical_analysis") or {}
    steps = response["execution"]["steps"]
    tools = [step["tool"] for step in steps]
    required_trace = {
        "modality_detection",
        "sar_translation_inference",
        "response_generation",
    }
    translated_tools = set(translated.get("optical_specialists_executed") or [])
    checks = {
        "upload_accepted": query_response.status_code == 200,
        "automatically_detected_as_sar": metadata.get("auto_detected_modality") == "sar_preview",
        "no_manual_modality_override": metadata.get("user_confirmed_modality") is None,
        "pix2pix_selected": translated.get("model") == "Pix2Pix",
        "generated_preview_created": bool(translated.get("generated_preview_url")),
        "original_query_processed": bool(translated_tools) and response.get("answer") is not None,
        "normal_response_returned": response.get("status") in {"success", "partial"},
        "required_trace_present": required_trace.issubset(tools) and any(tool.startswith("translated_optical_") for tool in tools),
        "no_failure": not any(step.get("status") == "failed" for step in steps if step.get("tool", "").startswith(("sar_translation_", "translated_optical_"))),
        "no_duplicate_model_load": health_after.get("load_count", 0) <= health_before.get("load_count", 0),
        "shared_model_reused": health_after.get("reuse_count", 0) > health_before.get("reuse_count", 0),
    }

    preview_url = translated.get("generated_preview_url")
    if preview_url:
        preview = client.get(preview_url)
        if preview.status_code == 200:
            (output / "generated_preview.png").write_bytes(preview.content)
        else:
            checks["generated_preview_created"] = False

    input_metadata = {
        "sample": str(sample),
        "query": args.query,
        "manual_modality_override": False,
        "submitted_primary_modality": "optical",
        "submitted_primary_image_modality": "auto",
        "inspection": inspection,
        "health_before": health_before,
        "health_after": health_after,
    }
    detection = {
        "auto_detected_modality": metadata.get("auto_detected_modality"),
        "effective_modality": metadata.get("effective_modality"),
        "confidence": metadata.get("auto_detection_confidence"),
        "reason": metadata.get("auto_detection_reason"),
        "limitations": metadata.get("modality_limitations"),
    }
    _json(output / "input_metadata.json", input_metadata)
    _json(output / "detection_decision.json", detection)
    _json(output / "api_response.json", response)
    _json(output / "execution_timeline.json", steps)
    _json(output / "smoke_checks.json", checks)

    passed = all(checks.values())
    report = [
        "# Single SAR Pix2Pix automatic-routing smoke",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Sample: `{sample}`",
        f"- Original query: `{args.query}`",
        f"- Detection: `{detection['auto_detected_modality']}` ({detection['confidence']})",
        f"- Detection reason: {detection['reason']}",
        f"- Translator: `{translated.get('model')}`",
        f"- Optical specialists: `{', '.join(sorted(translated_tools)) or 'none'}`",
        f"- Response status: `{response.get('status')}` / `{response.get('result_status')}`",
        f"- Shared model load count: {health_before.get('load_count')} → {health_after.get('load_count')}",
        f"- Shared model reuse count: {health_before.get('reuse_count')} → {health_after.get('reuse_count')}",
        "",
        "## Checks",
        "",
        *[f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in checks.items()],
        "",
        "The generated preview is an optical-like Pix2Pix representation, not observed optical truth.",
    ]
    (output / "smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
