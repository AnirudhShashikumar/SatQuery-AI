#!/usr/bin/env python3
"""Run one real in-process SatQuery bi-temporal request through ChangerEx."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--query", default="What changed between the earlier and later satellite images?")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    for path in (arguments.before, arguments.after, arguments.checkpoint):
        if not path.is_file():
            raise SystemExit(f"Required smoke input does not exist: {path}")

    os.environ["SATQUERY_CHANGE_ENGINE"] = "changerex"
    os.environ["SATQUERY_CHANGEREX_ENABLED"] = "true"
    os.environ["SATQUERY_CHANGEREX_CHECKPOINT"] = str(arguments.checkpoint.resolve())
    os.environ["SATQUERY_CHANGEREX_DEVICE"] = arguments.device
    os.environ["SVE_ENABLED"] = "false"
    os.environ["TTP_ENABLED"] = "false"

    # Import only after selecting the production engine so startup observes the
    # exact same environment as the request.
    from fastapi.testclient import TestClient
    from backend import app

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with arguments.before.open("rb") as before_handle, arguments.after.open("rb") as after_handle:
        with TestClient(app) as client:
            response = client.post(
                "/api/agent/query",
                data={
                    "query": arguments.query,
                    "input_mode": "bi_temporal",
                    "primary_modality": "optical",
                    "secondary_modality": "optical",
                    "primary_date": "2000-05-06",
                    "secondary_date": "2000-07-09",
                },
                files={
                    "primary_image": (arguments.before.name, before_handle, "image/png" if arguments.before.suffix.lower() == ".png" else "image/jpeg"),
                    "secondary_image": (arguments.after.name, after_handle, "image/png" if arguments.after.suffix.lower() == ".png" else "image/jpeg"),
                },
            )
            if response.status_code != 200:
                raise SystemExit(f"Smoke request failed ({response.status_code}): {response.text[:1000]}")
            payload = response.json()
            engine = payload.get("change_engine") or {}
            if engine.get("primary_tool") != "changerex_change_detector" or engine.get("fallback_used"):
                raise SystemExit(f"ChangerEx was not the learned engine: {engine}")

            previews = (payload.get("change_analysis") or {}).get("previews") or {}
            saved_previews: dict[str, str] = {}
            for name, url in previews.items():
                if not url or name in {"before", "after"}:
                    continue
                artifact = client.get(url)
                if artifact.status_code != 200:
                    raise SystemExit(f"Preview retrieval failed for {name}: {artifact.status_code}")
                target = output / f"{name}.png"
                target.write_bytes(artifact.content)
                saved_previews[name] = target.name

    (output / "response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    trace_tools = [step.get("tool") for step in (payload.get("execution") or {}).get("steps", [])]
    summary = {
        "status": payload.get("status"),
        "answer": payload.get("answer"),
        "change_engine": payload.get("change_engine"),
        "learned_result": payload.get("ttp_result"),
        "statistics": (payload.get("change_analysis") or {}).get("statistics"),
        "mask_comparison": payload.get("mask_comparison"),
        "execution_trace_tools": trace_tools,
        "saved_previews": saved_previews,
    }
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
