#!/usr/bin/env python3
"""Exercise a user-supplied preview pair through the real in-process API and exports."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(optical: Path, sar: Path, output: Path) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    fields = {"query": "Where do both modalities agree?", "input_mode": "cross_modal", "primary_modality": "optical", "secondary_modality": "sar", "primary_observation_role": "optical", "secondary_observation_role": "sar", "use_cache": "false"}
    files = {"primary_image": (optical.name, optical.read_bytes()), "secondary_image": (sar.name, sar.read_bytes())}
    reversed_response = client.post("/api/agent/query", data=fields, files={"primary_image": files["secondary_image"], "secondary_image": files["primary_image"]})
    reversed_result = reversed_response.json()
    assert reversed_response.status_code == 200 and reversed_result["status"] == "failed", reversed_result
    assert reversed_result["pair_compatibility"]["role_validation_status"] == "mismatch"
    response = client.post("/api/agent/query", data=fields, files=files)
    body = response.json()
    assert response.status_code == 200 and body["status"] == "success", body
    result = body["cross_modal_analysis"]
    assert body["result_status"] == "COMPLETED" and result["analysis_level"] == "qualitative"
    assert result["statistics"] is None and not result["quantitative_metrics_available"]
    assert "%" not in body["answer"] and "geospatial co-registration cannot be independently verified" in body["answer"]
    assert result["optical_preparation"] and result["sar_preparation"]
    observations = {item["observation_role"]: item for item in (body["primary_image_metadata"], body["secondary_image_metadata"])}
    output.mkdir(parents=True, exist_ok=True)
    (output / "response.json").write_text(json.dumps(body, indent=2))
    (output / "reversed-validation.json").write_text(json.dumps(reversed_result, indent=2))
    for item in result["evidence_products"]:
        assert item["source_observation_id"] == observations[item["source_role"]]["file_id"]
        if item["evidence_type"] == "source":
            assert item["reference"] == observations[item["source_role"]]["preview_url"]
        content = client.get(item["reference"])
        assert content.status_code == 200
        (output / f'{item["source_role"]}-{item["evidence_type"]}.png').write_bytes(content.content)
    report_response = client.post("/api/agent/report", json={"request_id": body["request_id"], "formats": ["pdf", "json", "zip"]})
    assert report_response.status_code == 200, report_response.text
    for item in report_response.json()["artifacts"]:
        download = client.get(item["url"])
        assert download.status_code == 200
        (output / f'mission-report.{item["format"]}').write_bytes(download.content)
        if item["format"] == "json":
            exported = download.json()["input_summary"]["observations_by_role"]
            assert set(exported) == set(observations)
            for role, observation in observations.items():
                for key in ("file_id", "observation_role", "auto_detected_modality", "effective_modality", "preview_url", "safe_name"):
                    assert exported[role][key] == observation[key]
    quantitative = client.post("/api/agent/query", data={**fields, "query": "What percentage overlaps exactly?"}, files=files).json()
    assert quantitative["status"] == "partial" and quantitative["cross_modal_analysis"]["statistics"] is None
    health = client.get("/api/agent/health").json()
    summary = {"status": "passed", "request_id": body["request_id"], "optical_file": optical.name, "sar_file": sar.name, "roles_validated": True, "reversed_blocked": True, "analysis_level": result["analysis_level"], "result_status": body["result_status"], "agreement_count": len(result["agreements"]), "evidence_products": len(result["evidence_products"]), "quantitative_request": quantitative["status"], "report_formats": [item["format"] for item in report_response.json()["artifacts"]], "health": health, "browser_visual_verification": "separate manual check required"}
    (output / "smoke.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--sar", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/optical_sar_preview_smoke"))
    args = parser.parse_args()
    result = run(args.optical, args.sar, args.output)
    print(f'Optical-SAR preview smoke: {result["status"]}; {result["evidence_products"]} native products; {result["result_status"]}')
