#!/usr/bin/env python3
"""Verify required/optional local SatQuery model artifacts without loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from evaluation_common import sha256_file
except ModuleNotFoundError:
    from scripts.evaluation_common import sha256_file


KNOWN = (
    ("SVE adapter", "models/satquery_vision_encoder_v1/satquery_vision_encoder_v1_adapter.pt", "a99c0bf0fb44044988ef1698483888c8a2e3a047d2d2d56478837575cf7626ea", True),
    ("RSVQA specialist", "models/rsvqa_specialist_v1/rsvqa_specialist_v1_head.pt", "71c0ab56ee650813bd495e8a3bc777353b6907a097af860e417f60523efe56ad", True),
    ("Grounding specialist", "models/grounding_specialist_v1_1/grounding_specialist_v1_1_head.pt", "5e8db30becadb1d063fc0154614ee2a2a4b7a2c2923db3fce4ff036c65007342", True),
    ("ChangerEx", "models/changerex/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth", "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618", True),
    # Legacy colourization checkpoints intentionally live one level above the
    # application directory, matching backend.py's ROOT_DIR resolution.
    ("SARFusionFormer", "../models/checkpoints/sarfusionformer_256_decoder_best.pt", None, False),
    ("Color corrector", "../models/checkpoints/color_corrector_256_best.pt", None, False),
)


def verify(root: Path) -> dict:
    rows = []
    for name, relative, expected, required in KNOWN:
        path = root / relative
        actual = sha256_file(path) if path.is_file() and expected else None
        rows.append({"name": name, "path": relative, "required": required, "present": path.is_file(), "expected_sha256": expected, "actual_sha256": actual, "verified": path.is_file() and (expected is None or actual == expected)})
    required_ok = all(row["verified"] for row in rows if row["required"])
    return {"status": "ready" if required_ok else "incomplete", "required_ok": required_ok, "models": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify(args.root.expanduser().resolve())
    print(json.dumps(result, indent=2) if args.json else "\n".join(f"{'OK' if row['verified'] else 'MISSING/UNVERIFIED'}  {row['name']}: {row['path']}" for row in result["models"]))
    raise SystemExit(0 if result["required_ok"] else 1)
