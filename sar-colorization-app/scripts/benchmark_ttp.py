"""Run the reproducible TTP/GeoVision benchmark on a configured CUDA deployment.

The manifest is JSON: {"samples": [{"id", "category", "before", "after",
"label" (optional), "modality", "expected_status" (optional)}]}.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


def mask_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("L")) > 0


def fetch_mask(session: requests.Session, base_url: str, path: str | None) -> np.ndarray | None:
    if not path:
        return None
    response = session.get(urljoin(base_url.rstrip("/") + "/", path.lstrip("/")), timeout=(3, 30))
    response.raise_for_status()
    return mask_from_bytes(response.content)


def scored(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and label dimensions differ.")
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "changed_area_error_percentage_points": abs(float(prediction.mean() - truth.mean())) * 100.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    return round(statistics.fmean(values), 6) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(path: Path, tiles: list[tuple[str, list[tuple[str, Image.Image]]]]) -> None:
    if not tiles:
        return
    cell_w, cell_h = 260, 220
    columns = max(len(items) for _, items in tiles)
    canvas = Image.new("RGB", (columns * cell_w, len(tiles) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, (sample_id, items) in enumerate(tiles):
        for column, (label, image) in enumerate(items):
            preview = image.convert("RGB")
            preview.thumbnail((cell_w - 16, cell_h - 42))
            x = column * cell_w + (cell_w - preview.width) // 2
            y = row * cell_h + 28 + (cell_h - 36 - preview.height) // 2
            canvas.paste(preview, (x, y))
            draw.text((column * cell_w + 8, row * cell_h + 7), f"{sample_id} · {label}", fill="black", font=font)
    canvas.save(path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/ttp/latest"))
    args = parser.parse_args()
    samples = json.loads(args.manifest.read_text(encoding="utf-8")).get("samples", [])
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for sample in samples:
        started = time.perf_counter()
        row: dict[str, Any] = {"sample_id": sample["id"], "category": sample["category"], "modality": sample.get("modality", "optical")}
        try:
            before_path, after_path = Path(sample["before"]), Path(sample["after"])
            with before_path.open("rb") as before, after_path.open("rb") as after:
                response = session.post(
                    f"{args.api_url.rstrip('/')}/api/agent/change",
                    files={"before_image": (before_path.name, before, "application/octet-stream"), "after_image": (after_path.name, after, "application/octet-stream")},
                    data={"before_date": "2025-01-01", "after_date": "2025-02-01", "modality": row["modality"]},
                    timeout=(3, 120),
                )
            payload = response.json()
            row.update({
                "http_status": response.status_code,
                "status": payload.get("status") if response.ok else (payload.get("detail") or {}).get("code"),
                "engine_mode": (payload.get("change_engine") or {}).get("mode"),
                "fallback_used": (payload.get("change_engine") or {}).get("fallback_used"),
                "fallback_reason": (payload.get("change_engine") or {}).get("fallback_reason"),
                "ttp_runtime_ms": (payload.get("ttp_result") or {}).get("runtime_ms"),
                "model_reused": (payload.get("ttp_result") or {}).get("reused_model"),
                "mask_iou_between_engines": (payload.get("mask_comparison") or {}).get("iou"),
                "agreement_percentage": (payload.get("mask_comparison") or {}).get("agreement_percentage"),
            })
            expected = sample.get("expected_status")
            if expected and row["status"] != expected:
                raise ValueError(f"Expected status {expected}, received {row['status']}.")
            previews = payload.get("previews") or {}
            ttp_mask = fetch_mask(session, args.api_url, previews.get("ttp_raw_mask"))
            deterministic_mask = fetch_mask(session, args.api_url, previews.get("deterministic_mask"))
            truth = mask_from_bytes(Path(sample["label"]).read_bytes()) if sample.get("label") else None
            if truth is not None and ttp_mask is not None:
                row.update({f"ttp_{key}": value for key, value in scored(ttp_mask, truth).items()})
            if truth is not None and deterministic_mask is not None:
                row.update({f"deterministic_{key}": value for key, value in scored(deterministic_mask, truth).items()})
            images: list[tuple[str, Image.Image]] = [("Before", Image.open(before_path).convert("RGB")), ("After", Image.open(after_path).convert("RGB"))]
            if ttp_mask is not None:
                images.append(("TTP", Image.fromarray(ttp_mask.astype(np.uint8) * 255)))
            if deterministic_mask is not None:
                images.append(("Deterministic", Image.fromarray(deterministic_mask.astype(np.uint8) * 255)))
            if truth is not None:
                images.append(("Label", Image.fromarray(truth.astype(np.uint8) * 255)))
            tiles.append((sample["id"], images))
        except Exception as error:
            row["failure"] = type(error).__name__
            failures.append({"sample_id": sample.get("id"), "category": sample.get("category"), "failure": type(error).__name__, "message": str(error)[:300]})
        row["total_runtime_ms"] = round((time.perf_counter() - started) * 1000)
        rows.append(row)

    categories: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    for category, values in sorted(grouped.items()):
        categories.append({
            "category": category, "sample_count": len(values), "failure_count": sum("failure" in row for row in values),
            **{f"mean_{key}": mean_numeric(values, key) for key in ("ttp_iou", "ttp_precision", "ttp_recall", "ttp_f1", "ttp_false_positive_rate", "ttp_false_negative_rate", "ttp_changed_area_error_percentage_points", "deterministic_iou", "ttp_runtime_ms", "mask_iou_between_engines")},
        })
    summary = {
        "schema_version": "1.0", "sample_count": len(rows), "failure_count": len(failures),
        "successful_hybrid_count": sum(row.get("engine_mode") in {"hybrid", "ttp"} for row in rows),
        "fallback_count": sum(bool(row.get("fallback_used")) for row in rows),
        "model_reuse_count": sum(bool(row.get("model_reused")) for row in rows),
        "mean_ttp_iou": mean_numeric(rows, "ttp_iou"), "mean_deterministic_iou": mean_numeric(rows, "deterministic_iou"),
        "mean_ttp_runtime_ms": mean_numeric(rows, "ttp_runtime_ms"),
        "disclaimer": "TTP outputs are model-generated binary predictions, not ground truth; engine agreement is not accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output / "per_sample.csv", rows)
    write_csv(args.output / "per_category.csv", categories)
    write_csv(args.output / "failure_cases.csv", failures)
    (args.output / "benchmark_report.md").write_text("# TTP benchmark report\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\nAll failures remain listed in `failure_cases.csv`. No universal winner is declared.\n", encoding="utf-8")
    contact_sheet(args.output / "visual_contact_sheet.png", tiles)


if __name__ == "__main__":
    main()
