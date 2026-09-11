#!/usr/bin/env python3
"""Compare Grounding DINO score policies without changing production inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satquery_agent.models import ImageFormat, ImageMetadata, Modality  # noqa: E402
from satquery_agent.specialists.grounder import RemoteSensingGrounder  # noqa: E402


MODES = ("disabled", "fallback", "blend", "specialist_only")
MODE_ENVIRONMENT = "SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE"


def _image_format(path: Path) -> ImageFormat:
    return {
        ".png": ImageFormat.PNG,
        ".jpg": ImageFormat.JPEG,
        ".jpeg": ImageFormat.JPEG,
        ".tif": ImageFormat.TIFF,
        ".tiff": ImageFormat.TIFF,
    }.get(path.suffix.lower(), ImageFormat.UNKNOWN)


def _mime_type(image_format: ImageFormat) -> str:
    return {
        ImageFormat.PNG: "image/png",
        ImageFormat.JPEG: "image/jpeg",
        ImageFormat.TIFF: "image/tiff",
        ImageFormat.GEOTIFF: "image/tiff",
    }.get(image_format, "application/octet-stream")


def build_metadata(path: Path, image: Image.Image) -> ImageMetadata:
    image_format = _image_format(path)
    return ImageMetadata(
        file_id="grounding-mode-comparison",
        original_name=path.name,
        safe_name=path.name,
        format=image_format,
        mime_type=_mime_type(image_format),
        size_bytes=path.stat().st_size,
        width=image.width,
        height=image.height,
        band_count=3,
        dtype="uint8",
        is_georeferenced=False,
        color_interpretation=["r", "g", "b"],
        warnings=[],
    )


def _signature(detections: Sequence[dict[str, Any]]) -> list[tuple[tuple[int, ...], str]]:
    return [
        (tuple(int(value) for value in detection["bbox_pixels"]), str(detection["label"]))
        for detection in detections
    ]


def save_annotated_preview(
    source: Image.Image,
    detections: Sequence[dict[str, Any]],
    mode: str,
    destination: Path,
) -> None:
    canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), f"{mode}: {len(detections)} accepted", fill=(255, 255, 255))
    for detection in detections:
        box = tuple(int(value) for value in detection["bbox_pixels"])
        label = f"{detection['label']} {float(detection['score']):.4f}"
        draw.rectangle(box, outline=(255, 214, 10), width=max(2, canvas.width // 500))
        label_y = max(25, box[1] - 14)
        draw.rectangle((box[0], label_y, min(canvas.width, box[0] + 8 * len(label)), label_y + 13), fill=(0, 0, 0))
        draw.text((box[0] + 2, label_y + 1), label, fill=(255, 255, 255))
    canvas.save(destination, format="PNG")
    canvas.close()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(image_path: Path, query: str, results: Sequence[dict[str, Any]]) -> str:
    disabled = next((result for result in results if result["mode"] == "disabled" and not result.get("error")), None)
    lines = [
        "# Grounding score-mode comparison",
        "",
        f"- Image: `{image_path}`",
        f"- Query: `{query}`",
        "- Grounding backbone: the same lazily loaded Grounding DINO instance for every mode",
        f"- One-time Grounding DINO initialization: `{results[0].get('backbone_initialization_ms', 'unavailable')} ms`",
        "",
        "| Mode | Accepted | Rejected | Top accepted score | Wall runtime (ms) | Same boxes/labels as disabled | Fallback |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        if result.get("error"):
            lines.append(f"| {result['mode']} | error | error | — | {result['wall_runtime_ms']} | — | {result['error']} |")
            continue
        top_score = result["scores"][0] if result["scores"] else None
        same = "baseline" if result["mode"] == "disabled" else str(result["same_geometry_and_labels_as_disabled"]).lower()
        fallback = result.get("fallback_reason") or ("yes" if result.get("fallback_used") else "no")
        lines.append(
            f"| {result['mode']} | {result['accepted_count']} | {result['rejected_count']} | "
            f"{top_score if top_score is not None else '—'} | {result['wall_runtime_ms']} | {same} | {fallback} |"
        )
    lines.extend(["", "## Accepted detections", ""])
    for result in results:
        lines.append(f"### {result['mode']}")
        lines.append("")
        if result.get("error"):
            lines.append(f"Error: `{result['error']}`")
        elif not result["detections"]:
            lines.append("No accepted detections.")
        else:
            lines.extend([
                "| Label | Score | Pixel box | Normalized box |",
                "|---|---:|---|---|",
            ])
            for detection in result["detections"]:
                lines.append(
                    f"| {detection['label']} | {float(detection['score']):.6f} | "
                    f"`{detection['bbox_pixels']}` | `{detection['bbox_normalized']}` |"
                )
        diagnostics = result.get("score_diagnostics", [])
        if diagnostics:
            lines.extend([
                "",
                "Score-policy diagnostics:",
                "",
                "| Proposal | Original DINO | Specialist | Final | Calibration | Fallback reason |",
                "|---:|---:|---:|---:|---|---|",
            ])
            for diagnostic in diagnostics:
                lines.append(
                    f"| {diagnostic['proposal_index']} | {diagnostic['original_grounding_score']} | "
                    f"{diagnostic['specialist_score'] if diagnostic['specialist_score'] is not None else '—'} | "
                    f"{diagnostic['final_score']} | {diagnostic['calibration_method'] or '—'} | "
                    f"{diagnostic['fallback_reason'] or '—'} |"
                )
        lines.extend(["", f"![{result['mode']} preview](preview_{result['mode']}.png)", ""])
    if disabled is not None:
        fallback = next((result for result in results if result["mode"] == "fallback" and not result.get("error")), None)
        if fallback is not None:
            preserved = (
                fallback["accepted_count"] >= disabled["accepted_count"]
                and fallback["same_geometry_and_labels_as_disabled"]
            )
            lines.extend([
                "## Production-safety check",
                "",
                f"Fallback preserved the disabled-mode accepted Grounding DINO result: **{str(preserved).lower()}**.",
                "",
            ])
    return "\n".join(lines) + "\n"


def compare_modes(image_path: Path, query: str, output_dir: Path) -> list[dict[str, Any]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grounder = RemoteSensingGrounder()
    previous_mode = os.environ.get(MODE_ENVIRONMENT)
    results: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    metadata = build_metadata(image_path, image)
    try:
        # Separate the one-time backbone load from per-mode inference so the
        # reported mode runtimes are comparable. Disabled mode guarantees the
        # specialist is neither loaded nor executed during this initialization.
        os.environ[MODE_ENVIRONMENT] = "disabled"
        load_started = time.perf_counter()
        grounder.load()
        backbone_initialization_ms = round((time.perf_counter() - load_started) * 1000)
        for mode in MODES:
            os.environ[MODE_ENVIRONMENT] = mode
            started = time.perf_counter()
            try:
                result = grounder.ground(
                    image,
                    query,
                    metadata,
                    Modality.OPTICAL,
                    ["r", "g", "b"],
                    "RGB comparison input",
                )
                wall_runtime_ms = round((time.perf_counter() - started) * 1000)
                serialized = result.model_dump(mode="json")
                detections = serialized["detections"]
                diagnostics = grounder.last_score_diagnostics
                fallback_reasons = sorted({item["fallback_reason"] for item in diagnostics if item.get("fallback_reason")})
                row = {
                    "mode": mode,
                    "accepted_count": result.accepted_detection_count,
                    "rejected_count": result.rejected_candidate_count,
                    "scores": [float(item["score"]) for item in detections],
                    "boxes_pixels": [item["bbox_pixels"] for item in detections],
                    "boxes_normalized": [item["bbox_normalized"] for item in detections],
                    "labels": [item["label"] for item in detections],
                    "wall_runtime_ms": wall_runtime_ms,
                    "reported_runtime_ms": result.runtime_ms,
                    "backbone_initialization_ms": backbone_initialization_ms,
                    "model_load_ms": result.model_load_ms,
                    "model_reused": result.model_reused,
                    "fallback_used": any(item.get("fallback_used") for item in diagnostics),
                    "fallback_reason": ", ".join(fallback_reasons) or None,
                    "calibration_methods": sorted({item["calibration_method"] for item in diagnostics if item.get("calibration_method")}),
                    "score_diagnostics": diagnostics,
                    "detections": detections,
                    "response": serialized,
                    "error": None,
                }
                baseline = results[0] if results and not results[0].get("error") else None
                row["same_geometry_and_labels_as_disabled"] = (
                    True if mode == "disabled" else baseline is not None and _signature(detections) == _signature(baseline["detections"])
                )
                results.append(row)
                for diagnostic in diagnostics:
                    diagnostic_rows.append({"mode": mode, **diagnostic})
                save_annotated_preview(image, detections, mode, output_dir / f"preview_{mode}.png")
            except Exception as error:  # comparison must preserve partial evidence
                results.append({
                    "mode": mode,
                    "error": f"{type(error).__name__}: {error}",
                    "wall_runtime_ms": round((time.perf_counter() - started) * 1000),
                })
                save_annotated_preview(image, [], mode, output_dir / f"preview_{mode}.png")
    finally:
        image.close()
        if previous_mode is None:
            os.environ.pop(MODE_ENVIRONMENT, None)
        else:
            os.environ[MODE_ENVIRONMENT] = previous_mode

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_rows = [{
        "mode": row["mode"],
        "accepted_count": row.get("accepted_count"),
        "rejected_count": row.get("rejected_count"),
        "scores": json.dumps(row.get("scores", [])),
        "boxes_pixels": json.dumps(row.get("boxes_pixels", [])),
        "labels": json.dumps(row.get("labels", [])),
        "wall_runtime_ms": row["wall_runtime_ms"],
        "backbone_initialization_ms": row.get("backbone_initialization_ms"),
        "model_load_ms": row.get("model_load_ms"),
        "model_reused": row.get("model_reused"),
        "same_geometry_and_labels_as_disabled": row.get("same_geometry_and_labels_as_disabled"),
        "fallback_used": row.get("fallback_used"),
        "fallback_reason": row.get("fallback_reason"),
        "calibration_methods": json.dumps(row.get("calibration_methods", [])),
        "error": row.get("error"),
    } for row in results]
    write_csv(output_dir / "comparison.csv", summary_rows, list(summary_rows[0]))
    diagnostic_fields = [
        "mode", "proposal_index", "original_grounding_score", "specialist_score", "final_score",
        "score_mode", "fallback_used", "fallback_reason", "calibration_method",
    ]
    write_csv(output_dir / "score_diagnostics.csv", diagnostic_rows, diagnostic_fields)
    (output_dir / "benchmark_report.md").write_text(
        _markdown_report(image_path, query, results), encoding="utf-8",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Optical RGB image to ground.")
    parser.add_argument("--query", required=True, help="Full grounding query.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "grounding_score_mode_comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = compare_modes(args.image.resolve(), args.query, args.output_dir.resolve())
    errors = [result for result in results if result.get("error")]
    disabled = next((result for result in results if result["mode"] == "disabled" and not result.get("error")), None)
    fallback = next((result for result in results if result["mode"] == "fallback" and not result.get("error")), None)
    safety_failure = (
        disabled is None
        or fallback is None
        or fallback["accepted_count"] < disabled["accepted_count"]
        or not fallback["same_geometry_and_labels_as_disabled"]
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "modes_completed": len(results) - len(errors),
        "errors": len(errors),
        "fallback_preserved_dino": not safety_failure,
    }, indent=2))
    return 1 if errors or safety_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
