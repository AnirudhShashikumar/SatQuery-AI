"""Environment-gated local demo sample manifest and safe file lookup."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .models import DemoManifest, DemoSampleFile, DemoWorkflow, InputMode, Modality


DEMO_DIR = (Path(__file__).resolve().parent / "demo_samples").resolve()
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\.(png|tif|tiff|jpg|jpeg)")


def demo_enabled() -> bool:
    return os.getenv("SATQUERY_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _file(role: str, name: str, mime: str) -> DemoSampleFile:
    return DemoSampleFile(
        role=role,
        filename=name,
        url=f"/api/agent/demo/files/{name}",
        mime_type=mime,
    )


def demo_manifest() -> DemoManifest:
    if not demo_enabled():
        return DemoManifest(enabled=False, workflows=[])
    workflows = [
        DemoWorkflow(
            id="single_vqa",
            title="Single-image understanding",
            description="Approved local optical sample with visible water-, vegetation-, and structure-support patterns.",
            input_mode=InputMode.SINGLE,
            primary_modality=Modality.OPTICAL,
            query="Is a water body visible?",
            files=[_file("primary", "single-optical.png", "image/png")],
        ),
        DemoWorkflow(
            id="change_vqa",
            title="Bi-temporal change analysis",
            description="Approved local before/after pair containing one measurable changed region.",
            input_mode=InputMode.BI_TEMPORAL,
            primary_modality=Modality.OPTICAL,
            secondary_modality=Modality.OPTICAL,
            query="How much of the image changed?",
            primary_date="2025-01-01",
            secondary_date="2025-02-01",
            files=[
                _file("primary", "change-before.png", "image/png"),
                _file("secondary", "change-after.png", "image/png"),
            ],
        ),
        DemoWorkflow(
            id="cross_modal",
            title="Optical–SAR joint analysis",
            description="Approved local exactly aligned GeoTIFF pair with real deterministic fusion evidence.",
            input_mode=InputMode.CROSS_MODAL,
            primary_modality=Modality.OPTICAL,
            secondary_modality=Modality.SAR,
            query="Where do both modalities agree?",
            files=[
                _file("primary", "cross-optical.tif", "image/tiff"),
                _file("secondary", "cross-sar.tif", "image/tiff"),
            ],
        ),
    ]
    return DemoManifest(enabled=True, workflows=workflows)


def demo_sample_path(name: str) -> Optional[Path]:
    if not demo_enabled() or not _SAFE_NAME.fullmatch(name):
        return None
    path = (DEMO_DIR / name).resolve()
    if path.parent != DEMO_DIR or not path.is_file():
        return None
    return path
