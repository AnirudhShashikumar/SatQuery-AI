"""Pinned configuration for the standalone ChangerEx extraction."""

from __future__ import annotations

from dataclasses import dataclass


MODEL_NAME = "ChangerEx IA-ResNetV1c-18 (LEVIR-CD)"
MODEL_CONFIG = "configs/changer/changer_ex_r18_512x512_40k_levircd.py"
OPENCD_VERSION = "v1.1.0"
OPENCD_COMMIT = "09c03eb1077f06191c5448ea1fcf2f2f88ec98c2"
CHECKPOINT_FILENAME = "ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth"
CHECKPOINT_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"
CHECKPOINT_URL = "https://drive.google.com/file/d/1SZ8DBgBryJ63X9AU0AQvaJ2ZH3oQHQHe/view"

RGB_MEAN = (123.675, 116.28, 103.53)
RGB_STD = (58.395, 57.12, 57.375)
SIZE_DIVISOR = 32
DEFAULT_MAXIMUM_DIMENSION = 1024
DEFAULT_THRESHOLD = 0.5
CLASS_NAMES = ("unchanged", "changed")
PARAMETER_COUNT = 11_390_946


@dataclass(frozen=True)
class ModelProvenance:
    model_name: str = MODEL_NAME
    config: str = MODEL_CONFIG
    opencd_version: str = OPENCD_VERSION
    opencd_commit: str = OPENCD_COMMIT
    checkpoint_filename: str = CHECKPOINT_FILENAME
    checkpoint_sha256: str = CHECKPOINT_SHA256
    checkpoint_url: str = CHECKPOINT_URL
    training_dataset: str = "LEVIR-CD"
    classes: tuple[str, str] = CLASS_NAMES


PROVENANCE = ModelProvenance()
