"""Generate deterministic, local-only SatQuery demo fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.io import MemoryFile


DESTINATION = Path(__file__).resolve().parents[1] / "satquery_agent" / "demo_samples"


def optical_array(size: int = 128) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = (167, 150, 112)
    image[12:56, 10:62] = (24, 62, 92)
    image[70:118, 8:62] = (51, 126, 63)
    for start in range(70, 118, 8):
        image[start:start + 3, 8:62] = (77, 151, 76)
    image[64:118, 70:122] = (188, 184, 175)
    for coordinate in range(68, 124, 10):
        image[64:118, coordinate:coordinate + 3] = (75, 78, 84)
        image[coordinate if coordinate < 118 else 114:coordinate + 3 if coordinate < 118 else 117, 70:122] = (75, 78, 84)
    return image


def save_geotiff(name: str, data: np.ndarray) -> None:
    bands = np.moveaxis(data, -1, 0) if data.ndim == 3 else data
    if bands.ndim == 2:
        bands = bands[None, :, :]
    profile = {
        "driver": "GTiff",
        "height": bands.shape[1],
        "width": bands.shape[2],
        "count": bands.shape[0],
        "dtype": str(bands.dtype),
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(70.0, 20.0, 0.01, 0.01),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(bands)
        (DESTINATION / name).write_bytes(memory.read())


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    optical = optical_array()
    Image.fromarray(optical).save(DESTINATION / "single-optical.png")
    before = Image.fromarray(optical)
    after = before.copy()
    ImageDraw.Draw(after).rectangle((82, 18, 116, 52), fill=(228, 218, 191))
    before.save(DESTINATION / "change-before.png")
    after.save(DESTINATION / "change-after.png")
    save_geotiff("cross-optical.tif", optical)
    grayscale = optical.astype(np.float32).mean(axis=2) / 255.0
    sar = np.stack([grayscale, np.clip(grayscale * 0.82 + 0.08, 0, 1)], axis=-1).astype(np.float32)
    save_geotiff("cross-sar.tif", sar)
    print(f"Generated five approved local demo files in {DESTINATION}")


if __name__ == "__main__":
    main()
