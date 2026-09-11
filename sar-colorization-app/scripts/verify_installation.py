#!/usr/bin/env python3
"""Read-only clean-install verification for SatQuery's local core."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path

try:
    from verify_models import verify as verify_models
except ModuleNotFoundError:
    from scripts.verify_models import verify as verify_models


REQUIRED_IMPORTS = ("fastapi", "numpy", "PIL", "pydantic", "rasterio", "skimage", "torch", "torchvision", "transformers", "tifffile", "uvicorn")


def verify(root: Path) -> dict:
    dependencies = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            dependencies.append({"name": name, "available": True, "version": getattr(module, "__version__", None)})
        except Exception as error:
            dependencies.append({"name": name, "available": False, "error": f"{type(error).__name__}: {error}"[:300]})
    files = {relative: (root / relative).is_file() for relative in ("backend.py", "requirements-backend.txt", "frontend/package.json", ".env.example", "benchmarks/schema/benchmark-result.schema.json")}
    models = verify_models(root)
    ready = all(item["available"] for item in dependencies) and all(files.values()) and models["required_ok"]
    return {"status": "ready" if ready else "incomplete", "python": platform.python_version(), "machine": platform.machine(), "dependencies": dependencies, "files": files, "models": models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify(args.root.expanduser().resolve())
    print(json.dumps(result, indent=2) if args.json else f"SatQuery installation: {result['status']}")
    raise SystemExit(0 if result["status"] == "ready" else 1)
