#!/usr/bin/env python3
"""Compatibility wrapper for the standalone Grounding Specialist trainer."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from training.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
