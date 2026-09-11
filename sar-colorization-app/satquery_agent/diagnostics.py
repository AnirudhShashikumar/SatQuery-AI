"""Opt-in, path-free developer diagnostics for specialist routing."""

from __future__ import annotations

import json
import logging
import os
from typing import Any


LOGGER = logging.getLogger("satquery.diagnostics")


def diagnostic(event: str, **values: Any) -> None:
    if os.getenv("SATQUERY_DEVELOPMENT_DIAGNOSTICS", "").strip().lower() not in {"1", "true", "yes"}:
        return
    safe = {key: value for key, value in values.items() if key not in {"path", "filename", "environment", "model_cache"}}
    LOGGER.info("[%s] %s", event, json.dumps(safe, sort_keys=True, default=str))
