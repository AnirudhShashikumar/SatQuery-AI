"""Keep the large optional SVE model disabled unless a focused test enables it."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_sve_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "false")
