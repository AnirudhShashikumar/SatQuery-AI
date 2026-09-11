"""Keep external heavyweight services isolated unless a focused test enables them."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_sve_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "false")
    monkeypatch.setenv("TTP_ENABLED", "false")
    monkeypatch.setenv("SATQUERY_CHANGEREX_ENABLED", "false")
