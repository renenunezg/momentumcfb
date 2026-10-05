"""The production write gate requires an explicit opt-in in every runtime."""

from __future__ import annotations

from backend.db import writes_allowed


def test_writes_allowed_logic(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MOMENTUMCFB_DB_WRITES", raising=False)
    assert writes_allowed() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert writes_allowed() is False
    monkeypatch.setenv("MOMENTUMCFB_DB_WRITES", "1")
    assert writes_allowed() is True
