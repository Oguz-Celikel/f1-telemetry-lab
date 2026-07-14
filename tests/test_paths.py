"""Tests for path resolution — the same code answers differently by context.

The frozen branches are what make the bundled macOS app work at all: launched
from Finder its working directory is ``/``, so the project-style defaults
would try to write where they cannot. These tests simulate the bundle by
setting the attribute PyInstaller sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from f1lab import paths


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    # PyInstaller marks its bundles by setting sys.frozen; nothing else does.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")


def test_defaults_are_project_style(monkeypatch: pytest.MonkeyPatch) -> None:
    # From a checkout or in Docker: everything under the working directory.
    monkeypatch.delenv("F1LAB_LOG_DIR", raising=False)
    monkeypatch.delenv("FASTF1_CACHE", raising=False)
    assert paths.log_directory() == Path("logs")
    assert paths.cache_directory() == Path(".fastf1-cache")


def test_environment_variables_always_win(monkeypatch: pytest.MonkeyPatch, frozen: None) -> None:
    # Docker relies on this: explicit configuration beats every default,
    # frozen or not.
    monkeypatch.setenv("F1LAB_LOG_DIR", "/mnt/logs")
    monkeypatch.setenv("FASTF1_CACHE", "/mnt/cache")
    assert paths.log_directory() == Path("/mnt/logs")
    assert paths.cache_directory() == Path("/mnt/cache")


def test_the_frozen_app_uses_macos_conventions(
    monkeypatch: pytest.MonkeyPatch, frozen: None
) -> None:
    # Launched from Finder the working directory is `/` — not writable — so
    # the bundle writes where macOS users and Console.app expect.
    monkeypatch.delenv("F1LAB_LOG_DIR", raising=False)
    monkeypatch.delenv("FASTF1_CACHE", raising=False)
    home = Path.home()
    assert paths.log_directory() == home / "Library" / "Logs" / "F1 Telemetry Lab"
    assert (
        paths.cache_directory()
        == home / "Library" / "Application Support" / "F1 Telemetry Lab" / "fastf1-cache"
    )
