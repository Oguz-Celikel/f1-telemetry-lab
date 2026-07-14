"""Where f1lab keeps its files — and why the answer depends on how it runs.

Run from a checkout (or in Docker), everything lives in the working directory:
``logs/`` and ``.fastf1-cache/``, project-style, easy to find and to delete.
The Docker compose file overrides both through environment variables so they
land on bind mounts.

Bundled into the macOS app, none of that works. An app launched from Finder
gets ``/`` as its working directory, which is not writable — so the frozen
build follows the platform conventions instead: telemetry cache under
``~/Library/Application Support``, logs under ``~/Library/Logs``, where
Console.app and every macOS user expects them.

Resolution order everywhere: explicit environment variable, then the frozen
platform location, then the working-directory default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "F1 Telemetry Lab"

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_CACHE_DIR = Path(".fastf1-cache")


def running_frozen() -> bool:
    """True inside a PyInstaller bundle — the attribute is its calling card."""
    return bool(getattr(sys, "frozen", False))


def log_directory() -> Path:
    """Where log files go. ``F1LAB_LOG_DIR`` wins; the app uses ~/Library/Logs."""
    env = os.environ.get("F1LAB_LOG_DIR")
    if env:
        return Path(env)
    if running_frozen() and sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return DEFAULT_LOG_DIR


def cache_directory() -> Path:
    """Where FastF1's downloads go. ``FASTF1_CACHE`` wins; the app uses
    ~/Library/Application Support."""
    env = os.environ.get("FASTF1_CACHE")
    if env:
        return Path(env)
    if running_frozen() and sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME / "fastf1-cache"
    return DEFAULT_CACHE_DIR
