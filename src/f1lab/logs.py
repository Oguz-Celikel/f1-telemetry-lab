"""Logging setup shared by the CLI and the desktop app.

Two destinations, two audiences. The console is for whoever is watching the
run right now. The rotating file is for the failure nobody was watching:
``docker compose run --rm`` containers are deleted the moment they exit and
take their stdout with them, and a GUI has no console at all — so the file,
bind-mounted out of the container and written on the host by the app, is the
only place an error survives.

The directory comes from ``F1LAB_LOG_DIR`` (the Docker image points it at the
bind-mounted ``/app/logs``) and defaults to ``logs/`` in the working
directory. CLI and GUI write separate files, so a crash in one is not buried
under the chatter of the other.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

DEFAULT_LOG_DIR = Path("logs")
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def log_directory() -> Path:
    """Where log files go; ``F1LAB_LOG_DIR`` overrides the default ``logs/``."""
    return Path(os.environ.get("F1LAB_LOG_DIR", str(DEFAULT_LOG_DIR)))


def setup_logging(filename: str) -> Path:
    """Configure the root logger: console at INFO plus a rotating file.

    Returns the log file's path so the caller can tell the user where to look.
    Replaces the root handlers rather than appending, which makes the call
    idempotent — a second setup (tests, re-entry) never duplicates output.

    The file rotates at 1 MB with three backups: enough history to diagnose
    yesterday's failure, bounded enough never to fill a disk.
    """
    directory = log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [console, file]
    return path


def log_uncaught_exceptions(logger: logging.Logger) -> None:
    """Send anything that escapes to the top of the stack through ``logger``.

    Qt swallows exceptions raised inside slots after printing them to a stderr
    nobody may be looking at; a crashing CLI at least prints, but to a console
    that is gone once the container exits. Hooking ``sys.excepthook`` gets the
    full traceback into the log file first — then defers to the default hook,
    so the console behaviour stays exactly as it was.
    """

    def hook(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, traceback))
        sys.__excepthook__(exc_type, exc, traceback)

    sys.excepthook = hook
