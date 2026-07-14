"""Tests for the logging setup shared by the CLI and the desktop app.

Each test that calls ``setup_logging`` restores the root logger afterwards
(see the fixture): the setup deliberately replaces the root handlers, and
leaving that in place would break pytest's own log capture for every test
that runs later.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from f1lab.logs import log_uncaught_exceptions, setup_logging
from f1lab.paths import log_directory


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    hook = sys.excepthook
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    sys.excepthook = hook


@pytest.fixture
def log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / "logs"
    monkeypatch.setenv("F1LAB_LOG_DIR", str(directory))
    return directory


def test_the_env_var_decides_the_directory(log_dir: Path) -> None:
    # The Docker image points F1LAB_LOG_DIR at a bind mount; this is the seam
    # that makes container logs land on the host.
    assert log_directory() == log_dir


def test_setup_creates_the_file_and_returns_its_path(log_dir: Path) -> None:
    path = setup_logging("test.log")
    assert path == log_dir / "test.log"
    logging.getLogger("f1lab.test").info("hello from the test")
    assert "hello from the test" in path.read_text()


def test_setup_is_idempotent(log_dir: Path) -> None:
    # Console + file, exactly once — however many times setup runs. Duplicated
    # handlers are the classic logging bug: every line printed twice.
    setup_logging("test.log")
    setup_logging("test.log")
    assert len(logging.getLogger().handlers) == 2


def test_errors_reach_the_file_with_their_traceback(log_dir: Path) -> None:
    # The file is only worth having if it holds enough to diagnose a failure —
    # the stack, not just the message.
    path = setup_logging("test.log")
    try:
        raise ValueError("the engine exploded")
    except ValueError:
        logging.getLogger("f1lab.test").exception("computation failed")
    content = path.read_text()
    assert "computation failed" in content
    assert "Traceback" in content
    assert "the engine exploded" in content


def test_uncaught_exceptions_are_logged_before_the_default_hook(
    log_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook writes to the log first, then behaves like the hook it replaced.

    Both halves matter: without the first, a GUI crash vanishes; without the
    second, the console user loses the traceback they used to get.
    """
    path = setup_logging("test.log")
    log_uncaught_exceptions(logging.getLogger("f1lab.test"))

    error = RuntimeError("nobody caught this")
    sys.excepthook(RuntimeError, error, None)

    assert "nobody caught this" in path.read_text()
    assert "nobody caught this" in capsys.readouterr().err  # default hook still ran
