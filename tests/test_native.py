"""Tests for engine selection — these run with or without the extension.

Deliberately kept out of test_native_parity.py: that module skips entirely when
the extension is missing, but the fallback logic *itself* has to be tested in
both worlds. These tests therefore assert what must hold in each case rather
than assuming a build.
"""

from __future__ import annotations

import pytest

from f1lab import native


def test_auto_matches_backend_name() -> None:
    # None and "auto" are the same request: whatever the default backend is.
    assert native.resolve_engine(None) == native.backend_name()
    assert native.resolve_engine("auto") == native.backend_name()
    assert native.backend_name() in {"cpp", "numpy"}


def test_numpy_is_always_available() -> None:
    # The reference engine is unconditional — this is the promise that lets the
    # package install on a machine with no C++ toolchain.
    assert native.resolve_engine("numpy") == "numpy"


def test_cpp_resolves_or_raises_depending_on_the_build() -> None:
    """Asking for C++ without the extension is an error, not a silent fallback.

    Falling back quietly would be friendlier and wrong: the benchmark and the
    parity suite both request "cpp" explicitly, and if that request could be
    served by numpy they would be comparing numpy with itself while reporting
    otherwise.
    """
    if native.HAS_NATIVE:
        assert native.resolve_engine("cpp") == "cpp"
    else:
        with pytest.raises(RuntimeError, match="not available"):
            native.resolve_engine("cpp")


def test_unknown_engine_raises() -> None:
    # A typo should fail loudly at the call site, not quietly pick a default.
    with pytest.raises(ValueError, match="unknown engine"):
        native.resolve_engine("fortran")
