"""Loader for the optional compiled extension ``f1lab._native``.

The package never *requires* the C++ core: every native function has a numpy
twin, and callers dispatch through :func:`resolve_engine`. When the extension
is missing (e.g. installed on a machine without a C++ toolchain) everything
transparently falls back to numpy.
"""

from __future__ import annotations

try:
    from f1lab._native import (
        delta_time as delta_time,
    )
    from f1lab._native import (
        detect_corners as detect_corners,
    )
    from f1lab._native import (
        pairwise_delta_matrix as pairwise_delta_matrix,
    )
    from f1lab._native import (
        resample_times as resample_times,
    )
except ImportError:  # pragma: no cover — depends on how f1lab was installed
    HAS_NATIVE = False
else:
    HAS_NATIVE = True


def backend_name() -> str:
    """Engine used when the caller does not force one."""
    return "cpp" if HAS_NATIVE else "numpy"


def resolve_engine(engine: str | None) -> str:
    """Normalise an engine request to ``"cpp"`` or ``"numpy"``.

    ``None`` (or ``"auto"``) picks the native engine when available. Asking
    for ``"cpp"`` without the compiled extension is an error rather than a
    silent fallback, so benchmarks and parity tests cannot lie.
    """
    if engine is None or engine == "auto":
        return backend_name()
    if engine == "cpp":
        if not HAS_NATIVE:
            raise RuntimeError(
                "engine='cpp' requested but the f1lab._native extension is not available"
            )
        return "cpp"
    if engine == "numpy":
        return "numpy"
    raise ValueError(f"unknown engine {engine!r}; expected 'cpp', 'numpy' or 'auto'")
