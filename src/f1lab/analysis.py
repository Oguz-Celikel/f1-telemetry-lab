"""Pure analysis logic for lap telemetry comparisons.

This module deliberately never imports FastF1: everything here operates on
plain numpy/pandas structures (or any object that quacks like a FastF1
session), so the unit tests can run without any network access.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from f1lab import native


class SessionLike(Protocol):
    """The tiny slice of ``fastf1.core.Session`` that this module relies on."""

    laps: Any


@dataclass(frozen=True)
class FastestLap:
    """A driver's fastest lap: lap time in seconds plus its car telemetry."""

    driver: str
    lap_time_s: float
    telemetry: pd.DataFrame


def fastest_lap_telemetry(session: SessionLike, driver: str) -> FastestLap:
    """Extract the fastest lap of ``driver`` from a loaded session.

    The session only needs to expose ``.laps`` with the FastF1 API shape,
    which keeps this function easy to fake in tests.
    """
    lap = session.laps.pick_drivers(driver).pick_fastest()
    if lap is None or pd.isna(lap["LapTime"]):
        raise ValueError(f"No valid fastest lap found for driver {driver!r}")
    telemetry: pd.DataFrame = lap.get_car_data().add_distance()
    if telemetry.empty:
        raise ValueError(f"Empty telemetry for driver {driver!r}")
    lap_time_s = float(lap["LapTime"].total_seconds())
    return FastestLap(driver=driver, lap_time_s=lap_time_s, telemetry=telemetry)


def distance_and_time(telemetry: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return distance [m] and lap-relative time [s] arrays from telemetry."""
    distance = np.asarray(telemetry["Distance"], dtype=np.float64)
    elapsed = telemetry["Time"] - telemetry["Time"].iloc[0]
    time_s = np.asarray(elapsed.dt.total_seconds(), dtype=np.float64)
    return distance, time_s


def compute_delta_time(
    ref_distance_m: NDArray[np.float64],
    ref_time_s: NDArray[np.float64],
    other_distance_m: NDArray[np.float64],
    other_time_s: NDArray[np.float64],
    *,
    engine: str | None = None,
) -> NDArray[np.float64]:
    """Time delta ``other - ref`` resampled onto the reference distance grid.

    Positive values mean the *other* driver is behind the reference driver at
    that point of the lap. ``engine`` forces ``"cpp"`` or ``"numpy"``; by
    default the C++ core is used whenever the compiled extension is present.
    """
    if ref_distance_m.shape != ref_time_s.shape:
        raise ValueError("reference distance and time arrays must have the same shape")
    if other_distance_m.shape != other_time_s.shape:
        raise ValueError("other distance and time arrays must have the same shape")
    if ref_distance_m.size == 0 or other_distance_m.size == 0:
        raise ValueError("telemetry arrays must not be empty")
    if native.resolve_engine(engine) == "cpp":
        return native.delta_time(ref_distance_m, ref_time_s, other_distance_m, other_time_s)
    other_time_on_ref = np.interp(ref_distance_m, other_distance_m, other_time_s)
    return np.asarray(other_time_on_ref - ref_time_s, dtype=np.float64)


def _validate_laps(
    distances: Sequence[NDArray[np.float64]],
    times: Sequence[NDArray[np.float64]],
    grid: NDArray[np.float64],
) -> None:
    """Check the inputs shared by the multi-lap functions.

    Validation lives on the Python side of the dispatch so that both engines
    reject the same inputs with the same messages.
    """
    if len(distances) != len(times):
        raise ValueError("distances and times must contain the same number of laps")
    if grid.size == 0:
        raise ValueError("grid must not be empty")
    for distance, time_s in zip(distances, times, strict=True):
        if distance.shape != time_s.shape:
            raise ValueError("lap distance and time arrays must have the same shape")
        if distance.size == 0:
            raise ValueError("telemetry arrays must not be empty")


def resample_times(
    distances: Sequence[NDArray[np.float64]],
    times: Sequence[NDArray[np.float64]],
    grid: NDArray[np.float64],
    *,
    engine: str | None = None,
) -> NDArray[np.float64]:
    """Resample each lap's cumulative time onto a shared distance grid.

    Returns an ``(n_laps, grid.size)`` matrix; row ``i`` is lap ``i``'s time
    at every grid point.
    """
    _validate_laps(distances, times, grid)
    if native.resolve_engine(engine) == "cpp":
        return native.resample_times(list(distances), list(times), grid)
    out = np.empty((len(distances), grid.size), dtype=np.float64)
    for i, (distance, time_s) in enumerate(zip(distances, times, strict=True)):
        out[i] = np.interp(grid, distance, time_s)
    return out


def pairwise_delta_matrix(
    distances: Sequence[NDArray[np.float64]],
    times: Sequence[NDArray[np.float64]],
    grid: NDArray[np.float64],
    *,
    engine: str | None = None,
) -> NDArray[np.float64]:
    """Every-driver-versus-every-driver delta cube on a shared distance grid.

    ``out[i, j, k]`` is how far lap ``j`` is behind lap ``i`` at grid point
    ``k`` (seconds); the cube is antisymmetric with a zero diagonal.
    """
    _validate_laps(distances, times, grid)
    if native.resolve_engine(engine) == "cpp":
        return native.pairwise_delta_matrix(list(distances), list(times), grid)
    resampled = resample_times(distances, times, grid, engine="numpy")
    return np.asarray(resampled[np.newaxis, :, :] - resampled[:, np.newaxis, :], dtype=np.float64)


def format_lap_time(lap_time_s: float) -> str:
    """Format seconds as ``m:ss.mmm``, e.g. ``87.432 -> '1:27.432'``."""
    if lap_time_s < 0:
        raise ValueError("lap time must be non-negative")
    minutes, seconds = divmod(lap_time_s, 60.0)
    return f"{int(minutes)}:{seconds:06.3f}"


def slugify(text: str) -> str:
    """Lowercase ``text`` and squash anything non-alphanumeric to ``_``."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def build_output_path(
    output_dir: Path,
    year: int,
    event_name: str,
    session_name: str,
    driver_1: str,
    driver_2: str,
) -> Path:
    """Deterministic PNG path, e.g. ``output/2026_british_grand_prix_r_ver_vs_nor.png``."""
    drivers = f"{slugify(driver_1)}_vs_{slugify(driver_2)}"
    stem = f"{year}_{slugify(event_name)}_{slugify(session_name)}_{drivers}"
    return output_dir / f"{stem}.png"
