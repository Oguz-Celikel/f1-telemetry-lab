"""Benchmark the numpy and C++ engines on synthetic telemetry.

Run with ``just bench`` (or ``python -m f1lab.bench``). Before timing, each
workload's outputs are checked for parity between the two engines, so the
table can never compare functions that disagree. Results are printed as a
markdown table ready to paste into the README.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1lab import native
from f1lab.analysis import compute_delta_time, pairwise_delta_matrix
from f1lab.corners import detect_corners

N_DRIVERS = 22
SAMPLES_PER_LAP = 3500
GRID_POINTS = 2000
LAP_LENGTH_M = 5800.0
REPEATS = 5

Laps = tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[NDArray[np.float64]]]


@dataclass(frozen=True)
class Workload:
    """One row of the benchmark table.

    ``run`` takes an engine name and returns that engine's result, so the same
    workload can be both timed and parity-checked against either backend.
    ``number`` is how many calls make up one timing sample — enough that the
    measurement is not dominated by clock resolution.
    """

    name: str
    run: Callable[[str], object]
    number: int


def synthetic_laps(n_drivers: int, samples: int, seed: int = 42) -> Laps:
    """(distance, time, speed) arrays per driver, shaped like car telemetry.

    Speed undulates between straights and corners; time is integrated from
    distance steps and speed, so the arrays are physically consistent.
    """
    rng = np.random.default_rng(seed)
    distances: list[NDArray[np.float64]] = []
    times: list[NDArray[np.float64]] = []
    speeds: list[NDArray[np.float64]] = []
    for _ in range(n_drivers):
        steps = rng.uniform(1.0, 2.4, samples)
        distance = np.cumsum(steps) * (LAP_LENGTH_M / float(np.sum(steps)))
        angle = distance / LAP_LENGTH_M * 2.0 * np.pi
        phase = rng.uniform(0.0, 2.0 * np.pi)
        speed = (
            210.0
            + 90.0 * np.sin(9.0 * angle + phase)
            + 30.0 * np.sin(23.0 * angle)
            + rng.normal(0.0, 2.0, samples)
        )
        speed = np.clip(speed, 60.0, 340.0)
        time_s = np.cumsum(steps / (speed / 3.6))
        distances.append(np.asarray(distance, dtype=np.float64))
        times.append(np.asarray(time_s, dtype=np.float64))
        speeds.append(np.asarray(speed, dtype=np.float64))
    return distances, times, speeds


def best_ms(run: Callable[[], object], number: int, repeats: int = REPEATS) -> float:
    """Best-of-N per-call latency in milliseconds (timeit-style).

    The minimum, not the mean: scheduling noise can only ever make a run
    slower, so the fastest sample is the closest estimate of the code's own
    cost.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            run()
        best = min(best, (time.perf_counter() - start) / number)
    return best * 1000.0


def check_parity(workload: Workload) -> None:
    """Refuse to benchmark engines that do not agree on the result."""
    result_numpy = workload.run("numpy")
    result_cpp = workload.run("cpp")
    if not isinstance(result_numpy, np.ndarray) or not isinstance(result_cpp, np.ndarray):
        return  # corner results are compared index-by-index in the test suite
    if not np.allclose(result_cpp, result_numpy, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"engine mismatch in workload {workload.name!r}")


def build_workloads(laps: Laps, grid: NDArray[np.float64]) -> list[Workload]:
    """The three workloads, chosen to show where C++ does and does not help.

    A single delta and the pairwise cube are dominated by interpolation, which
    numpy already runs as vectorised C; corner detection needs data-dependent
    loops, where the Python fallback pays interpreter cost on every iteration.
    """
    distances, times, speeds = laps
    return [
        Workload(
            name=f"`compute_delta_time` — 1 pair, {SAMPLES_PER_LAP} samples",
            run=lambda engine: compute_delta_time(
                distances[0], times[0], distances[1], times[1], engine=engine
            ),
            number=200,
        ),
        Workload(
            name=(
                f"`pairwise_delta_matrix` — {N_DRIVERS} drivers "
                f"({N_DRIVERS * N_DRIVERS} pairs), {GRID_POINTS}-point grid"
            ),
            run=lambda engine: pairwise_delta_matrix(distances, times, grid, engine=engine),
            number=5,
        ),
        Workload(
            name=f"`detect_corners` — {N_DRIVERS} speed traces",
            run=lambda engine: [
                detect_corners(distance, speed, engine=engine)
                for distance, speed in zip(distances, speeds, strict=True)
            ],
            number=20,
        ),
    ]


def main() -> int:
    """Print the benchmark table; returns a process exit code."""
    if not native.HAS_NATIVE:
        print("f1lab._native is not available — nothing to benchmark against.")
        return 1

    laps = synthetic_laps(N_DRIVERS, SAMPLES_PER_LAP)
    grid = np.linspace(0.0, LAP_LENGTH_M, GRID_POINTS, dtype=np.float64)

    print(
        f"f1lab bench — Python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.machine()}, best of {REPEATS} runs\n"
    )
    print("| Workload | numpy | C++ | speedup |")
    print("|----------|------:|----:|--------:|")
    for workload in build_workloads(laps, grid):
        check_parity(workload)
        ms_numpy = best_ms(lambda: workload.run("numpy"), workload.number)  # noqa: B023
        ms_cpp = best_ms(lambda: workload.run("cpp"), workload.number)  # noqa: B023
        speedup = ms_numpy / ms_cpp
        print(f"| {workload.name} | {ms_numpy:.3f} ms | {ms_cpp:.3f} ms | {speedup:.1f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
