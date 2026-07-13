"""Smoke tests for the benchmark helpers, without running the full timings.

The benchmark is what backs the performance claims in the README, so its data
generator and its timer are code like any other — untested, they rot, and the
table starts lying. Actual timings are not asserted: they depend on the machine
and would make the suite slow and flaky.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1lab import bench


def test_synthetic_laps_are_consistent() -> None:
    """The generated telemetry must describe laps that could physically happen.

    Distance and elapsed time strictly increase, speed stays inside the clip
    range, and every lap covers exactly the configured track length. A generator
    that produced, say, a non-monotonic distance array would quietly benchmark
    the interpolators on input they never see in practice.
    """
    distances, times, speeds = bench.synthetic_laps(n_drivers=3, samples=200, seed=7)
    assert len(distances) == len(times) == len(speeds) == 3
    for distance, time_s, speed in zip(distances, times, speeds, strict=True):
        assert distance.shape == time_s.shape == speed.shape == (200,)
        assert np.all(np.diff(distance) > 0)
        assert np.all(np.diff(time_s) > 0)
        assert distance[-1] == pytest.approx(bench.LAP_LENGTH_M)
        assert np.all((speed >= 60.0) & (speed <= 340.0))


def test_synthetic_laps_are_deterministic() -> None:
    # Same seed, same data — otherwise two benchmark runs would be timing
    # different workloads and could not be compared.
    first = bench.synthetic_laps(n_drivers=2, samples=50, seed=1)
    second = bench.synthetic_laps(n_drivers=2, samples=50, seed=1)
    np.testing.assert_array_equal(first[0][0], second[0][0])
    np.testing.assert_array_equal(first[1][1], second[1][1])


def test_best_ms_measures_something_positive() -> None:
    # The timer's contract: it calls the workload exactly number * repeats times
    # and reports a non-negative duration. The call count is the real assertion —
    # a timer that ran the code fewer times than it thought would divide by the
    # wrong number and understate every measurement.
    calls = 0

    def noop() -> None:
        nonlocal calls
        calls += 1

    elapsed_ms = bench.best_ms(noop, number=3, repeats=2)
    assert elapsed_ms >= 0.0
    assert calls == 6
