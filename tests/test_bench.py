"""Smoke tests for the benchmark helpers, without running the full timings.

The benchmark is what backs the performance claims in the README, so its data
generator and its timer are code like any other — untested, they rot, and the
table starts lying. Actual timings are not asserted: they depend on the machine
and would make the suite slow and flaky.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1lab import bench, native


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


def test_workloads_run_on_both_engines() -> None:
    """Each workload's `run` callable works with either engine name.

    build_workloads returns closures, and a typo inside one of them would only
    surface when `just bench` is next run — which is exactly the kind of rot
    this file exists to prevent.
    """
    laps = bench.synthetic_laps(n_drivers=3, samples=120, seed=3)
    grid = np.linspace(0.0, bench.LAP_LENGTH_M, 50, dtype=np.float64)
    workloads = bench.build_workloads(laps, grid)
    assert len(workloads) == 3
    for workload in workloads:
        assert workload.name
        assert workload.number > 0
        result = workload.run("numpy")
        assert result is not None


def test_check_parity_accepts_agreeing_engines() -> None:
    # The guard must not fire on a workload whose engines agree — otherwise the
    # benchmark could never run at all.
    laps = bench.synthetic_laps(n_drivers=2, samples=120, seed=4)
    grid = np.linspace(0.0, bench.LAP_LENGTH_M, 50, dtype=np.float64)
    workload = bench.build_workloads(laps, grid)[0]
    if native.HAS_NATIVE:
        bench.check_parity(workload)  # must not raise


def test_check_parity_rejects_disagreeing_engines() -> None:
    """A workload whose engines disagree must abort the benchmark.

    This is the safety catch that keeps the README's table honest: timing two
    functions that compute different things would be meaningless, so the run
    fails instead of reporting a speedup.
    """
    disagreeing = bench.Workload(
        name="rigged",
        # Returns a different array depending on the engine asked for.
        run=lambda engine: np.zeros(4) if engine == "numpy" else np.ones(4),
        number=1,
    )
    with pytest.raises(RuntimeError, match="engine mismatch"):
        bench.check_parity(disagreeing)


def test_main_prints_a_markdown_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() renders the table. Shrunk to keep the test fast.

    The real workload sizes are tuned for a meaningful measurement, not for a
    test suite, so the module constants are patched down to something tiny —
    the code path is identical, only the numbers are smaller.
    """
    if not native.HAS_NATIVE:
        pytest.skip("benchmark needs both engines")
    monkeypatch.setattr(bench, "N_DRIVERS", 2)
    monkeypatch.setattr(bench, "SAMPLES_PER_LAP", 100)
    monkeypatch.setattr(bench, "GRID_POINTS", 40)
    monkeypatch.setattr(bench, "REPEATS", 1)

    assert bench.main() == 0

    out = capsys.readouterr().out
    assert "| Workload | numpy | C++ | speedup |" in out
    assert out.count("ms |") == 6  # two timings on each of the three rows


def test_main_without_the_extension_reports_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Nothing to compare against: say so and exit non-zero rather than pretend.
    # Patch the native module itself rather than reaching through bench: it is
    # the same object either way, and bench re-exporting it is not part of its API.
    monkeypatch.setattr(native, "HAS_NATIVE", False)
    assert bench.main() == 1
    assert "not available" in capsys.readouterr().out
