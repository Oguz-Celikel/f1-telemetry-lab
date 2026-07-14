"""Unit tests for the pure analysis logic — no FastF1 import, no network.

``f1lab.analysis`` never imports FastF1, and that design decision pays off
here: the session object it reads can be replaced by a handful of small fakes.
They implement only the four calls the module actually makes
(``pick_drivers``, ``pick_fastest``, ``get_car_data``, ``add_distance``), which
is why they are hand-written rather than mocked — a fake that only supports
what the code legitimately uses will fail loudly if the code starts reaching
for something new.

The maths tests use closed-form expected values (a car at constant speed covers
d metres in d/v seconds), so they check the algorithm rather than record its
current output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1lab.analysis import (
    build_output_path,
    compute_delta_time,
    distance_and_time,
    fastest_lap_telemetry,
    format_lap_time,
    pairwise_delta_matrix,
    resample_times,
    slugify,
)


def make_telemetry(
    distance_m: list[float], time_s: list[float], speed_kmh: list[float]
) -> pd.DataFrame:
    """A DataFrame with the columns FastF1's car data actually carries."""
    # Time must be a Timedelta, as in the real thing: distance_and_time calls
    # .dt.total_seconds() on it.
    return pd.DataFrame(
        {
            "Distance": distance_m,
            "Time": pd.to_timedelta(time_s, unit="s"),
            "Speed": speed_kmh,
        }
    )


class FakeCarData:
    """Stands in for the object returned by ``Lap.get_car_data()``."""

    def __init__(self, telemetry: pd.DataFrame) -> None:
        self._telemetry = telemetry

    def add_distance(self) -> pd.DataFrame:
        return self._telemetry


class FakeLap:
    """Stands in for ``fastf1.core.Lap``."""

    def __init__(self, lap_time: pd.Timedelta, telemetry: pd.DataFrame) -> None:
        self._lap_time = lap_time
        self._telemetry = telemetry

    def __getitem__(self, key: str) -> pd.Timedelta:
        # Raising on any other key keeps the fake honest: if the code under test
        # starts reading a column this fake does not model, the test fails
        # rather than silently passing on invented data.
        if key != "LapTime":
            raise KeyError(key)
        return self._lap_time

    def get_car_data(self) -> FakeCarData:
        return FakeCarData(self._telemetry)


class FakeLaps:
    """Stands in for ``fastf1.core.Laps``."""

    def __init__(self, fastest: FakeLap | None) -> None:
        self._fastest = fastest
        # Records which drivers were asked for, so a test can assert the right
        # one was actually selected.
        self.picked: list[str] = []

    def pick_drivers(self, driver: str) -> FakeLaps:
        self.picked.append(driver)
        return self  # chainable, like the real API

    def pick_fastest(self) -> FakeLap | None:
        return self._fastest


class FakeSession:
    """Structural stand-in for ``fastf1.core.Session``.

    Satisfies the ``SessionLike`` protocol by having a ``.laps`` attribute —
    no inheritance, no registration.
    """

    def __init__(self, fastest: FakeLap | None) -> None:
        self.laps = FakeLaps(fastest)


class TestComputeDeltaTime:
    """The core calculation. Engine is left unset, so the default one is used."""

    def test_identical_laps_give_zero_delta(self) -> None:
        # A lap against itself: nowhere ahead, nowhere behind. Catches sign
        # flips and off-by-one errors in the resampling.
        distance = np.linspace(0.0, 1000.0, 51, dtype=np.float64)
        time = distance / 50.0
        delta = compute_delta_time(distance, time, distance, time)
        assert delta.shape == distance.shape
        assert np.allclose(delta, 0.0)

    def test_constant_speed_gap_grows_linearly(self) -> None:
        # 40 m/s against 50 m/s: the slower car is d/40 - d/50 = d/200 seconds
        # behind at distance d, so exactly 5 s down after a kilometre. The
        # expected curve comes from the physics, not from a previous run.
        distance = np.linspace(0.0, 1000.0, 101, dtype=np.float64)
        ref_time = distance / 50.0
        other_time = distance / 40.0
        delta = compute_delta_time(distance, ref_time, distance, other_time)
        assert np.allclose(delta, distance / 200.0)
        assert delta[-1] == pytest.approx(5.0)

    def test_other_lap_is_resampled_onto_reference_grid(self) -> None:
        # The output always follows the *reference* lap's sampling, even when
        # the other lap was recorded at a different rate. This is what makes the
        # delta plottable against the reference's distance axis.
        ref_distance = np.linspace(0.0, 100.0, 11, dtype=np.float64)
        other_distance = np.linspace(0.0, 100.0, 7, dtype=np.float64)  # coarser
        delta = compute_delta_time(
            ref_distance, ref_distance / 10.0, other_distance, other_distance / 8.0
        )
        assert delta.shape == ref_distance.shape

    def test_mismatched_shapes_raise(self) -> None:
        # `match` pins the message, not just the type: the C++ engine raises the
        # same text, and test_native_parity.py relies on that.
        with pytest.raises(ValueError, match="reference distance and time"):
            compute_delta_time(np.zeros(5), np.zeros(4), np.zeros(5), np.zeros(5))
        # Both laps are validated, and the message says which one is at fault —
        # so the second pair gets its own case rather than being assumed.
        with pytest.raises(ValueError, match="other distance and time"):
            compute_delta_time(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(4))

    def test_empty_arrays_raise(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            compute_delta_time(empty, empty, empty, empty)


class TestResampleAndPairwise:
    """Numpy-engine tests; the C++ twin is covered by the parity suite.

    Pinned with ``engine="numpy"`` so the reference implementation is still
    exercised on machines where the extension is installed and would otherwise
    take over.
    """

    def test_resample_times_puts_each_lap_on_the_grid(self) -> None:
        # Two laps sampled at different rates (21 and 6 points) land on one
        # shared 11-point grid. Both are linear, so resampling is exact and the
        # expected rows are closed-form.
        grid = np.linspace(0.0, 100.0, 11, dtype=np.float64)
        d1 = np.linspace(0.0, 100.0, 21, dtype=np.float64)
        d2 = np.linspace(0.0, 100.0, 6, dtype=np.float64)
        matrix = resample_times([d1, d2], [d1 / 10.0, d2 / 8.0], grid, engine="numpy")
        assert matrix.shape == (2, 11)
        assert np.allclose(matrix[0], grid / 10.0)
        assert np.allclose(matrix[1], grid / 8.0)

    def test_pairwise_matrix_is_antisymmetric_with_zero_diagonal(self) -> None:
        # Structural properties rather than spot values: delta(i, j) must equal
        # -delta(j, i) everywhere and the diagonal must be zero. A transposed
        # index breaks this in every cell, whereas a few sampled values might
        # not notice.
        grid = np.linspace(0.0, 500.0, 26, dtype=np.float64)
        distance = np.linspace(0.0, 500.0, 51, dtype=np.float64)
        distances = [distance] * 3
        times = [distance / 50.0, distance / 45.0, distance / 55.0]
        cube = pairwise_delta_matrix(distances, times, grid, engine="numpy")
        assert cube.shape == (3, 3, 26)
        assert np.allclose(cube + cube.transpose(1, 0, 2), 0.0)
        assert np.allclose(cube[np.arange(3), np.arange(3)], 0.0)
        # And one anchored value, to fix the sign convention: driver 1 (45 m/s)
        # ends the lap behind driver 0 (50 m/s), so the delta is positive.
        assert cube[0, 1, -1] == pytest.approx(500.0 / 45.0 - 500.0 / 50.0)

    def test_lap_count_mismatch_raises(self) -> None:
        grid = np.linspace(0.0, 10.0, 5, dtype=np.float64)
        with pytest.raises(ValueError, match="same number of laps"):
            resample_times([grid, grid], [grid], grid, engine="numpy")

    def test_empty_grid_raises(self) -> None:
        lap = np.linspace(0.0, 10.0, 5, dtype=np.float64)
        with pytest.raises(ValueError, match="grid must not be empty"):
            pairwise_delta_matrix([lap], [lap], np.array([], dtype=np.float64))

    def test_ragged_lap_raises(self) -> None:
        grid = np.linspace(0.0, 10.0, 5, dtype=np.float64)
        with pytest.raises(ValueError, match="same shape"):
            resample_times([grid], [grid[:-1]], grid, engine="numpy")

    def test_empty_lap_raises(self) -> None:
        # An empty lap among otherwise valid ones: caught per-lap, not just by
        # the length check on the list.
        grid = np.linspace(0.0, 10.0, 5, dtype=np.float64)
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            resample_times([grid, empty], [grid, empty], grid, engine="numpy")


class TestFastestLapTelemetry:
    """The FastF1 adapter, driven entirely through the fakes above."""

    def test_returns_lap_time_and_telemetry(self) -> None:
        telemetry = make_telemetry([0.0, 100.0], [0.0, 2.0], [280.0, 300.0])
        session = FakeSession(FakeLap(pd.Timedelta(87.432, unit="s"), telemetry))
        result = fastest_lap_telemetry(session, "VER")
        assert result.driver == "VER"
        assert result.lap_time_s == pytest.approx(87.432)
        # `is`, not `==`: the telemetry is carried through, not copied.
        assert result.telemetry is telemetry
        # And the driver we asked for is the driver that was selected.
        assert session.laps.picked == ["VER"]

    def test_missing_driver_raises(self) -> None:
        # No lap at all — e.g. a driver who never set a time. The message names
        # the driver, so the CLI can print something the user can act on.
        session = FakeSession(None)
        with pytest.raises(ValueError, match="NOR"):
            fastest_lap_telemetry(session, "NOR")

    def test_lap_without_time_raises(self) -> None:
        # A lap exists but its time is NaT (deleted, or the session ended
        # mid-lap). Distinct from the case above and easy to overlook.
        telemetry = make_telemetry([0.0], [0.0], [100.0])
        session = FakeSession(FakeLap(pd.NaT, telemetry))
        with pytest.raises(ValueError, match="HAM"):
            fastest_lap_telemetry(session, "HAM")

    def test_empty_telemetry_raises(self) -> None:
        # A timed lap whose car data never arrived: fail here rather than let an
        # empty array reach the interpolator.
        session = FakeSession(FakeLap(pd.Timedelta(90, unit="s"), pd.DataFrame()))
        with pytest.raises(ValueError, match="PIA"):
            fastest_lap_telemetry(session, "PIA")


def test_distance_and_time_normalises_to_lap_start() -> None:
    # FastF1's Time column is session-relative, so a lap starting five seconds
    # in must be shifted to zero — otherwise two laps could not be compared.
    telemetry = make_telemetry([0.0, 10.0, 20.0], [5.0, 6.0, 7.0], [200.0, 210.0, 220.0])
    distance, time_s = distance_and_time(telemetry)
    assert np.allclose(distance, [0.0, 10.0, 20.0])
    assert np.allclose(time_s, [0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    ("seconds", "expected"),
    # Three shapes that catch the usual formatting slips: the ordinary case,
    # a sub-minute lap (the seconds field must stay zero-padded), and one that
    # lands exactly on a minute boundary.
    [(87.432, "1:27.432"), (59.5, "0:59.500"), (125.0, "2:05.000")],
)
def test_format_lap_time(seconds: float, expected: str) -> None:
    assert format_lap_time(seconds) == expected


def test_format_lap_time_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_lap_time(-1.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Great Britain", "great_britain"),
        ("Emilia-Romagna", "emilia_romagna"),  # punctuation collapses too
        ("R", "r"),
        ("***", "unknown"),  # everything stripped: never return an empty name
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_build_output_path() -> None:
    # The naming contract, spelled out: same inputs always give the same file,
    # so re-running an analysis overwrites its plot instead of littering the
    # output directory with near-duplicates.
    path = build_output_path(Path("output"), 2026, "British Grand Prix", "R", "VER", "NOR")
    assert path == Path("output/2026_british_grand_prix_r_ver_vs_nor.png")
