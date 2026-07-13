"""Unit tests for corner detection, pinned to the numpy reference engine.

Every call passes ``engine="numpy"`` explicitly. Without it these tests would
exercise whichever backend happens to be installed — and on a machine with the
extension built, the numpy implementation would never be tested at all, even
though it is the code that runs for users without a compiler.

The C++ twin is covered by its own Catch2 suite (``cpp/tests/test_corners.cpp``,
the same scenarios), and test_native_parity.py then asserts the two engines
return identical indices.

Test data is synthetic: piecewise-linear speed profiles with corners placed by
hand, so the expected apexes are known rather than recorded from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from f1lab.corners import Corners, detect_corners, smooth

STEP_M = 5.0


def make_trace(
    breakpoints: list[tuple[float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a lap from (distance, speed) breakpoints joined by straight lines."""
    xs = np.array([point[0] for point in breakpoints], dtype=np.float64)
    ys = np.array([point[1] for point in breakpoints], dtype=np.float64)
    # Half a step of slack so the final breakpoint is included despite float
    # rounding in arange.
    distance = np.arange(0.0, xs[-1] + STEP_M / 2.0, STEP_M, dtype=np.float64)
    speed = np.interp(distance, xs, ys)
    return distance, np.asarray(speed, dtype=np.float64)


class TestSmooth:
    """The moving average, tested on its own before anything relies on it."""

    def test_window_one_is_identity(self) -> None:
        # A window of one averages each sample with nothing else.
        values = np.array([1.0, 5.0, 2.0, 8.0])
        assert np.array_equal(smooth(values, 1), values)

    def test_interior_is_plain_average(self) -> None:
        # Away from the edges, smoothing is the mean of the three samples.
        values = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        smoothed = smooth(values, 3)
        assert smoothed[2] == pytest.approx(6.0)

    def test_window_shrinks_at_the_edges(self) -> None:
        # The first sample averages only itself and its right neighbour: 5, not
        # 10/3. Zero-padding would produce the latter and invent a speed drop at
        # the start of every lap.
        values = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        smoothed = smooth(values, 3)
        assert smoothed[0] == pytest.approx(5.0)

    @pytest.mark.parametrize("window", [0, -1, 2, 4])
    def test_invalid_window_raises(self, window: int) -> None:
        # Zero, negative and even windows all have no valid centre to average
        # around and must be rejected rather than quietly adjusted.
        with pytest.raises(ValueError, match="odd integer"):
            smooth(np.array([1.0, 2.0]), window)


class TestDetectCorners:
    def test_two_clear_corners(self) -> None:
        """Two unmistakable corners: right count, right places, right order.

        The ordering assertions are the sharp ones — brake_1 < apex_1 <
        brake_2 < apex_2 is what fails if a braking point gets attached to the
        wrong corner.
        """
        distance, speed = make_trace(
            [
                (0.0, 300.0),
                (500.0, 300.0),
                (700.0, 100.0),
                (1100.0, 280.0),
                (1400.0, 280.0),
                (1600.0, 150.0),
                (1900.0, 300.0),
            ]
        )
        corners = detect_corners(distance, speed, engine="numpy")
        assert len(corners) == 2
        # A tolerance band, not an exact index: smoothing may shift an apex by
        # a sample or two, and that is acceptable behaviour.
        apex_1, apex_2 = corners.apex_indices
        assert 680.0 <= distance[apex_1] <= 720.0
        assert 1580.0 <= distance[apex_2] <= 1620.0
        brake_1, brake_2 = corners.brake_indices
        assert brake_1 < apex_1
        assert apex_1 < brake_2 < apex_2
        # Braking for turn 1 starts back on the preceding straight.
        assert distance[brake_1] <= 520.0

    def test_small_lift_is_not_a_corner(self) -> None:
        """A 10 km/h lift is a local minimum, but below the prominence threshold.

        This is the false-positive filter: without it every ripple in the trace
        would be reported as a corner.
        """
        distance, speed = make_trace(
            [(0.0, 300.0), (400.0, 300.0), (500.0, 290.0), (600.0, 300.0), (1000.0, 300.0)]
        )
        corners = detect_corners(distance, speed, engine="numpy")
        assert len(corners) == 0

    def test_double_apex_merges_into_the_slower_one(self) -> None:
        """One trace, two thresholds, two outcomes — so this tests the parameter.

        With min_separation_m=100 the double-apex complex is a single corner
        (reported at the slower of the two minima); with 20 it is two corners.
        """
        distance, speed = make_trace(
            [
                (0.0, 300.0),
                (400.0, 300.0),
                (500.0, 120.0),
                (520.0, 160.0),
                (540.0, 100.0),
                (800.0, 300.0),
            ]
        )
        merged = detect_corners(
            distance, speed, smooth_window=3, min_separation_m=100.0, engine="numpy"
        )
        assert len(merged) == 1
        assert 520.0 <= distance[merged.apex_indices[0]] <= 560.0

        separate = detect_corners(
            distance, speed, smooth_window=3, min_separation_m=20.0, engine="numpy"
        )
        assert len(separate) == 2

    def test_flat_trace_has_no_corners(self) -> None:
        # Constant speed: no local minima, so nothing to report.
        distance = np.arange(0.0, 500.0, STEP_M)
        speed = np.full_like(distance, 300.0)
        corners = detect_corners(distance, speed, engine="numpy")
        assert len(corners) == 0

    def test_too_short_trace_returns_empty(self) -> None:
        # Valid input, empty answer — not an error. Pins the boundary between
        # "malformed" and "nothing to find".
        corners = detect_corners(np.array([0.0, 5.0]), np.array([300.0, 300.0]), engine="numpy")
        assert isinstance(corners, Corners)
        assert len(corners) == 0

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            detect_corners(np.zeros(5), np.zeros(4), engine="numpy")

    def test_empty_arrays_raise(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="must not be empty"):
            detect_corners(empty, empty, engine="numpy")

    def test_invalid_parameters_raise(self) -> None:
        # Each tuning parameter has a guard; all three are checked here.
        distance = np.arange(0.0, 100.0, STEP_M)
        speed = np.full_like(distance, 300.0)
        with pytest.raises(ValueError, match="odd integer"):
            detect_corners(distance, speed, smooth_window=4, engine="numpy")
        with pytest.raises(ValueError, match="positive"):
            detect_corners(distance, speed, min_drop_kmh=0.0, engine="numpy")
        with pytest.raises(ValueError, match="non-negative"):
            detect_corners(distance, speed, min_separation_m=-1.0, engine="numpy")
