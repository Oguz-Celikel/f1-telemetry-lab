"""Parity tests: the C++ engine must reproduce the numpy engine's results.

This is the suite that earns the project's central claim — the same analysis,
implemented twice, proven equivalent. Without it the fallback would be a
liability: users on a machine without a compiler would silently get *different*
numbers rather than the same ones more slowly.

Two levels of strictness, on purpose:

* Deltas and matrices are compared at ``rtol=atol=1e-12``. Interpolation is
  arithmetic, and the two implementations evaluate it in the same order, so
  they agree to the last few bits — a loose tolerance would hide real drift.
* Corner detection is compared for *exact* index equality. Indices are the
  output of comparisons against thresholds, so a single differently-rounded
  smoothing value could move an apex by a whole sample. Demanding exactness is
  what forces both engines to keep their float operations in the same order
  (prefix-sum smoothing mirroring ``np.cumsum``); a tolerance here would let
  the two drift apart unnoticed.

Inputs are pseudo-random rather than hand-picked: hand-picked laps only prove
the engines agree on the cases their author thought of. Each test seeds its own
generator, so failures are reproducible.

The whole module is skipped — not failed — when f1lab was installed without the
extension, mirroring the package's own "native is optional" contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from f1lab import native
from f1lab.analysis import compute_delta_time, pairwise_delta_matrix, resample_times
from f1lab.corners import detect_corners

pytestmark = pytest.mark.skipif(not native.HAS_NATIVE, reason="f1lab._native extension not built")

Lap = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def random_lap(rng: np.random.Generator, samples: int) -> Lap:
    """A jittery but physically consistent (distance, time, speed) triple.

    Uneven sampling steps and noisy speed are deliberate: the noise is what
    stresses the corner detector's prominence walk, and an early version of
    that algorithm was caught by exactly this kind of input, not by the clean
    synthetic traces of the unit tests.
    """
    steps = rng.uniform(0.5, 3.0, samples)
    distance = np.cumsum(steps)
    speed = 210.0 + 90.0 * np.sin(distance / 250.0 + rng.uniform(0.0, 6.0))
    speed = np.clip(speed + rng.normal(0.0, 5.0, samples), 60.0, 340.0)
    # Time integrated from the speed the car is actually doing, so the arrays
    # describe a lap that could physically happen.
    time_s = np.cumsum(steps / (speed / 3.6))
    return distance, time_s, np.asarray(speed, dtype=np.float64)


@pytest.mark.parametrize("seed", range(5))
def test_delta_time_parity(seed: int) -> None:
    """Both engines produce the same delta trace, shape and dtype included.

    The two laps are sampled at different rates (700 vs 650 points) so the
    resampling path is genuinely exercised rather than short-circuited.
    """
    rng = np.random.default_rng(seed)
    ref_d, ref_t, _ = random_lap(rng, 700)
    oth_d, oth_t, _ = random_lap(rng, 650)
    out_numpy = compute_delta_time(ref_d, ref_t, oth_d, oth_t, engine="numpy")
    out_cpp = compute_delta_time(ref_d, ref_t, oth_d, oth_t, engine="cpp")
    assert out_cpp.dtype == np.float64
    assert out_cpp.shape == out_numpy.shape
    np.testing.assert_allclose(out_cpp, out_numpy, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_resample_and_pairwise_parity(seed: int) -> None:
    """The multi-lap paths agree, including the shape of the (n, n, g) cube."""
    rng = np.random.default_rng(100 + seed)
    # Laps of differing lengths, as in a real session.
    laps = [random_lap(rng, int(rng.integers(300, 800))) for _ in range(6)]
    distances = [lap[0] for lap in laps]
    times = [lap[1] for lap in laps]
    # Stop the grid at the shortest lap so every lap is interpolated inside its
    # own range — otherwise the comparison would mostly be testing edge clamping.
    grid = np.linspace(0.0, min(float(d[-1]) for d in distances), 400)

    resampled_numpy = resample_times(distances, times, grid, engine="numpy")
    resampled_cpp = resample_times(distances, times, grid, engine="cpp")
    assert resampled_cpp.shape == (6, 400)
    np.testing.assert_allclose(resampled_cpp, resampled_numpy, rtol=1e-12, atol=1e-12)

    cube_numpy = pairwise_delta_matrix(distances, times, grid, engine="numpy")
    cube_cpp = pairwise_delta_matrix(distances, times, grid, engine="cpp")
    assert cube_cpp.shape == (6, 6, 400)
    np.testing.assert_allclose(cube_cpp, cube_numpy, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_detect_corners_parity_is_exact(seed: int) -> None:
    """Corner indices match exactly — no tolerance, on eight noisy laps."""
    rng = np.random.default_rng(200 + seed)
    distance, _, speed = random_lap(rng, 2000)
    corners_numpy = detect_corners(distance, speed, engine="numpy")
    corners_cpp = detect_corners(distance, speed, engine="cpp")
    np.testing.assert_array_equal(corners_cpp.apex_indices, corners_numpy.apex_indices)
    np.testing.assert_array_equal(corners_cpp.brake_indices, corners_numpy.brake_indices)
    assert corners_cpp.apex_indices.dtype == np.int64
    # Two empty results are also "equal", which would make the assertions above
    # vacuous. This line demands the engines actually agreed on something — and
    # it is what caught an early prominence rule that found no corners at all.
    assert len(corners_cpp) > 0


def test_default_engine_is_cpp_when_built() -> None:
    """With the extension installed, the automatic choice is the C++ engine.

    Guards the benchmark's honesty: if auto-selection silently fell back to
    numpy, `just bench` would be timing numpy against itself.
    """
    assert native.backend_name() == "cpp"
    ref = np.linspace(0.0, 100.0, 11)
    out = compute_delta_time(ref, ref / 10.0, ref, ref / 10.0)  # auto engine
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_cpp_engine_validates_like_numpy() -> None:
    """Failure behaviour is part of parity: same exception, same message."""
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="must not be empty"):
        compute_delta_time(empty, empty, empty, empty, engine="cpp")
    with pytest.raises(ValueError, match="same shape"):
        compute_delta_time(np.zeros(5), np.zeros(4), np.zeros(5), np.zeros(5), engine="cpp")
