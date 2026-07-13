"""Corner detection from a speed-vs-distance telemetry trace.

A corner is a prominent local minimum of the smoothed speed trace (the apex),
paired with the point where braking for it started (the local speed maximum
just before). The numpy code here is the reference implementation; the C++
core (``cpp/src/corners.cpp``) mirrors it operation for operation — including
the prefix-sum smoothing — so both backends return *identical* indices, which
the parity tests assert exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1lab import native

# FastF1 car data arrives at ~4 Hz (roughly one sample per 18 m at speed), so
# a 3-sample window already averages over ~55 m of track; anything wider
# starts to blur genuinely separate corners (e.g. Silverstone's Village/Loop
# complex) into one. Noise robustness comes from the prominence walk, not
# from heavy smoothing.
DEFAULT_SMOOTH_WINDOW = 3
DEFAULT_MIN_DROP_KMH = 15.0
DEFAULT_MIN_SEPARATION_M = 50.0


@dataclass(frozen=True)
class Corners:
    """Detected corners, in track order.

    Both arrays have the same length and hold indices into the caller's
    original telemetry — not values — so any channel can be looked up at those
    samples: ``distance_m[corners.apex_indices]`` gives the apex distances,
    ``speed_kmh[corners.brake_indices]`` the speeds at the braking points.
    """

    apex_indices: NDArray[np.int64]
    brake_indices: NDArray[np.int64]

    def __len__(self) -> int:
        """Number of corners detected."""
        return int(self.apex_indices.size)


def smooth(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Centered moving average; the window shrinks at the array edges."""
    if window < 1 or window % 2 == 0:
        raise ValueError("smooth_window must be an odd integer >= 1")
    n = values.size
    half = window // 2
    # Prefix-sum formulation: same operation order as the C++ core, so both
    # backends round identically.
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half, n - 1)
    return np.asarray((prefix[hi + 1] - prefix[lo]) / (hi - lo + 1), dtype=np.float64)


def detect_corners(
    distance_m: NDArray[np.float64],
    speed_kmh: NDArray[np.float64],
    *,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    min_drop_kmh: float = DEFAULT_MIN_DROP_KMH,
    min_separation_m: float = DEFAULT_MIN_SEPARATION_M,
    engine: str | None = None,
) -> Corners:
    """Find apex and braking points in a single lap's speed trace.

    A local minimum counts as an apex when, walking outwards in both
    directions, the smoothed speed rises by at least ``min_drop_kmh`` before
    any sample dips below the minimum itself (a deeper nearby minimum shadows
    it). Apexes closer together than ``min_separation_m`` merge into the
    slower one.
    """
    if distance_m.shape != speed_kmh.shape:
        raise ValueError("distance and speed arrays must have the same shape")
    if distance_m.size == 0:
        raise ValueError("telemetry arrays must not be empty")
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError("smooth_window must be an odd integer >= 1")
    if min_drop_kmh <= 0.0:
        raise ValueError("min_drop_kmh must be positive")
    if min_separation_m < 0.0:
        raise ValueError("min_separation_m must be non-negative")
    if native.resolve_engine(engine) == "cpp":
        apex, brake = native.detect_corners(
            distance_m, speed_kmh, smooth_window, min_drop_kmh, min_separation_m
        )
        return Corners(apex_indices=apex, brake_indices=brake)
    return _detect_corners_numpy(
        distance_m, speed_kmh, smooth_window, min_drop_kmh, min_separation_m
    )


def _detect_corners_numpy(
    distance_m: NDArray[np.float64],
    speed_kmh: NDArray[np.float64],
    smooth_window: int,
    min_drop_kmh: float,
    min_separation_m: float,
) -> Corners:
    """Reference implementation of :func:`detect_corners`, in five steps.

    Inputs are assumed already validated by the caller. The prominence and
    braking-point walks are data-dependent loops that do not vectorise, which
    is precisely why the C++ twin is roughly 100x faster here — see
    ``python -m f1lab.bench``.
    """
    n = speed_kmh.size
    empty = np.array([], dtype=np.int64)
    if n < 3:
        return Corners(apex_indices=empty, brake_indices=empty)
    smoothed = smooth(speed_kmh, smooth_window)

    # Candidate apexes: local minima of the smoothed trace. `<=` on the left
    # and `<` on the right picks the last sample of a flat valley floor.
    interior = np.arange(1, n - 1)
    is_minimum = (smoothed[1:-1] <= smoothed[:-2]) & (smoothed[1:-1] < smoothed[2:])
    candidates = interior[is_minimum]

    # Prominence: walk outwards from the candidate in both directions. The
    # candidate is a corner if the speed rises by at least min_drop on each
    # side before any sample dips below the candidate itself (in which case a
    # deeper minimum nearby shadows this one). Robust to small wiggles,
    # unlike comparing against neighbouring candidates.
    accepted: list[int] = []
    for candidate in candidates:
        i = int(candidate)
        rises_left = False
        for k in range(i - 1, -1, -1):
            if smoothed[k] < smoothed[i]:
                break
            if smoothed[k] - smoothed[i] >= min_drop_kmh:
                rises_left = True
                break
        rises_right = False
        for k in range(i + 1, n):
            if smoothed[k] < smoothed[i]:
                break
            if smoothed[k] - smoothed[i] >= min_drop_kmh:
                rises_right = True
                break
        if rises_left and rises_right:
            accepted.append(i)

    # Merge apexes closer than min_separation_m, keeping the slower one.
    kept: list[int] = []
    for i in accepted:
        if kept and distance_m[i] - distance_m[kept[-1]] < min_separation_m:
            if smoothed[i] < smoothed[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)

    # Braking point: walk back from the apex tracking the running maximum;
    # stop once the trace falls min_drop below that maximum (we have crossed
    # the peak of the approach and are descending into the previous corner).
    brakes: list[int] = []
    for apex in kept:
        best = apex
        k = apex
        while k > 0:
            k -= 1
            if smoothed[k] > smoothed[best]:
                best = k
            if smoothed[best] - smoothed[k] >= min_drop_kmh:
                break
        brakes.append(best)

    return Corners(
        apex_indices=np.asarray(kept, dtype=np.int64),
        brake_indices=np.asarray(brakes, dtype=np.int64),
    )
