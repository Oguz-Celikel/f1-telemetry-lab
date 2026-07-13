"""Type stub for the compiled extension built from ``cpp/src/bindings.cpp``.

mypy cannot see inside a ``.so``, so the signatures live here. Nothing reads
this file at runtime — Python imports the compiled module itself. Keep it in
step with the ``m.def`` calls in ``bindings.cpp``.

The functions are documented on their numpy twins in ``f1lab.analysis`` and
``f1lab.corners``, which is what callers actually use; these are the raw
kernels the dispatch layer forwards to once its own validation has passed.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def delta_time(
    ref_distance_m: NDArray[np.float64],
    ref_time_s: NDArray[np.float64],
    other_distance_m: NDArray[np.float64],
    other_time_s: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def resample_times(
    distances: Sequence[NDArray[np.float64]],
    times: Sequence[NDArray[np.float64]],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def pairwise_delta_matrix(
    distances: Sequence[NDArray[np.float64]],
    times: Sequence[NDArray[np.float64]],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def detect_corners(
    distance_m: NDArray[np.float64],
    speed_kmh: NDArray[np.float64],
    smooth_window: int,
    min_drop_kmh: float,
    min_separation_m: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
