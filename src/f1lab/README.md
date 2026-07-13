# src/f1lab/ — the Python package

The user-facing half of the project: it loads a session, computes the
comparison, and renders the plot. It also holds the **numpy reference
implementation** of every calculation the C++ engine provides — the two are
kept equivalent by the parity tests.

```
src/f1lab/
├── compare_laps.py   FastF1 loading, matplotlib rendering, the CLI
├── analysis.py       delta time, resampling, the pairwise delta cube
├── corners.py        apex and braking-point detection
├── native.py         loads f1lab._native; decides which engine a call uses
├── bench.py          numpy vs C++ benchmark (python -m f1lab.bench)
└── _native.pyi       type stub for the compiled extension
```

## Layering

`analysis.py` and `corners.py` never import FastF1. They take plain numpy
arrays and return plain numpy arrays, which is what lets the unit tests run
without a network connection, replacing the session object with a few small
fakes. Everything that talks to the outside world — downloading a session,
drawing a figure, parsing arguments — lives in `compare_laps.py`.

The practical consequence: to use this as a library, you only need the pure
half.

```python
import numpy as np
from f1lab.analysis import compute_delta_time
from f1lab.corners import detect_corners

delta = compute_delta_time(ref_distance_m, ref_time_s, other_distance_m, other_time_s)
corners = detect_corners(distance_m, speed_kmh)
apex_speeds = speed_kmh[corners.apex_indices]
```

## Engine dispatch

Every computational function takes an optional `engine` argument and asks
`native.resolve_engine()` which implementation to use:

| `engine` | Behaviour |
|----------|-----------|
| omitted / `"auto"` | C++ if the extension was built, numpy otherwise |
| `"cpp"` | C++, or `RuntimeError` if the extension is missing |
| `"numpy"` | the reference implementation, always available |

Two decisions worth spelling out.

**Asking for `"cpp"` when it is unavailable raises instead of falling back.**
A silent fallback would be friendlier and dishonest: `bench.py` and the parity
tests both request `"cpp"` explicitly, and if that request could be served by
numpy they would be comparing numpy against itself while claiming otherwise.
The automatic path is the one that falls back; the explicit one does not.

**Validation happens in Python, before the dispatch.** Both engines therefore
reject the same inputs with the same `ValueError` messages — the C++ side
raises `std::invalid_argument` with matching text, which pybind11 converts.
Behaviour on bad input is part of the parity contract, not an afterthought.

## Keeping the two implementations in step

The numpy code here is the reference; `cpp/src/` mirrors it. Where the float
arithmetic could diverge, the Python side is written to match the C++ side
rather than to be idiomatic — the clearest case is the prefix-sum smoothing in
`corners.py`, which uses `np.cumsum` because that performs the same additions
in the same order as the C++ loop. Float addition is not associative, so the
order is part of the contract: it is what allows `tests/test_native_parity.py`
to compare corner indices for exact equality rather than approximate agreement.

If you change one implementation, change the other in the same commit.

## Tests

| File | Covers |
|------|--------|
| `tests/test_analysis.py` | delta, resampling, the pairwise cube, formatting |
| `tests/test_corners.py` | corner detection, pinned to the numpy engine |
| `tests/test_native.py` | engine selection and the fallback rules |
| `tests/test_native_parity.py` | the two engines agree (skipped without the extension) |
| `tests/test_compare_laps.py` | CLI parsing and the rendering path |
| `tests/test_bench.py` | the benchmark's data generator and timer |

`tests/test_corners.py` passes `engine="numpy"` explicitly on every call. Left
to the default it would exercise C++ on a machine where the extension is built,
and the numpy implementation — the code that actually runs for users without a
compiler — would never be tested.
