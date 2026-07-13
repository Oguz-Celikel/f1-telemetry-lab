# cpp/ — the native telemetry engine

The C++ core behind the `f1lab._native` extension module.

```
cpp/
├── CMakeLists.txt        core library, pybind11 module, Catch2 tests
├── src/
│   ├── delta.{hpp,cpp}     np.interp-compatible interpolation, lap deltas,
│   │                       the pairwise (n, n, grid) delta cube
│   ├── corners.{hpp,cpp}   smoothing, apex and braking-point detection
│   └── bindings.cpp        the pybind11 module: zero-copy numpy I/O
└── tests/                  Catch2 unit tests — pure C++, no Python involved
```

The core (`delta`, `corners`) knows nothing about Python. The bindings are a
thin layer on top. That separation is what lets the Catch2 suite compile and
run without Python or pybind11 present, and it means a failing test points at
the algorithm rather than at the boundary.

## How it is built

Two consumers share one `CMakeLists.txt`:

- **`pip install .`** — and therefore `just build`. scikit-build-core configures
  this directory with `SKBUILD` defined, compiles `_native`, and installs it
  into the `f1lab` package next to the Python sources. No manual CMake step.
- **`just test-cpp`** — configures the directory standalone with
  `-DF1LAB_BUILD_TESTS=ON`. Catch2 is fetched with `FetchContent` (pinned to a
  release tarball), and the suite runs under `ctest`. Python is not involved.

Tests are off by default so that a `pip install` never downloads Catch2.

## The boundary

`bindings.cpp` is deliberately small. It does three things:

1. **Borrows numpy buffers without copying.** Arrays arrive as
   `py::array_t<double, c_style | forcecast>` and are handed to the core as
   `std::span`. A well-formed float64 array costs nothing; anything else (a
   float32 array, a non-contiguous slice) is converted exactly once on the way
   in.
2. **Hands results back without copying.** A result `std::vector` is moved to
   the heap and wrapped in a numpy array whose owner is a `py::capsule`. When
   the array's refcount reaches zero the capsule frees the vector — ownership
   passes from C++ to Python's garbage collector.
3. **Releases the GIL** around every call into the core, so other Python
   threads can run while C++ computes.

Errors need no translation: the core throws `std::invalid_argument` with the
same message text the numpy implementation uses, and pybind11 surfaces it as a
Python `ValueError`.

## Ground rules

Every exported function has a numpy twin in `src/f1lab/`. **Keep the float
arithmetic identical in operation and order on both sides.** Float addition is
not associative, so the order is not an implementation detail — it is what
makes the two engines agree bit for bit. The clearest example is the prefix-sum
smoothing in `corners.cpp`, written to match `np.cumsum` exactly.

This is what allows `tests/test_native_parity.py` to compare corner indices for
*exact* equality rather than approximate agreement, which in turn is what makes
the numpy fallback trustworthy: users without a compiler get the same numbers,
not merely similar ones.

If you change one implementation, change the other in the same commit.

## Performance

`just bench` times both engines on synthetic telemetry, checking they agree
before it reports anything. On an Apple M-series container (aarch64), Python
3.14, best of five runs:

| Workload | numpy | C++ | Speedup |
|----------|------:|----:|--------:|
| `compute_delta_time` — 1 pair, 3500 samples | 0.011 ms | 0.009 ms | 1.2x |
| `pairwise_delta_matrix` — 22 drivers (484 pairs), 2000-point grid | 0.555 ms | 0.363 ms | 1.5x |
| `detect_corners` — 22 speed traces | 37.702 ms | 0.370 ms | 101.8x |

The gap tracks how well the workload vectorises, not how much C++ is involved.

Interpolation is arithmetic over whole arrays, and numpy already runs it as
compiled C — so the delta workloads gain little. They gain anything at all only
because `interp` exploits the fact that telemetry distance grids are sorted:
walking both sequences together is O(n + m), where a binary search per point
would be O(m log n). `np.interp` uses the same trick internally, and before the
C++ side did, it was *slower* than numpy on these workloads.

Corner detection is the opposite case. Its prominence and braking-point passes
are data-dependent loops with early exits — the sort of thing numpy cannot
vectorise, so the reference implementation pays Python interpreter cost on every
iteration. That is where the two orders of magnitude come from.

## Build gotcha

The official `python:slim` images build CPython with `CXX=gcc`, and
scikit-build-core forwards that sysconfig value into CMake. The `gcc` driver
compiles C++ happily but does not link `libstdc++`, so the extension builds
cleanly and then dies at import with `undefined symbol:
std::runtime_error::what()`. The Dockerfile pins `ENV CXX=g++` to force the
real C++ driver.
