// Delta-time computations over telemetry arrays.
//
// Every function here has a numpy twin in src/f1lab/analysis.py; the Python
// parity tests assert both produce the same results. Keep the arithmetic
// order identical when changing either side.
//
// Inputs are borrowed views (std::span), never owned: the caller — in
// practice the pybind11 layer — keeps the underlying numpy buffers alive for
// the duration of the call. Results are returned as flat row-major vectors;
// the binding layer attaches the logical shape on the way back to Python.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace f1lab {

// Linear interpolation of the curve (xs, ys) sampled at `grid`, with
// np.interp semantics: xs must be sorted ascending, and grid points outside
// [xs.front(), xs.back()] clamp to the edge ys values.
//
// A sorted `grid` (the common case) is interpolated in O(n + m) by walking
// both sequences together; an unsorted grid falls back to a binary search
// per point. Throws std::invalid_argument if xs and ys differ in size or xs
// is empty.
std::vector<double> interp(std::span<const double> grid, std::span<const double> xs,
                           std::span<const double> ys);

// Time delta "other - ref" resampled onto the reference distance grid.
// Positive values mean the other driver is behind the reference driver.
//
// Two laps cannot be compared sample by sample — the cars reach the same
// point of the track at different times — so the other lap's time curve is
// first resampled onto the reference lap's distance grid, then subtracted.
//
// Each (distance, time) pair must be equal in length and non-empty;
// std::invalid_argument otherwise. The result has the length of ref_*.
std::vector<double> delta_time(std::span<const double> ref_distance_m,
                               std::span<const double> ref_time_s,
                               std::span<const double> other_distance_m,
                               std::span<const double> other_time_s);

// n drivers' lap-time curves resampled onto one shared distance grid.
//
// Returns a flat row-major (n, grid.size()) matrix: row i holds lap i's
// elapsed time at every grid point. This is the shared first step of any
// multi-lap comparison — once every lap lives on the same distance axis,
// laps can be subtracted element-wise.
//
// `distances` and `times` must have the same length, each lap must be a
// non-empty equal-length pair, and the grid must be non-empty;
// std::invalid_argument otherwise.
std::vector<double> resample_times(const std::vector<std::span<const double>>& distances,
                                   const std::vector<std::span<const double>>& times,
                                   std::span<const double> grid);

// Pairwise delta cube: out[i][j][k] = time_j(grid[k]) - time_i(grid[k]),
// i.e. how far driver j is behind driver i at grid point k.
//
// Returns a flat row-major (n, n, grid.size()) tensor, antisymmetric with a
// zero diagonal. Resampling happens once per lap rather than once per pair,
// so the cost is n interpolations plus n*n*g subtractions — not n*n
// interpolations. Validates its inputs through resample_times.
std::vector<double> pairwise_delta_matrix(const std::vector<std::span<const double>>& distances,
                                          const std::vector<std::span<const double>>& times,
                                          std::span<const double> grid);

}  // namespace f1lab
