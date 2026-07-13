// Corner detection from a speed-vs-distance trace.
//
// Mirrored by the numpy implementation in src/f1lab/corners.py. The smoothing
// uses an explicit prefix sum so both sides perform the exact same float
// operations in the exact same order — the parity tests compare indices for
// strict equality, not just closeness.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace f1lab {

// One entry per detected corner, in track order. Indices point into the
// caller's original telemetry arrays (not the smoothed copy), so the caller
// can look up distance, speed, time or any other channel at those samples.
// Both vectors always have the same length.
struct CornerResult {
    // The slowest point of the corner: the apex.
    std::vector<std::int64_t> apex_indices;
    // Where braking for that corner began: the speed peak on the approach.
    std::vector<std::int64_t> brake_indices;
};

// Centered moving average over `values`.
//
// The window shrinks at the array edges rather than padding with zeros,
// which would fake a speed drop at the start and end of the lap. `window`
// must be odd (so it has a true centre) and >= 1; std::invalid_argument
// otherwise. A window of 1 returns a copy of the input.
std::vector<double> smooth(std::span<const double> values, std::size_t window);

// Detect corners as prominent local minima of the smoothed speed trace.
//
// Raw speed is noisy, so minima are found on a `smooth_window`-wide moving
// average. A local minimum qualifies as an apex when, walking outwards in
// both directions, the smoothed speed rises by at least `min_drop_kmh`
// before any sample dips below the minimum itself — a dip means a deeper
// minimum nearby shadows this candidate, which is therefore not its own
// corner. This makes detection robust to small wiggles in the trace, unlike
// comparing a candidate only against its immediate neighbours.
//
// Apexes closer together than `min_separation_m` are merged into the slower
// one, so a chicane or double-apex complex is reported as a single corner.
//
// distance_m and speed_kmh must be equal in length and non-empty,
// min_drop_kmh positive and min_separation_m non-negative;
// std::invalid_argument otherwise. Traces shorter than three samples cannot
// contain a local minimum and yield an empty result rather than an error.
CornerResult detect_corners(std::span<const double> distance_m, std::span<const double> speed_kmh,
                            std::size_t smooth_window, double min_drop_kmh,
                            double min_separation_m);

}  // namespace f1lab
