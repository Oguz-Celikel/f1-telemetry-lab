#include "delta.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace f1lab {

namespace {

// Error messages mirror the ValueError texts raised by f1lab.analysis, so the
// caller sees identical failures whichever backend handled the call — the
// Python tests match on these strings for both engines.
void require_pair(std::span<const double> distance, std::span<const double> time,
                  const char* label) {
    if (distance.size() != time.size()) {
        throw std::invalid_argument(std::string(label) +
                                    " distance and time arrays must have the same shape");
    }
    if (distance.empty()) {
        throw std::invalid_argument("telemetry arrays must not be empty");
    }
}

// Interpolate a single point. `hi` is the index of the first sample strictly
// greater than x (upper-bound semantics); both search strategies below feed
// into this one function so they cannot drift apart.
double interp_at(std::size_t hi, double x, std::span<const double> xs, std::span<const double> ys) {
    if (hi == 0) {
        return ys.front();  // x is left of the curve: clamp.
    }
    if (hi == xs.size()) {
        return ys.back();  // x is right of the curve: clamp.
    }
    // Written exactly as np.interp evaluates it — slope first, then
    // multiply-add — so both engines round identically.
    const double slope = (ys[hi] - ys[hi - 1]) / (xs[hi] - xs[hi - 1]);
    return slope * (x - xs[hi - 1]) + ys[hi - 1];
}

// Interpolate into a caller-provided buffer. Separating this from the
// allocating `interp` lets resample_times fill each row of its matrix in
// place, without a temporary vector per lap.
void interp_into(std::span<double> out, std::span<const double> grid, std::span<const double> xs,
                 std::span<const double> ys) {
    if (xs.size() != ys.size()) {
        throw std::invalid_argument("xs and ys must have the same shape");
    }
    if (xs.empty()) {
        throw std::invalid_argument("xs must not be empty");
    }
    if (std::is_sorted(grid.begin(), grid.end())) {
        // Fast path. Telemetry distance grids are monotonic, so the two
        // sorted sequences can be merged in one pass: `hi` never moves
        // backwards, giving O(n + m) instead of O(m log n) for a binary
        // search per point. np.interp exploits the same property internally,
        // and without this the C++ engine loses to numpy on delta workloads.
        std::size_t hi = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            const double x = grid[i];
            while (hi < xs.size() && xs[hi] <= x) {
                ++hi;
            }
            out[i] = interp_at(hi, x, xs, ys);
        }
    } else {
        // General path: correctness does not depend on a sorted grid.
        for (std::size_t i = 0; i < grid.size(); ++i) {
            const double x = grid[i];
            const auto it = std::upper_bound(xs.begin(), xs.end(), x);
            out[i] = interp_at(static_cast<std::size_t>(it - xs.begin()), x, xs, ys);
        }
    }
}

}  // namespace

std::vector<double> interp(std::span<const double> grid, std::span<const double> xs,
                           std::span<const double> ys) {
    std::vector<double> out(grid.size());
    interp_into(out, grid, xs, ys);
    return out;
}

std::vector<double> delta_time(std::span<const double> ref_distance_m,
                               std::span<const double> ref_time_s,
                               std::span<const double> other_distance_m,
                               std::span<const double> other_time_s) {
    require_pair(ref_distance_m, ref_time_s, "reference");
    require_pair(other_distance_m, other_time_s, "other");
    // Ask the other lap "what was your elapsed time at each metre the
    // reference driver passed?", then subtract the reference's own time.
    std::vector<double> out = interp(ref_distance_m, other_distance_m, other_time_s);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] -= ref_time_s[i];
    }
    return out;
}

std::vector<double> resample_times(const std::vector<std::span<const double>>& distances,
                                   const std::vector<std::span<const double>>& times,
                                   std::span<const double> grid) {
    if (distances.size() != times.size()) {
        throw std::invalid_argument("distances and times must contain the same number of laps");
    }
    if (grid.empty()) {
        throw std::invalid_argument("grid must not be empty");
    }
    const std::size_t n = distances.size();
    const std::size_t g = grid.size();
    // Allocate the whole (n, g) matrix once and let each lap interpolate
    // straight into its own row.
    std::vector<double> out(n * g);
    for (std::size_t i = 0; i < n; ++i) {
        require_pair(distances[i], times[i], "lap");
        interp_into({out.data() + i * g, g}, grid, distances[i], times[i]);
    }
    return out;
}

std::vector<double> pairwise_delta_matrix(const std::vector<std::span<const double>>& distances,
                                          const std::vector<std::span<const double>>& times,
                                          std::span<const double> grid) {
    // Interpolate once per lap, not once per pair: with 22 drivers that is 22
    // interpolations feeding 484 cheap subtractions.
    const std::vector<double> resampled = resample_times(distances, times, grid);
    const std::size_t n = distances.size();
    const std::size_t g = grid.size();
    std::vector<double> out(n * n * g);
    for (std::size_t i = 0; i < n; ++i) {
        const double* row_i = resampled.data() + i * g;
        for (std::size_t j = 0; j < n; ++j) {
            const double* row_j = resampled.data() + j * g;
            // Innermost loop walks contiguous memory, which vectorises well.
            double* cell = out.data() + (i * n + j) * g;
            for (std::size_t k = 0; k < g; ++k) {
                cell[k] = row_j[k] - row_i[k];
            }
        }
    }
    return out;
}

}  // namespace f1lab
