// Unit tests for the delta engine (cpp/src/delta.cpp).
//
// Pure C++: no Python, no pybind11. The core is a separate CMake target from
// the bindings precisely so it can be tested on its own — a failure here is a
// bug in the algorithm, never in the boundary layer.
//
// Expected values come from closed-form arithmetic (a car at a constant speed
// covers d metres in d/v seconds), not from a previous run of this code, so
// the tests cannot rubber-stamp a regression. Where a property is stronger
// than any single value — antisymmetry of the delta cube, for instance — the
// property is asserted across the whole result instead.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <vector>

#include "delta.hpp"

// Floats are compared with a tolerance, never with ==; rounding makes exact
// equality a coin toss.
using Catch::Matchers::WithinAbs;

namespace {

// np.linspace: `count` evenly spaced samples from start to stop inclusive.
std::vector<double> linspace(double start, double stop, std::size_t count) {
    std::vector<double> out(count);
    const double step = (stop - start) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = start + step * static_cast<double>(i);
    }
    return out;
}

// Distance / speed = time, elementwise: turns a distance array into the time
// array of a car holding that speed for the whole lap.
std::vector<double> divide(const std::vector<double>& values, double divisor) {
    std::vector<double> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i] / divisor;
    }
    return out;
}

}  // namespace

// The np.interp contract, point by point: exact at the nodes, linear between
// them, and clamped — not extrapolated — outside the curve.
TEST_CASE("interp matches endpoints and clamps outside the range") {
    const std::vector<double> xs = {0.0, 10.0, 20.0};
    const std::vector<double> ys = {0.0, 100.0, 150.0};
    const std::vector<double> grid = {-5.0, 0.0, 5.0, 15.0, 20.0, 25.0};
    const std::vector<double> out = f1lab::interp(grid, xs, ys);
    REQUIRE(out.size() == grid.size());
    CHECK_THAT(out[0], WithinAbs(0.0, 1e-12));    // clamped left
    CHECK_THAT(out[1], WithinAbs(0.0, 1e-12));    // exact node
    CHECK_THAT(out[2], WithinAbs(50.0, 1e-12));   // halfway 0 -> 100
    CHECK_THAT(out[3], WithinAbs(125.0, 1e-12));  // halfway 100 -> 150
    CHECK_THAT(out[4], WithinAbs(150.0, 1e-12));  // exact last node
    CHECK_THAT(out[5], WithinAbs(150.0, 1e-12));  // clamped right
}

// interp has two code paths — the merge walk for sorted grids and a binary
// search otherwise — and only the fast one runs in production. This feeds the
// same points through both and demands identical answers, so the optimisation
// can never quietly change a result.
TEST_CASE("interp gives the same answers for an unsorted grid") {
    const std::vector<double> xs = {0.0, 10.0, 20.0};
    const std::vector<double> ys = {0.0, 100.0, 150.0};
    const std::vector<double> sorted_grid = {-5.0, 0.0, 5.0, 15.0, 25.0};
    const std::vector<double> shuffled_grid = {15.0, -5.0, 25.0, 5.0, 0.0};
    const std::vector<double> from_sorted = f1lab::interp(sorted_grid, xs, ys);
    const std::vector<double> from_shuffled = f1lab::interp(shuffled_grid, xs, ys);
    // Same x, same y — whichever order the points were asked for.
    CHECK(from_shuffled[0] == from_sorted[3]);
    CHECK(from_shuffled[1] == from_sorted[0]);
    CHECK(from_shuffled[2] == from_sorted[4]);
    CHECK(from_shuffled[3] == from_sorted[2]);
    CHECK(from_shuffled[4] == from_sorted[1]);
}

// The baseline sanity check: a lap compared with itself is nowhere ahead or
// behind. Catches sign flips and off-by-one errors in the resampling.
TEST_CASE("identical laps give a zero delta") {
    const std::vector<double> distance = linspace(0.0, 1000.0, 51);
    const std::vector<double> time = divide(distance, 50.0);
    const std::vector<double> delta = f1lab::delta_time(distance, time, distance, time);
    REQUIRE(delta.size() == distance.size());
    for (const double value : delta) {
        CHECK_THAT(value, WithinAbs(0.0, 1e-12));
    }
}

// Physics as the oracle: 40 m/s against 50 m/s means the slower car is
// d/40 - d/50 = d/200 seconds behind at distance d, and exactly 5 s down
// after a kilometre. The expected curve is derived, not recorded.
TEST_CASE("constant speed difference grows linearly") {
    const std::vector<double> distance = linspace(0.0, 1000.0, 101);
    const std::vector<double> ref_time = divide(distance, 50.0);    // 50 m/s
    const std::vector<double> other_time = divide(distance, 40.0);  // slower
    const std::vector<double> delta = f1lab::delta_time(distance, ref_time, distance, other_time);
    for (std::size_t i = 0; i < delta.size(); ++i) {
        CHECK_THAT(delta[i], WithinAbs(distance[i] / 200.0, 1e-9));
    }
    CHECK_THAT(delta.back(), WithinAbs(5.0, 1e-9));
}

// Malformed telemetry must be rejected, not silently truncated. The exception
// type matters: pybind11 turns std::invalid_argument into a Python
// ValueError, which is what the Python tests expect from either engine.
TEST_CASE("delta_time validates its inputs") {
    const std::vector<double> five(5, 0.0);
    const std::vector<double> four(4, 0.0);
    const std::vector<double> empty;
    CHECK_THROWS_AS(f1lab::delta_time(five, four, five, five), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::delta_time(five, five, five, four), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::delta_time(empty, empty, empty, empty), std::invalid_argument);
}

// Row i of the flat matrix must be lap i on the shared grid, whatever the lap
// was sampled at originally (21 points vs 6 here). Both laps are linear, so
// resampling is exact and the expected values are again closed-form.
TEST_CASE("resample_times produces one row per lap on the shared grid") {
    const std::vector<double> grid = linspace(0.0, 100.0, 11);
    const std::vector<double> d1 = linspace(0.0, 100.0, 21);
    const std::vector<double> d2 = linspace(0.0, 100.0, 6);  // coarser sampling
    const std::vector<double> t1 = divide(d1, 10.0);
    const std::vector<double> t2 = divide(d2, 8.0);
    const std::vector<double> matrix = f1lab::resample_times({d1, d2}, {t1, t2}, grid);
    REQUIRE(matrix.size() == 2 * grid.size());
    CHECK_THAT(matrix[5], WithinAbs(grid[5] / 10.0, 1e-12));               // row 0: 10 m/s
    CHECK_THAT(matrix[grid.size() + 5], WithinAbs(grid[5] / 8.0, 1e-12));  // row 1: 8 m/s
}

// Property-based rather than value-based: whatever the numbers are, the cube
// must satisfy delta(i, j) == -delta(j, i) and delta(i, i) == 0 in every one
// of its cells. A transposed index or a swapped subtraction breaks this
// everywhere at once, while a handful of spot-checked values might miss it.
TEST_CASE("pairwise_delta_matrix is antisymmetric with a zero diagonal") {
    const std::vector<double> grid = linspace(0.0, 500.0, 26);
    const std::vector<double> distance = linspace(0.0, 500.0, 51);
    const std::vector<std::vector<double>> times = {divide(distance, 50.0), divide(distance, 45.0),
                                                    divide(distance, 55.0)};
    const std::vector<double> cube = f1lab::pairwise_delta_matrix(
        {distance, distance, distance}, {times[0], times[1], times[2]}, grid);
    const std::size_t n = 3;
    const std::size_t g = grid.size();
    REQUIRE(cube.size() == n * n * g);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < g; ++k) {
                const double forward = cube[(i * n + j) * g + k];
                const double backward = cube[(j * n + i) * g + k];
                CHECK_THAT(forward + backward, WithinAbs(0.0, 1e-12));
                if (i == j) {
                    CHECK_THAT(forward, WithinAbs(0.0, 1e-12));
                }
            }
        }
    }
}

// The list-level failures the single-lap validation cannot see: a lap missing
// its time array, an empty grid, an empty lap.
TEST_CASE("resample_times validates lap counts and the grid") {
    const std::vector<double> d = linspace(0.0, 10.0, 5);
    const std::vector<double> t = divide(d, 2.0);
    const std::vector<double> empty;
    CHECK_THROWS_AS(f1lab::resample_times({d, d}, {t}, d), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::resample_times({d}, {t}, empty), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::resample_times({empty}, {empty}, d), std::invalid_argument);
}
