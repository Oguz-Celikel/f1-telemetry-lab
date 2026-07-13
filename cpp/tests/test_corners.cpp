// Unit tests for the corner detector (cpp/src/corners.cpp).
//
// Strategy: build synthetic laps from piecewise-linear speed profiles. Because
// the corners are placed by hand, the expected apexes are known exactly and
// the assertions do not depend on real telemetry (which would drag a network
// download into the unit tests).
//
// tests/test_corners.py mirrors these scenarios against the numpy engine, and
// tests/test_native_parity.py then asserts the two engines return identical
// indices — so the same behaviour is pinned from three directions.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include "corners.hpp"
#include "delta.hpp"

namespace {

// Sampling step of the synthetic traces, in metres.
constexpr double kStepM = 5.0;

struct Trace {
    std::vector<double> distance;
    std::vector<double> speed;
};

// Build an idealised straights-and-corners lap: (distance, speed) breakpoints
// joined by straight lines, sampled every kStepM metres. interp is reused as
// the line-filler — it is already covered by test_delta.cpp.
Trace make_trace(const std::vector<std::pair<double, double>>& breakpoints) {
    std::vector<double> xs;
    std::vector<double> ys;
    for (const auto& [x, y] : breakpoints) {
        xs.push_back(x);
        ys.push_back(y);
    }
    Trace trace;
    for (double d = 0.0; d <= xs.back(); d += kStepM) {
        trace.distance.push_back(d);
    }
    trace.speed = f1lab::interp(trace.distance, xs, ys);
    return trace;
}

}  // namespace

// The smoothing contract: a window of 1 changes nothing, and an even window
// is rejected because it has no true centre to average around.
TEST_CASE("smooth is an identity for window 1 and validates the window") {
    const std::vector<double> values = {1.0, 5.0, 2.0, 8.0};
    const std::vector<double> out = f1lab::smooth(values, 1);
    for (std::size_t i = 0; i < values.size(); ++i) {
        CHECK(out[i] == values[i]);
    }
    CHECK_THROWS_AS(f1lab::smooth(values, 0), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::smooth(values, 4), std::invalid_argument);
}

// The happy path: two unmistakable corners must yield exactly two apexes, each
// near its true minimum, each preceded by a braking point on the straight
// before it. The ordering assertions (brake_1 < apex_1 < brake_2 < apex_2) are
// what would catch a braking point being attached to the wrong corner.
TEST_CASE("two clear corners are detected with brakes before apexes") {
    // Flat-out to 500 m, corner down to 100 at 700 m, back up to 280 by
    // 1100 m, hold, second corner down to 150 at 1600 m, recover to 300.
    const Trace trace = make_trace({{0.0, 300.0},
                                    {500.0, 300.0},
                                    {700.0, 100.0},
                                    {1100.0, 280.0},
                                    {1400.0, 280.0},
                                    {1600.0, 150.0},
                                    {1900.0, 300.0}});
    const f1lab::CornerResult result =
        f1lab::detect_corners(trace.distance, trace.speed, 7, 15.0, 50.0);

    REQUIRE(result.apex_indices.size() == 2);
    REQUIRE(result.brake_indices.size() == 2);

    // Apexes land near the true minima. A tolerance band rather than an exact
    // index: smoothing legitimately shifts the minimum by a sample or two.
    const auto apex_1 = static_cast<std::size_t>(result.apex_indices[0]);
    const auto apex_2 = static_cast<std::size_t>(result.apex_indices[1]);
    CHECK(trace.distance[apex_1] >= 680.0);
    CHECK(trace.distance[apex_1] <= 720.0);
    CHECK(trace.distance[apex_2] >= 1580.0);
    CHECK(trace.distance[apex_2] <= 1620.0);

    // Each braking point precedes its own apex and follows the previous one.
    const auto brake_1 = static_cast<std::size_t>(result.brake_indices[0]);
    const auto brake_2 = static_cast<std::size_t>(result.brake_indices[1]);
    CHECK(brake_1 < apex_1);
    CHECK(brake_2 > apex_1);
    CHECK(brake_2 < apex_2);
    // Braking for turn 1 begins on the flat-out section, at top speed.
    CHECK(trace.distance[brake_1] <= 520.0);
    CHECK(trace.speed[brake_1] > trace.speed[apex_1] + 100.0);
}

// The false-positive filter. A brief lift is a local minimum too, so without
// the prominence rule it would be reported as a corner.
TEST_CASE("dips smaller than min_drop_kmh are ignored") {
    // A 10 km/h lift at 500 m is not a corner with min_drop = 15.
    const Trace trace =
        make_trace({{0.0, 300.0}, {400.0, 300.0}, {500.0, 290.0}, {600.0, 300.0}, {1000.0, 300.0}});
    const f1lab::CornerResult result =
        f1lab::detect_corners(trace.distance, trace.speed, 7, 15.0, 50.0);
    CHECK(result.apex_indices.empty());
}

// min_separation_m in both directions: the same double-apex complex counts as
// one corner when the threshold spans it and two when it does not. Driving
// both outcomes from one trace tests the parameter itself, not the trace.
TEST_CASE("apexes closer than min_separation_m merge into the slower one") {
    // Double-apex complex: minima at 500 m (120 km/h) and 540 m (100 km/h).
    const Trace trace = make_trace({{0.0, 300.0},
                                    {400.0, 300.0},
                                    {500.0, 120.0},
                                    {520.0, 160.0},
                                    {540.0, 100.0},
                                    {800.0, 300.0}});
    const f1lab::CornerResult merged =
        f1lab::detect_corners(trace.distance, trace.speed, 3, 15.0, 100.0);
    REQUIRE(merged.apex_indices.size() == 1);
    // The slower of the two minima (100 km/h at ~540 m) survives the merge.
    const auto apex = static_cast<std::size_t>(merged.apex_indices[0]);
    CHECK(trace.distance[apex] >= 520.0);
    CHECK(trace.distance[apex] <= 560.0);

    const f1lab::CornerResult separate =
        f1lab::detect_corners(trace.distance, trace.speed, 3, 15.0, 20.0);
    CHECK(separate.apex_indices.size() == 2);
}

// Every guard in the signature: mismatched lengths, empty telemetry, an even
// smoothing window, a non-positive drop threshold, a negative separation.
TEST_CASE("detect_corners validates its inputs") {
    const Trace trace = make_trace({{0.0, 300.0}, {100.0, 300.0}});
    const std::vector<double> shorter(trace.distance.size() - 1, 0.0);
    const std::vector<double> empty;
    CHECK_THROWS_AS(f1lab::detect_corners(trace.distance, shorter, 7, 15.0, 50.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(f1lab::detect_corners(empty, empty, 7, 15.0, 50.0), std::invalid_argument);
    CHECK_THROWS_AS(f1lab::detect_corners(trace.distance, trace.speed, 4, 15.0, 50.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(f1lab::detect_corners(trace.distance, trace.speed, 7, 0.0, 50.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(f1lab::detect_corners(trace.distance, trace.speed, 7, 15.0, -1.0),
                    std::invalid_argument);
}

// A trace too short to hold an interior local minimum is not an error — there
// simply are no corners in it. Pins the boundary between "invalid input" and
// "valid input, empty answer".
TEST_CASE("traces with fewer than three samples yield no corners") {
    const std::vector<double> distance = {0.0, 5.0};
    const std::vector<double> speed = {300.0, 300.0};
    const f1lab::CornerResult result = f1lab::detect_corners(distance, speed, 7, 15.0, 50.0);
    CHECK(result.apex_indices.empty());
    CHECK(result.brake_indices.empty());
}
