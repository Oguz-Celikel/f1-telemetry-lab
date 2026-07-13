#include "corners.hpp"

#include <algorithm>
#include <stdexcept>

namespace f1lab {

std::vector<double> smooth(std::span<const double> values, std::size_t window) {
    if (window < 1 || window % 2 == 0) {
        throw std::invalid_argument("smooth_window must be an odd integer >= 1");
    }
    const std::size_t n = values.size();
    // Prefix sums turn every window average into two lookups, so the whole
    // pass is O(n) rather than O(n * window). The sequential accumulation
    // also matches np.cumsum's operation order, which is what lets the two
    // engines agree bit for bit — float addition is not associative, so the
    // order is part of the contract, not an implementation detail.
    std::vector<double> prefix(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + values[i];
    }
    const std::size_t half = window / 2;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Clip the window to the array instead of padding: near the edges it
        // simply averages fewer samples. `i > half` guards the subtraction —
        // std::size_t is unsigned, so i - half would wrap around.
        const std::size_t lo = i > half ? i - half : 0;
        const std::size_t hi = std::min(i + half, n > 0 ? n - 1 : 0);
        out[i] = (prefix[hi + 1] - prefix[lo]) / static_cast<double>(hi - lo + 1);
    }
    return out;
}

CornerResult detect_corners(std::span<const double> distance_m, std::span<const double> speed_kmh,
                            std::size_t smooth_window, double min_drop_kmh,
                            double min_separation_m) {
    if (distance_m.size() != speed_kmh.size()) {
        throw std::invalid_argument("distance and speed arrays must have the same shape");
    }
    if (distance_m.empty()) {
        throw std::invalid_argument("telemetry arrays must not be empty");
    }
    if (min_drop_kmh <= 0.0) {
        throw std::invalid_argument("min_drop_kmh must be positive");
    }
    if (min_separation_m < 0.0) {
        throw std::invalid_argument("min_separation_m must be non-negative");
    }

    CornerResult result;
    const std::size_t n = speed_kmh.size();
    if (n < 3) {
        return result;  // Too short to contain an interior local minimum.
    }
    // Step 1: smooth. Every decision below reads `s`, but the indices we
    // return address the caller's original arrays.
    const std::vector<double> s = smooth(speed_kmh, smooth_window);

    // Step 2: collect local minima as corner candidates. The asymmetry —
    // `<=` on the left, `<` on the right — makes a flat valley floor yield
    // exactly one candidate (its last sample) instead of one per sample.
    std::vector<std::size_t> candidates;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (s[i] <= s[i - 1] && s[i] < s[i + 1]) {
            candidates.push_back(i);
        }
    }

    // Step 3: keep only prominent candidates. Walk outwards from each one in
    // both directions: a rise of min_drop_kmh before any sample dips below
    // the candidate means this really is the bottom of its own corner. Hitting
    // a lower sample first means a deeper minimum nearby shadows it — that
    // deeper one is the real apex, and this candidate is just a wiggle on the
    // way into it.
    std::vector<std::size_t> accepted;
    for (const std::size_t i : candidates) {
        bool rises_left = false;
        // `k-- > 0` tests then decrements, so the body sees i-1 ... 0 and the
        // loop still terminates on unsigned k (a plain `k >= 0` never would).
        for (std::size_t k = i; k-- > 0;) {
            if (s[k] < s[i]) {
                break;
            }
            if (s[k] - s[i] >= min_drop_kmh) {
                rises_left = true;
                break;
            }
        }
        bool rises_right = false;
        for (std::size_t k = i + 1; k < n; ++k) {
            if (s[k] < s[i]) {
                break;
            }
            if (s[k] - s[i] >= min_drop_kmh) {
                rises_right = true;
                break;
            }
        }
        if (rises_left && rises_right) {
            accepted.push_back(i);
        }
    }

    // Step 4: collapse apexes closer than min_separation_m into the slower
    // one, so a chicane counts as a single corner. Candidates arrive in track
    // order, so comparing against the last kept apex is enough.
    std::vector<std::size_t> kept;
    for (const std::size_t i : accepted) {
        if (!kept.empty() && distance_m[i] - distance_m[kept.back()] < min_separation_m) {
            if (s[i] < s[kept.back()]) {
                kept.back() = i;
            }
        } else {
            kept.push_back(i);
        }
    }

    // Step 5: find where braking began. Walk back from the apex tracking the
    // highest speed seen so far; once the trace has fallen min_drop_kmh below
    // that peak we have crossed it and are descending into the previous
    // corner, so the peak itself is the braking point. Tracking a running
    // maximum (rather than stopping at the first sample that is not higher
    // than its neighbour) keeps this stable on a noisy approach.
    for (const std::size_t apex : kept) {
        std::size_t best = apex;
        std::size_t k = apex;
        while (k > 0) {
            --k;
            if (s[k] > s[best]) {
                best = k;
            }
            if (s[best] - s[k] >= min_drop_kmh) {
                break;
            }
        }
        result.apex_indices.push_back(static_cast<std::int64_t>(apex));
        result.brake_indices.push_back(static_cast<std::int64_t>(best));
    }
    return result;
}

}  // namespace f1lab
