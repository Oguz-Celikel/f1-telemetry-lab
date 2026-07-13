// pybind11 bindings — exposes the C++ core as the Python module f1lab._native.
//
// float64 C-contiguous numpy arrays cross the boundary without copying
// (buffer protocol); anything else is force-cast once on the way in. The GIL
// is released while the C++ core runs, so other Python threads can proceed.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corners.hpp"
#include "delta.hpp"

namespace py = pybind11;

namespace {

// The array type accepted at the boundary. `c_style` demands a contiguous
// row-major buffer and `forcecast` converts anything else (a float32 array, a
// non-contiguous slice) exactly once on the way in — so a well-formed float64
// array costs nothing, and an awkward one still works.
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Borrow a numpy buffer as a span: no copy, no ownership. Safe because the
// argument keeps the array alive for the whole call.
std::span<const double> as_span(const DoubleArray& array) {
    if (array.ndim() != 1) {
        throw std::invalid_argument("expected a one-dimensional array");
    }
    return {array.data(), static_cast<std::size_t>(array.size())};
}

// Same, for the list-of-arrays arguments taken by the multi-lap functions.
std::vector<std::span<const double>> as_spans(const std::vector<DoubleArray>& arrays) {
    std::vector<std::span<const double>> spans;
    spans.reserve(arrays.size());
    for (const DoubleArray& array : arrays) {
        spans.push_back(as_span(array));
    }
    return spans;
}

// Hand a result vector to Python without copying it: the vector is moved to
// the heap, and the capsule — which numpy holds as the array's owner — frees
// it once the array's refcount reaches zero. Ownership of the buffer thus
// passes from C++ to Python's garbage collector.
template <typename T>
py::array_t<T> steal_into_array(std::vector<T>&& values, const std::vector<py::ssize_t>& shape) {
    auto* heap = new std::vector<T>(std::move(values));
    py::capsule owner(heap, [](void* ptr) { delete static_cast<std::vector<T>*>(ptr); });
    return py::array_t<T>(shape, heap->data(), owner);
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "f1lab native analysis core (C++/pybind11)";

    m.def(
        "delta_time",
        [](const DoubleArray& ref_distance_m, const DoubleArray& ref_time_s,
           const DoubleArray& other_distance_m, const DoubleArray& other_time_s) {
            const auto ref_d = as_span(ref_distance_m);
            const auto ref_t = as_span(ref_time_s);
            const auto oth_d = as_span(other_distance_m);
            const auto oth_t = as_span(other_time_s);
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = f1lab::delta_time(ref_d, ref_t, oth_d, oth_t);
            }
            return steal_into_array(std::move(out), {static_cast<py::ssize_t>(ref_d.size())});
        },
        py::arg("ref_distance_m"), py::arg("ref_time_s"), py::arg("other_distance_m"),
        py::arg("other_time_s"),
        "Time delta 'other - ref' resampled onto the reference distance grid.");

    m.def(
        "resample_times",
        [](const std::vector<DoubleArray>& distances, const std::vector<DoubleArray>& times,
           const DoubleArray& grid) {
            const auto dist_spans = as_spans(distances);
            const auto time_spans = as_spans(times);
            const auto grid_span = as_span(grid);
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = f1lab::resample_times(dist_spans, time_spans, grid_span);
            }
            return steal_into_array(std::move(out), {static_cast<py::ssize_t>(dist_spans.size()),
                                                     static_cast<py::ssize_t>(grid_span.size())});
        },
        py::arg("distances"), py::arg("times"), py::arg("grid"),
        "Resample n drivers' lap-time curves onto a shared distance grid -> (n, g).");

    m.def(
        "pairwise_delta_matrix",
        [](const std::vector<DoubleArray>& distances, const std::vector<DoubleArray>& times,
           const DoubleArray& grid) {
            const auto dist_spans = as_spans(distances);
            const auto time_spans = as_spans(times);
            const auto grid_span = as_span(grid);
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = f1lab::pairwise_delta_matrix(dist_spans, time_spans, grid_span);
            }
            const auto n = static_cast<py::ssize_t>(dist_spans.size());
            return steal_into_array(std::move(out),
                                    {n, n, static_cast<py::ssize_t>(grid_span.size())});
        },
        py::arg("distances"), py::arg("times"), py::arg("grid"),
        "Pairwise delta cube: out[i, j, k] = time_j(grid[k]) - time_i(grid[k]).");

    m.def(
        "detect_corners",
        [](const DoubleArray& distance_m, const DoubleArray& speed_kmh, std::size_t smooth_window,
           double min_drop_kmh, double min_separation_m) {
            const auto dist = as_span(distance_m);
            const auto speed = as_span(speed_kmh);
            f1lab::CornerResult result;
            {
                py::gil_scoped_release release;
                result = f1lab::detect_corners(dist, speed, smooth_window, min_drop_kmh,
                                               min_separation_m);
            }
            const auto count = static_cast<py::ssize_t>(result.apex_indices.size());
            return py::make_tuple(steal_into_array(std::move(result.apex_indices), {count}),
                                  steal_into_array(std::move(result.brake_indices), {count}));
        },
        py::arg("distance_m"), py::arg("speed_kmh"), py::arg("smooth_window"),
        py::arg("min_drop_kmh"), py::arg("min_separation_m"),
        "Detect corners in a speed trace -> (apex_indices, brake_indices).");
}
