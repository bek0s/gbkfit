#pragma once

#include <array>
#include <stdexcept>
#include <string>

#include <nanobind/ndarray.h>

// Helpers for the arrays that the native modules receive from Python.
// The arrays are NumPy arrays for the host module and CuPy arrays for
// the cuda module; nanobind checks their dtype, device and layout.
namespace gbkfit::bindings {

namespace nb = nanobind;

// A C-contiguous array in the memory of Device (nb::device::cpu or
// nb::device::cuda), optionally with more constraints (e.g. nb::ndim)
template<typename Device, typename T, typename... Constraints>
using Array = nb::ndarray<T, Device, nb::c_contig, Constraints...>;

// The data of an array, or nullptr for an optional array given as None
template<typename A> auto
data(const A& array)
{
    return array.is_valid() ? array.data() : nullptr;
}

// The (x, y, z) size of a three-dimensional array of shape (z, y, x)
template<typename A> std::array<int, 3>
size_xyz(const A& array)
{
    return {int(array.shape(2)), int(array.shape(1)), int(array.shape(0))};
}

// Raise a ValueError in Python if a condition is not met
inline void
require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

// Require an optional array (if given) to have the shape of another
template<typename A, typename B> void
require_same_shape(
        const A& array, const B& reference,
        const std::string& name, const std::string& reference_name)
{
    if (!array.is_valid()) {
        return;
    }
    bool same = array.ndim() == reference.ndim();
    for (size_t i = 0; same && i < array.ndim(); ++i) {
        same = array.shape(i) == reference.shape(i);
    }
    require(same, name + " must have the shape of " + reference_name);
}

} // namespace gbkfit::bindings
