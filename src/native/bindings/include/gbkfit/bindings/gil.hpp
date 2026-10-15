#pragma once

#include <nanobind/nanobind.h>

namespace gbkfit::bindings {

namespace nb = nanobind;

// The functions that run kernels release the GIL while they run (they
// use no Python objects), so that other Python threads run meanwhile:
// e.g. those of evaluations on other devices
using ReleaseGil = nb::call_guard<nb::gil_scoped_release>;

} // namespace gbkfit::bindings
