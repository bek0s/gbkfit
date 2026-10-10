
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/bindings/dmodel.hpp"
#include "gbkfit/bindings/gmodel.hpp"
#include "gbkfit/bindings/objective.hpp"
#include "gbkfit/cuda/wrapper.hpp"

using namespace gbkfit;

namespace nb = nanobind;

template<typename T> void
bind_precision(nb::module_& m, const std::string& suffix)
{
    using Device = nb::device::cuda;
    using Kernels = cuda::Wrapper<T>;
    bindings::DModel<T, Device, Kernels>::bind(m, suffix);
    bindings::GModel<T, Device, Kernels>::bind(m, suffix);
    bindings::Objective<T, Device, Kernels>::bind(m, suffix);
}

NB_MODULE(EXTENSION_NAME, m)
{
    // The classes of each precision: float32 (suffix f32) and float64
    // (suffix f64)
    bind_precision<float>(m, "f32");
    bind_precision<double>(m, "f64");
}
