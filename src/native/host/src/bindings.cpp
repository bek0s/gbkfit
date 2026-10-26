
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include "gbkfit/bindings/dmodel.hpp"
#include "gbkfit/bindings/gmodel.hpp"
#include "gbkfit/bindings/objective.hpp"
#include "gbkfit/bindings/orbits.hpp"
#include "gbkfit/host/fft.hpp"
#include "gbkfit/host/kernels.hpp"

using namespace gbkfit;

namespace nb = nanobind;

template<typename T, typename Device> void
bind_precision(nb::module_& m, const std::string& suffix)
{
    using Kernels = host::Wrapper<T>;
    bindings::DModel<T, Device, Kernels>::bind(m, suffix);
    bindings::GModel<T, Device, Kernels>::bind(m, suffix);
    bindings::Objective<T, Device, Kernels>::bind(m, suffix);
    bindings::Orbits<T, Device, Kernels>::bind(m, suffix);
    host::FFTConvolution<T, T>::bind(m, "FFTConvolution" + suffix);
}

NB_MODULE(EXTENSION_NAME, m)
{
    using Device = nb::device::cpu;

    // The classes of each precision: float32 (suffix f32) and float64
    // (suffix f64)
    bind_precision<float, Device>(m, "f32");
    bind_precision<double, Device>(m, "f64");

    // The convolutions of float32 cubes by float64 transforms
    host::FFTConvolution<float, double>::bind(m, "FFTConvolutionf32f64");
}
