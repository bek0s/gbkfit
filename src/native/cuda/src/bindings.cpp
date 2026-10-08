
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/bindings/dmodel.hpp"
#include "gbkfit/bindings/gmodel.hpp"
#include "gbkfit/bindings/objective.hpp"
#include "gbkfit/cuda/wrapper.hpp"

using namespace gbkfit;

namespace nb = nanobind;

NB_MODULE(EXTENSION_NAME, m)
{
    using Device = nb::device::cuda;
    using Kernels = cuda::Wrapper<float>;

    bindings::DModel<float, Device, Kernels>::bind(m, "f32");
    bindings::GModel<float, Device, Kernels>::bind(m, "f32");
    bindings::Objective<float, Device, Kernels>::bind(m, "f32");
}
