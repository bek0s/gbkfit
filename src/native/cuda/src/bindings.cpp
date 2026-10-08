
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/bindings/dmodel.hpp"
#include "gbkfit/bindings/objective.hpp"
#include "gbkfit/cuda/gmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

using namespace gbkfit;

namespace nb = nanobind;

NB_MODULE(EXTENSION_NAME, m)
{
    using Device = nb::device::cuda;
    using Kernels = cuda::Wrapper<float>;

    bindings::DModel<float, Device, Kernels>::bind(m, "DModelf32");
    bindings::Objective<float, Device, Kernels>::bind(m, "Objectivef32");

    nb::class_<cuda::GModel<float>>(m, "GModelf32")
            .def(nb::init<>())
            .def("wcube_evaluate", &cuda::GModel<float>::wcube_evaluate)
            .def("mcdisk_evaluate", &cuda::GModel<float>::mcdisk_evaluate)
            .def("smdisk_evaluate", &cuda::GModel<float>::smdisk_evaluate);
}
