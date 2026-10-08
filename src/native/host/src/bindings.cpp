
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/bindings/dmodel.hpp"
#include "gbkfit/bindings/objective.hpp"
#include "gbkfit/host/fft.hpp"
#include "gbkfit/host/gmodels.hpp"
#include "gbkfit/host/kernels.hpp"

using namespace gbkfit;

namespace nb = nanobind;

NB_MODULE(EXTENSION_NAME, m)
{
    using Device = nb::device::cpu;
    using Kernels = host::Wrapper<float>;

    bindings::DModel<float, Device, Kernels>::bind(m, "DModelf32");
    bindings::Objective<float, Device, Kernels>::bind(m, "Objectivef32");
    host::FFT<float>::bind(m, "FFTf32");

    nb::class_<host::GModel<float>>(m, "GModelf32")
            .def(nb::init<>())
            .def("convolve", &host::GModel<float>::convolve)
            .def("wcube_evaluate", &host::GModel<float>::wcube_evaluate)
            .def("mcdisk_evaluate", &host::GModel<float>::mcdisk_evaluate)
            .def("smdisk_evaluate", &host::GModel<float>::smdisk_evaluate);
}
