
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/host/dmodels.hpp"
#include "gbkfit/host/fft.hpp"
#include "gbkfit/host/gmodels.hpp"
#include "gbkfit/host/objective.hpp"

using namespace gbkfit::host;

namespace nb = nanobind;

NB_MODULE(EXTENSION_NAME, m)
{
    nb::class_<DModel<float>>(m, "DModelf32")
            .def(nb::init<>())
            .def("dcube_downscale", &DModel<float>::dcube_downscale)
            .def("dcube_mask", &DModel<float>::dcube_mask)
            .def("mmaps_moments", &DModel<float>::mmaps_moments);

    nb::class_<GModel<float>>(m, "GModelf32")
            .def(nb::init<>())
            .def("convolve", &GModel<float>::convolve)
            .def("wcube_evaluate", &GModel<float>::wcube_evaluate)
            .def("mcdisk_evaluate", &GModel<float>::mcdisk_evaluate)
            .def("smdisk_evaluate", &GModel<float>::smdisk_evaluate);

    nb::class_<Objective<float>>(m, "Objectivef32")
            .def(nb::init<>())
            .def("count_pixels", &Objective<float>::count_pixels)
            .def("residual", &Objective<float>::residual)
            .def("residual_sum", &Objective<float>::residual_sum);

    nb::class_<FFT<float>>(m, "FFTf32")
            .def(nb::init<>())
            .def("fft_r2c", &FFT<float>::fft_r2c)
            .def("fft_c2r", &FFT<float>::fft_c2r)
            .def("fft_convolve", &FFT<float>::fft_convolve)
            .def("fft_convolve_cached", &FFT<float>::fft_convolve_cached);
}
