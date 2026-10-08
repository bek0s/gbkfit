
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/gmodels.hpp"
#include "gbkfit/cuda/objective.hpp"

using namespace gbkfit::cuda;

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
            .def("wcube_evaluate", &GModel<float>::wcube_evaluate)
            .def("mcdisk_evaluate", &GModel<float>::mcdisk_evaluate)
            .def("smdisk_evaluate", &GModel<float>::smdisk_evaluate);

    nb::class_<Objective<float>>(m, "Objectivef32")
            .def(nb::init<>())
            .def("count_pixels", &Objective<float>::count_pixels)
            .def("residual", &Objective<float>::residual);
}
