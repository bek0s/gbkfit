
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/gmodels.hpp"
#include "gbkfit/cuda/objective.hpp"

using namespace gbkfit::cuda;

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m)
{
    py::class_<DModelDCube<float>>(m, "DModelDCubef32")
            .def(py::init<>())
            .def("convolve", &DModelDCube<float>::convolve)
            .def("downscale", &DModelDCube<float>::downscale)
            .def("make_mask", &DModelDCube<float>::make_mask);

    py::class_<DModelMMaps<float>>(m, "DModelMMapsf32")
            .def(py::init<>())
            .def("moments", &DModelMMaps<float>::moments);

    py::class_<GModel<float>>(m, "GModelf32")
            .def(py::init<>())
            .def("make_wcube", &GModel<float>::make_wcube);

    py::class_<GModelMCDisk<float>>(m, "GModelMCDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelMCDisk<float>::evaluate);

    py::class_<GModelSMDisk<float>>(m, "GModelSMDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelSMDisk<float>::evaluate);

    py::class_<Objective<float>>(m, "Objectivef32")
            .def(py::init<>())
            .def("count_pixels", &Objective<float>::count_pixels);
}
