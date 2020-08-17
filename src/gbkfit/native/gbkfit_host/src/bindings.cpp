
#include <pybind11/pybind11.h>

#include "gbkfit/host/dmodels.hpp"
#include "gbkfit/host/gmodels.hpp"

using namespace gbkfit::host;

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m)
{
    py::class_<DModelDCube<float>>(m, "DModelDCubef32")
            .def(py::init<>())
            .def("prepare", &DModelDCube<float>::prepare)
            .def("convolve", &DModelDCube<float>::convolve)
            .def("downscale", &DModelDCube<float>::downscale)
            .def("make_mask", &DModelDCube<float>::make_mask)
            .def("apply_mask", &DModelDCube<float>::apply_mask);

    py::class_<DModelMMaps<float>>(m, "DModelMMapsf32")
            .def(py::init<>())
            .def("prepare", &DModelMMaps<float>::prepare)
            .def("moments", &DModelMMaps<float>::moments);

    py::class_<GModelMCDisk<float>>(m, "GModelMCDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelMCDisk<float>::evaluate);

    py::class_<GModelSMDisk<float>>(m, "GModelSMDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelSMDisk<float>::evaluate);
}
