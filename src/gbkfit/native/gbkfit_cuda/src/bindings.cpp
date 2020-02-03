
#include <pybind11/pybind11.h>

#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/gmodels.hpp"

using namespace gbkfit::cuda;

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m)
{
    py::class_<DModelDCube<float>>(m, "DModelDCubef32")
            .def(py::init<>())
            .def("downscale", &DModelDCube<float>::downscale)
            .def("moments", &DModelDCube<float>::moments);

    py::class_<GModelMCDisk<float>>(m, "GModelMCDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelMCDisk<float>::evaluate);

    py::class_<GModelSMDisk<float>>(m, "GModelSMDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelSMDisk<float>::evaluate);
}
