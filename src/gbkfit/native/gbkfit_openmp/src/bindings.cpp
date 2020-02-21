
#include <pybind11/pybind11.h>

#include "gbkfit/openmp/dmodels.hpp"
#include "gbkfit/openmp/gmodels.hpp"

using namespace gbkfit;
using namespace gbkfit::openmp;

namespace py = pybind11;

PYBIND11_MODULE(EXTENSION_NAME, m)
{
    py::class_<DModelDCube<float>>(m, "DModelDCubef32")
            .def(py::init<
                 int, int, int,
                 int, int, int,
                 int, int, int,
                 int, int, int,
                 Ptr, Ptr, Ptr, Ptr, Ptr>())
            .def("downscale", &DModelDCube<float>::downscale)
            .def("convolve", &DModelDCube<float>::convolve);

    py::class_<DModelMMaps<float>>(m, "DModelMMapsf32")
            .def(py::init<>())
            .def("moments", &DModelMMaps<float>::moments);

    py::class_<GModelMCDisk<float>>(m, "GModelMCDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelMCDisk<float>::evaluate);

    py::class_<GModelSMDisk<float>>(m, "GModelSMDiskf32")
            .def(py::init<>())
            .def("evaluate", &GModelSMDisk<float>::evaluate);
}
