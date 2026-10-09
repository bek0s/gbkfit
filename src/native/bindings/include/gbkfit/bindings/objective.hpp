#pragma once

#include "gbkfit/bindings/arrays.hpp"

namespace gbkfit::bindings {

// The objective operations of a native module. Kernels launches the
// kernels of the module, on the memory of Device. The arrays can have
// any shape; they are processed element by element.
template<typename T, typename Device, typename Kernels>
struct Objective
{
    using Data = Array<Device, T>;
    using ConstData = Array<Device, const T>;
    // A sum, in double precision: a cube can have millions of terms
    using Sum = Array<Device, double, nb::shape<1>>;

    // (mdl_d - obs_d) / obs_e * mdl_w * obs_m * mdl_m * weight, where
    // all arrays but obs_d and mdl_d are optional
    static void
    residual(
            ConstData obs_d, ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w, ConstData mdl_m,
            T weight, Data res)
    {
        require_same_shape(obs_e, obs_d, "obs_e", "obs_d");
        require_same_shape(obs_m, obs_d, "obs_m", "obs_d");
        require_same_shape(mdl_d, obs_d, "mdl_d", "obs_d");
        require_same_shape(mdl_w, obs_d, "mdl_w", "obs_d");
        require_same_shape(mdl_m, obs_d, "mdl_m", "obs_d");
        require_same_shape(res, obs_d, "res", "obs_d");
        Kernels::objective_residual(
                obs_d.data(), data(obs_e), data(obs_m),
                mdl_d.data(), data(mdl_w), data(mdl_m),
                int(obs_d.size()), weight, res.data());
    }

    // The sum of the squared or absolute residuals
    static void
    residual_sum(ConstData residual, bool squared, Sum sum)
    {
        Kernels::objective_residual_sum(
                residual.data(), int(residual.size()), squared, sum.data());
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<Objective>(m, ("Objective" + suffix).c_str())
                .def(nb::init<>())
                .def_static("residual", &residual,
                        nb::arg("obs_d").noconvert(),
                        nb::arg("obs_e").noconvert().none(),
                        nb::arg("obs_m").noconvert().none(),
                        nb::arg("mdl_d").noconvert(),
                        nb::arg("mdl_w").noconvert().none(),
                        nb::arg("mdl_m").noconvert().none(),
                        nb::arg("weight"),
                        nb::arg("res").noconvert())
                .def_static("residual_sum", &residual_sum,
                        nb::arg("residual").noconvert(),
                        nb::arg("squared"),
                        nb::arg("sum").noconvert());
    }
};

} // namespace gbkfit::bindings
