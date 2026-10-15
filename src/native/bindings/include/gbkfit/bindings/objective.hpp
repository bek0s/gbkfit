#pragma once

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/bindings/gil.hpp"

namespace gbkfit::bindings {

// The objective operations of a native module. Kernels launches the
// kernels of the module, on the memory of Device: each function on the
// device of the array it writes (see DeviceScope of the Kernels). The
// arrays can have any shape; they are processed element by element.
template<typename T, typename Device, typename Kernels>
struct Objective
{
    using Data = Array<Device, T>;
    using ConstData = Array<Device, const T>;
    // A sum, in double precision: a cube can have millions of terms
    using Sum = Array<Device, double, nb::shape<1>>;
    // Sums of each spectrum, in double precision
    using Sums = Array<Device, double>;

    // (mdl_d - obs_d) / obs_e * mdl_w * obs_m * weight, where all arrays
    // but obs_d and mdl_d are optional; 0 where the data are masked (or,
    // with a limit, not NaN, that of an upper limit of noise
    // limit_error), and missing * weight where the model has no value
    // (NaN) and the data have one (see gbkfit::objective_residual)
    static void
    residual(
            ConstData obs_d, ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w,
            T weight, T missing, T limit, T limit_error, Data res)
    {
        const typename Kernels::DeviceScope scope(res.device_id());
        require_same_shape(obs_e, obs_d, "obs_e", "obs_d");
        require_same_shape(obs_m, obs_d, "obs_m", "obs_d");
        require_same_shape(mdl_d, obs_d, "mdl_d", "obs_d");
        require_same_shape(mdl_w, obs_d, "mdl_w", "obs_d");
        require_same_shape(res, obs_d, "res", "obs_d");
        Kernels::objective_residual(
                obs_d.data(), data(obs_e), data(obs_m),
                mdl_d.data(), data(mdl_w),
                int(obs_d.size()), weight, missing, limit, limit_error,
                res.data());
    }

    // The sum of the squared or absolute residuals
    static void
    residual_sum(ConstData residual, bool squared, Sum sum)
    {
        const typename Kernels::DeviceScope scope(sum.device_id());
        Kernels::objective_residual_sum(
                residual.data(), int(residual.size()), squared, sum.data());
    }

    // The sums of each spectrum (the values along the first axis, at each
    // point of the others) that its amplitude takes, into a and b
    // (of the shape of the other axes; see
    // gbkfit::objective_amplitude_sums)
    static void
    amplitude_sums(
            ConstData obs_d, ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w, Sums a, Sums b)
    {
        const typename Kernels::DeviceScope scope(a.device_id());
        require_same_shape(obs_e, obs_d, "obs_e", "obs_d");
        require_same_shape(obs_m, obs_d, "obs_m", "obs_d");
        require_same_shape(mdl_d, obs_d, "mdl_d", "obs_d");
        require_same_shape(mdl_w, obs_d, "mdl_w", "obs_d");
        require(obs_d.ndim() >= 1 && obs_d.shape(0) > 0,
                "obs_d must have a first axis");
        const int nchannels = int(obs_d.shape(0));
        const int nspectra = int(obs_d.size() / obs_d.shape(0));
        require(a.size() == size_t(nspectra) && b.size() == size_t(nspectra),
                "a and b must have one value per spectrum");
        Kernels::objective_amplitude_sums(
                obs_d.data(), data(obs_e), data(obs_m),
                mdl_d.data(), data(mdl_w),
                nchannels, nspectra, a.data(), b.data());
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<Objective>(m, ("Objective" + suffix).c_str())
                .def(nb::init<>())
                .def_static("residual", &residual,
                        ReleaseGil(),
                        nb::arg("obs_d").noconvert(),
                        nb::arg("obs_e").noconvert().none(),
                        nb::arg("obs_m").noconvert().none(),
                        nb::arg("mdl_d").noconvert(),
                        nb::arg("mdl_w").noconvert().none(),
                        nb::arg("weight"),
                        nb::arg("missing"),
                        nb::arg("limit"),
                        nb::arg("limit_error"),
                        nb::arg("res").noconvert())
                .def_static("residual_sum", &residual_sum,
                        ReleaseGil(),
                        nb::arg("residual").noconvert(),
                        nb::arg("squared"),
                        nb::arg("sum").noconvert())
                .def_static("amplitude_sums", &amplitude_sums,
                        ReleaseGil(),
                        nb::arg("obs_d").noconvert(),
                        nb::arg("obs_e").noconvert().none(),
                        nb::arg("obs_m").noconvert().none(),
                        nb::arg("mdl_d").noconvert(),
                        nb::arg("mdl_w").noconvert().none(),
                        nb::arg("a").noconvert(),
                        nb::arg("b").noconvert());
    }
};

} // namespace gbkfit::bindings
