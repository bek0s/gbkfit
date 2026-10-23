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

    // (mdl_d - obs_d) / noise * mdl_w * obs_m * weight, the noise that of
    // the error obs_e (1 without), sqrt((scale obs_e)^2 + jitter^2), where
    // all arrays but obs_d and mdl_d are optional; 0 where the data are
    // masked (or, with a limit, not NaN, that of an upper limit with the
    // noise of the error limit_error), and missing * weight where the
    // model has no value (NaN) and the data have one (see
    // gbkfit::objective_residual)
    static void
    residual(
            ConstData obs_d, ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w,
            T weight, T missing, T limit, T limit_error,
            T scale, T jitter, Data res)
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
                scale, jitter, res.data());
    }

    // The sum of the squared or absolute residuals
    static void
    residual_sum(ConstData residual, bool squared, Sum sum)
    {
        const typename Kernels::DeviceScope scope(sum.device_id());
        Kernels::objective_residual_sum(
                residual.data(), int(residual.size()), squared, sum.data());
    }

    // The sums of each spectrum (the values along the axis, at each
    // point of the others) that its amplitude takes, into a and b
    // (of one value per point of the other axes; see
    // gbkfit::objective_amplitude_terms), each value with its noise (as
    // in residual)
    static void
    amplitude_sums(
            ConstData obs_d, ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w, T scale, T jitter,
            int axis, Sums a, Sums b)
    {
        const typename Kernels::DeviceScope scope(a.device_id());
        require_same_shape(obs_e, obs_d, "obs_e", "obs_d");
        require_same_shape(obs_m, obs_d, "obs_m", "obs_d");
        require_same_shape(mdl_d, obs_d, "mdl_d", "obs_d");
        require_same_shape(mdl_w, obs_d, "mdl_w", "obs_d");
        require(axis >= 0 && size_t(axis) < obs_d.ndim()
                && obs_d.shape(axis) > 0,
                "obs_d must have the axis of the spectra");
        size_t nouter = 1, ninner = 1;
        for (size_t k = 0; k < obs_d.ndim(); ++k) {
            if (k < size_t(axis)) {
                nouter *= obs_d.shape(k);
            } else if (k > size_t(axis)) {
                ninner *= obs_d.shape(k);
            }
        }
        require(a.size() == nouter * ninner && b.size() == nouter * ninner,
                "a and b must have one value per spectrum");
        Kernels::objective_amplitude_sums(
                obs_d.data(), data(obs_e), data(obs_m),
                mdl_d.data(), data(mdl_w),
                scale, jitter, int(nouter), int(obs_d.shape(axis)),
                int(ninner), a.data(), b.data());
    }

    // The square of the noise residual of a data item whose noise is
    // fitted, with the noise and the weight of residual (see
    // gbkfit::objective_noise_term), into sum
    static void
    noise_sum(
            ConstData obs_e, ConstData obs_m,
            ConstData mdl_d, ConstData mdl_w,
            T weight, T scale, T jitter, T scale_minimum, Sum sum)
    {
        const typename Kernels::DeviceScope scope(sum.device_id());
        require_same_shape(obs_e, mdl_d, "obs_e", "mdl_d");
        require_same_shape(obs_m, mdl_d, "obs_m", "mdl_d");
        require_same_shape(mdl_w, mdl_d, "mdl_w", "mdl_d");
        Kernels::objective_noise_sum(
                data(obs_e), data(obs_m),
                mdl_d.data(), data(mdl_w),
                int(mdl_d.size()), weight, scale, jitter, scale_minimum,
                sum.data());
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
                        nb::arg("scale"),
                        nb::arg("jitter"),
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
                        nb::arg("scale"),
                        nb::arg("jitter"),
                        nb::arg("axis"),
                        nb::arg("a").noconvert(),
                        nb::arg("b").noconvert())
                .def_static("noise_sum", &noise_sum,
                        ReleaseGil(),
                        nb::arg("obs_e").noconvert().none(),
                        nb::arg("obs_m").noconvert().none(),
                        nb::arg("mdl_d").noconvert(),
                        nb::arg("mdl_w").noconvert().none(),
                        nb::arg("weight"),
                        nb::arg("scale"),
                        nb::arg("jitter"),
                        nb::arg("scale_minimum"),
                        nb::arg("sum").noconvert());
    }
};

} // namespace gbkfit::bindings
