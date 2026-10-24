#pragma once

#include <cmath>

#include "gbkfit/math/math.hpp"
#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

// The residual of a model value m where the data are below a detection
// limit (an upper limit), with the error (noise) of the data there: the
// data are censored, and -2 log Phi((limit - m) / error) is the
// chi-square of the probability that they are below the limit (in double
// precision; 0 where the model has no value). It is about 0 for a model
// far below the limit, and about (m - limit) / error far above it.
template<typename T> constexpr T
objective_upper_limit_residual(T model, T limit, T error)
{
    if (std::isnan(model))
        return 0;
    const double z = (double(limit) - double(model)) / double(error);
    return T(std::sqrt(-2 * log_ndtr(z)));
}

// The noise of an error e: e times the scale of the noise, and its jitter,
// in quadrature, sqrt((scale e)^2 + jitter^2) (see LikelihoodGaussian).
// (hypot: errors of tiny numbers, e.g. 1e-20 erg/s/cm^2/A, do not
// underflow when squared, and the noise of a scale of 1 and no jitter is
// the error itself)
template<typename T> constexpr T
objective_noise(T error, T scale, T jitter)
{
    return std::hypot(scale * error, jitter);
}

// The weighted residual of value i: (mdl_d - obs_d) / noise * mdl_w *
// obs_m * weight, the noise that of the error obs_e (1 without errors;
// see objective_noise). A value the data mask (obs_m = 0) has none (0),
// unless a limit is given (not NaN): the data are then below it there,
// with the noise of the error limit_error (see
// objective_upper_limit_residual). Where the data have a value but the
// model none (NaN, e.g. the velocity of a spectrum without flux), it is
// 'missing' times the weight: otherwise a model could lower the
// chi-square by having no values.
template<typename T> constexpr void
objective_residual(
        int i,
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T missing, T limit, T limit_error,
        T scale, T jitter, T* res)
{
    // (the data are NaN where masked, and NaN times 0 is NaN)
    if (obs_m && obs_m[i] == 0) {
        if (std::isnan(limit)) {
            res[i] = 0;
            return;
        }
        T residual = objective_upper_limit_residual(
                mdl_d[i], limit, objective_noise(limit_error, scale, jitter));
        if (mdl_w) {
            residual *= mdl_w[i];
        }
        res[i] = residual * weight;
        return;
    }

    if (std::isnan(mdl_d[i])) {
        res[i] = missing * weight;
        return;
    }

    T residual = mdl_d[i] - obs_d[i];

    residual /= objective_noise(obs_e ? obs_e[i] : T(1), scale, jitter);

    if (mdl_w) {
        residual *= mdl_w[i];
    }

    if (obs_m) {
        residual *= obs_m[i];
    }

    res[i] = residual * weight;
}

// Add value i of a spectrum to the sums that its amplitude takes
// (see Objective.amplitude_sums): a = sum W mdl_d^2 and b = sum W mdl_d
// obs_d, W = (obs_m * mdl_w / noise)^2, the noise that of the error obs_e
// (see objective_residual), over its values that the data do not mask
// (obs_m = 0) and that the model has (not NaN). The terms are in the
// precision of the data, as the residuals (double precision is slow on
// many GPUs); the sums in double precision.
template<typename T> constexpr void
objective_amplitude_terms(
        long long i,
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        T scale, T jitter, double& a, double& b)
{
    if ((obs_m && obs_m[i] == 0) || std::isnan(mdl_d[i])) {
        return;
    }
    T w = T(1) / objective_noise(obs_e ? obs_e[i] : T(1), scale, jitter);
    if (mdl_w) {
        w *= mdl_w[i];
    }
    if (obs_m) {
        w *= obs_m[i];
    }
    const T wm = w * mdl_d[i];
    a += wm * wm;
    b += wm * w * obs_d[i];
}

// The term of value i in the noise residual of a data item whose noise is
// fitted (see LikelihoodGaussian): c^2 log(noise^2 / (scale_minimum
// error)^2), c the factor of its residual (see objective_residual: the
// weight, times mdl_w and obs_m), and error that of obs_e (1 without
// errors). It is the share of the value in the normalisation of the
// Gaussian, c^2 log(noise^2), less a part that the noise does not change,
// so that it is never negative (the scale is at least scale_minimum). The
// values the data mask have none (an upper limit is a probability, which
// has no normalisation), nor those the model lacks (their residual,
// 'missing', does not take the noise).
template<typename T> constexpr double
objective_noise_term(
        long long i,
        const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        T weight, T scale, T jitter, T scale_minimum)
{
    if ((obs_m && obs_m[i] == 0) || std::isnan(mdl_d[i])) {
        return 0;
    }
    double c = weight;
    if (mdl_w) {
        c *= mdl_w[i];
    }
    if (obs_m) {
        c *= obs_m[i];
    }
    // (noise / (scale_minimum error), in double precision)
    const double error = obs_e ? double(obs_e[i]) : 1.0;
    const double ratio = std::hypot(double(scale), double(jitter) / error)
            / double(scale_minimum);
    return c * c * 2 * std::log(ratio);
}

} // namespace gbkfit
