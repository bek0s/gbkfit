#pragma once

#include <cmath>

#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

// The weighted residual of value i: (mdl_d - obs_d) / obs_e * mdl_w *
// obs_m * weight. A value the data mask (obs_m = 0) has none (0). Where
// the data have a value but the model none (NaN, e.g. the velocity of a
// spectrum without flux), it is 'missing' times the weight: otherwise a
// model could lower the chi-square by having no values.
template<typename T> constexpr void
objective_residual(
        int i,
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T missing, T* res)
{
    // (the data are NaN where masked, and NaN times 0 is NaN)
    if (obs_m && obs_m[i] == 0) {
        res[i] = 0;
        return;
    }

    if (std::isnan(mdl_d[i])) {
        res[i] = missing * weight;
        return;
    }

    T residual = mdl_d[i] - obs_d[i];

    if (obs_e) {
        residual /= obs_e[i];
    }

    if (mdl_w) {
        residual *= mdl_w[i];
    }

    if (obs_m) {
        residual *= obs_m[i];
    }

    res[i] = residual * weight;
}

// The sums of spectrum s (of nspectra; its value of channel c at c *
// nspectra + s) that its free amplitude takes (see
// Objective.amplitude_sums): a = sum W mdl_d^2, b = sum W mdl_d obs_d and
// f = sum mdl_d, W = (obs_m * mdl_w / obs_e)^2, over its values that the
// data do not mask (obs_m = 0) and that the model has (not NaN); in
// double precision
template<typename T> constexpr void
objective_amplitude_sums(
        int s,
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int nchannels, int nspectra, double* a, double* b, double* f)
{
    double sum_a = 0, sum_b = 0, sum_f = 0;
    for (int c = 0; c < nchannels; ++c)
    {
        const long long i = (long long)c * nspectra + s;
        if ((obs_m && obs_m[i] == 0) || std::isnan(mdl_d[i])) {
            continue;
        }
        double w = 1;
        if (obs_e) {
            w /= obs_e[i];
        }
        if (mdl_w) {
            w *= mdl_w[i];
        }
        if (obs_m) {
            w *= obs_m[i];
        }
        const double m = mdl_d[i];
        sum_a += w * w * m * m;
        sum_b += w * w * m * obs_d[i];
        sum_f += m;
    }
    a[s] = sum_a;
    b[s] = sum_b;
    f[s] = sum_f;
}

} // namespace gbkfit
