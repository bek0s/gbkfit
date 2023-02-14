#pragma once

#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

template<typename T> constexpr void
objective_residual(
        int i,
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w, const T* mdl_m,
        int size, T weight, T* res)
{
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

    if (mdl_m) {
        residual *= mdl_m[i];
    }

    res[i] = residual * weight;
}

} // namespace gbkfit
