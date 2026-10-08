#pragma once

#include "gbkfit/host/common.hpp"

namespace gbkfit::host {

template<typename T>
struct Objective
{
    void
    count_pixels(Ptr data1, Ptr data2, int size, T epsilon, Ptr counts) const;

    void
    residual(
            Ptr obs_d, Ptr obs_e, Ptr obs_m,
            Ptr mdl_d, Ptr mdl_w, Ptr mdl_m,
            int size, T weight, Ptr residual) const;

    void
    residual_sum(Ptr residual, int size, bool squared, Ptr sum) const;
};

} // namespace gbkfit::host
