#pragma once

#include "gbkfit/cuda/common.hpp"

namespace gbkfit::cuda {

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
};

} // namespace gbkfit::cuda
