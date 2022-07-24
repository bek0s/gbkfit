#pragma once

#include "gbkfit/cuda/common.hpp"

namespace gbkfit::cuda {

template<typename T>
struct Objective
{
    void
    count_pixels(Ptr data1, Ptr data2, int size, T epsilon, Ptr counts) const;
};

} // namespace gbkfit::cuda
