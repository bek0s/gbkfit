#pragma once

#include "gbkfit/cuda/common.hpp"

namespace gbkfit::cuda {

template<typename T>
struct Objective
{
public:

    Objective() {}

    ~Objective() {}

    void
    count_pixels(Ptr data, Ptr model, int size, Ptr counts) const;
};

} // namespace gbkfit::cuda
