#pragma once

#include "gbkfit/host/common.hpp"

namespace gbkfit::host {

template<typename T>
struct Objective
{
    void
    count_pixels(Ptr data1, Ptr data2, int size, T epsilon, Ptr counts) const;
};

} // namespace gbkfit::host
