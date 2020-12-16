#pragma once

#include "gbkfit/host/common.hpp"

namespace gbkfit::host {

template<typename T>
struct Objective
{
public:

    Objective() {}

    ~Objective() {}

    void
    count_pixels(Ptr data, Ptr model, int size, Ptr counts) const;
};

} // namespace gbkfit::host
