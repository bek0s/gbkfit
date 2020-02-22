#pragma once

#include <gbkfit/common.hpp>

namespace gbkfit { namespace cuda {

template<typename T>
struct DModelDCube
{
public:

    DModelDCube(void){}

    ~DModelDCube() {}

    void
    prepare(void) {}

    void
    cleanup(void) {}

    void
    convolve(void) const {}

    void
    downscale(void) const {}
};

template<typename T>
struct DModelMMaps
{
public:

    DModelMMaps(void) {}

    ~DModelMMaps() {}

    void
    prepare(void) {}

    void
    moments(void) const;
};

}} // namespace gbkfit::cuda
