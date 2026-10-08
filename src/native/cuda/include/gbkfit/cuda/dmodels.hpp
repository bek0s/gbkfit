#pragma once

#include "gbkfit/cuda/common.hpp"

namespace gbkfit::cuda {

template<typename T>
struct DModel
{
    void
    dcube_downscale(
            std::array<int, 3> scale,
            std::array<int, 3> offset,
            std::array<int, 3> src_size,
            std::array<int, 3> dst_size,
            Ptr src_dcube, Ptr dst_dcube) const;

    void
    dcube_mask(
            T cutoff, bool apply,
            std::array<int, 3> size,
            Ptr dcube_d, Ptr dcube_m, Ptr dcube_w) const;

    void
    mmaps_moments(
            std::array<int, 3> size,
            std::array<T, 3> step,
            std::array<T, 3> zero,
            Ptr dcube_d, Ptr dcube_w,
            T cutoff, int norders, Ptr orders,
            Ptr mmaps_d, Ptr mmaps_m, Ptr mmaps_w) const;
};

} // namespace gbkfit::cuda
