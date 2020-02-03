#pragma once

#include <gbkfit/common.hpp>

namespace gbkfit { namespace cuda {

template<typename T>
struct DModelDCube
{
    void
    downscale(
            int scale_x, int scale_y, int scale_z,
            int offset_x, int offset_y, int offset_z,
            int src_size_x, int src_size_y, int src_size_z,
            int dst_size_x, int dst_size_y, int dst_size_z,
            CPtr src_cube, Ptr dst_cube) const
    {
    }

    void
    moments(int spat_size_x, int spat_size_y,
            int spec_size,
            T spec_step,
            T spec_zero,
            T nanval,
            CPtr scube,
            Ptr mmap0, Ptr mmap1, Ptr mmap2, Ptr mmap3, Ptr mmap4) const
    {
    }
};

}} // namespace gbkfit::cuda
