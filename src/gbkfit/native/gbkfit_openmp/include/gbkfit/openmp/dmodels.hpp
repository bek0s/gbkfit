#pragma once

#include <gbkfit/common.hpp>

#include "kernels_main.hpp"

namespace gbkfit { namespace openmp {

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
        kernels::dcube_downscale(
                scale_x, scale_y, scale_z,
                offset_x, offset_y, offset_z,
                src_size_x, src_size_y, src_size_z,
                dst_size_x, dst_size_y, dst_size_z,
                reinterpret_cast<const T*>(src_cube),
                reinterpret_cast<T*>(dst_cube));
    }
};

template<typename T>
struct DModelMMaps
{
    void
    moments(int spat_size_x, int spat_size_y,
            int spec_size,
            T spec_step,
            T spec_zero,
            T nanval,
            CPtr scube,
            int mcount, CPtr morders, Ptr mmaps) const
    {
        kernels::dcube_moments(
                spat_size_x, spat_size_y,
                spec_size,
                spec_step,
                spec_zero,
                nanval,
                reinterpret_cast<const T*>(scube),
                mcount,
                reinterpret_cast<const int*>(morders),
                reinterpret_cast<T*>(mmaps));
    }
};

}} // namespace gbkfit::openmp
