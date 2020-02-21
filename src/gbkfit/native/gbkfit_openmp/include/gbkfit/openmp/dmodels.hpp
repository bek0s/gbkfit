#pragma once

#include <gbkfit/common.hpp>

#include "kernels_main.hpp"

namespace gbkfit { namespace openmp {

template<typename T>
struct DModelDCube
{
    DModelDCube(
            int size_lo_x, int size_lo_y, int size_lo_z,
            int size_hi_x, int size_hi_y, int size_hi_z,
            int edge_hi_x, int edge_hi_y, int edge_hi_z,
            int scale_x, int scale_y, int scale_z,
            Ptr scube_lo,
            Ptr scube_hi, Ptr scube_hi_fft,
            Ptr psf3d_hi, Ptr psf3d_hi_fft);

    ~DModelDCube();

    void
    convolve(void) const;

    void
    downscale(void) const;

private:

    std::array<int, 3> m_size_lo;
    std::array<int, 3> m_size_hi;
    std::array<int, 3> m_edge_hi;
    std::array<int, 3> m_scale;
    T* m_scube_lo;
    T* m_scube_hi;
    T* m_scube_hi_fft;
    T* m_psf3d_hi_fft;
    void* m_scube_fft_plan_r2c;
    void* m_scube_fft_plan_c2r;
    void* m_psf3d_fft_plan_r2c;
};

template<typename T>
struct DModelMMaps
{/*
    DModelMMaps(
            int spat_size_x, int spat_size_y,
            int spec_size,
            T spec_step,
            T spec_zero,
            CPtr scube,)
    {

    }*/
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
