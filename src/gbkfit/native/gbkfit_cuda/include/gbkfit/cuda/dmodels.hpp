#pragma once

#include "gbkfit/cuda/common.hpp"
#include "gbkfit/cuda/fftutils.hpp"

namespace gbkfit { namespace cuda {

template<typename T>
struct DModelDCube
{
public:

    DModelDCube(void);

    ~DModelDCube();

    void
    prepare(int size_lo_x, int size_lo_y, int size_lo_z,
            int size_hi_x, int size_hi_y, int size_hi_z,
            int edge_hi_x, int edge_hi_y, int edge_hi_z,
            int scale_x, int scale_y, int scale_z,
            Ptr scube_lo,
            Ptr scube_hi, Ptr scube_hi_fft,
            Ptr psf3d_hi, Ptr psf3d_hi_fft);

    void
    cleanup(void);

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
    cufftHandle m_scube_fft_plan_r2c;
    cufftHandle m_scube_fft_plan_c2r;
    cufftHandle m_psf3d_fft_plan_r2c;
};

template<typename T>
struct DModelMMaps
{
public:

    DModelMMaps(void);

    ~DModelMMaps() {}

    void
    prepare(int spat_size_x, int spat_size_y,
            int spec_size,
            T spec_step,
            T spec_zero,
            T nanval,
            Ptr scube,
            Ptr mmaps,
            Ptr mmaps_orders,
            int mmaps_count) {};

    void
    moments(void) const {}

private:

    int m_spat_size_x;
    int m_spat_size_y;
    int m_spec_size;
    T m_spec_step;
    T m_spec_zero;
    T m_nanval;
    T* m_scube;
    T* m_mmaps;
    int m_mmaps_count;
    int* m_mmaps_orders;
};

}} // namespace gbkfit::cuda
