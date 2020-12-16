#pragma once

#include "gbkfit/cuda/common.hpp"
#include "gbkfit/cuda/fftutils.hpp"

namespace gbkfit::cuda {

template<typename T>
struct DModelDCube
{
public:

    DModelDCube(void);

    ~DModelDCube();

    void
    cleanup(void) const;

    void
    convolve(
            std::array<int, 3> size,
            Ptr scube, Ptr scube_fft,
            Ptr psf3d, Ptr psf3d_fft) const;

    void
    downscale(
            std::array<int, 3> scale,
            std::array<int, 3> offset,
            std::array<int, 3> src_size,
            std::array<int, 3> dst_size,
            Ptr src_cube, Ptr dst_cube) const;

    void
    make_mask(
            bool mask_spat, bool mask_spec, T mask_coef,
            std::array<int, 3> size,
            Ptr cube, Ptr mask) const;

private:

    mutable std::array<int, 3> m_size;
    mutable T* m_scube;
    mutable T* m_psf3d;
    mutable T* m_scube_fft;
    mutable T* m_psf3d_fft;
    mutable cufftHandle m_scube_fft_plan_r2c;
    mutable cufftHandle m_scube_fft_plan_c2r;
    mutable cufftHandle m_psf3d_fft_plan_r2c;
};

template<typename T>
struct DModelMMaps
{
public:

    DModelMMaps(void) {}

    ~DModelMMaps() {}

    void
    moments(std::array<int, 3> size,
            std::array<T, 3> step,
            std::array<T, 3> zero,
            Ptr scube,
            Ptr mmaps,
            Ptr masks,
            Ptr orders,
            int norders) const;
};

} // namespace gbkfit::cuda
