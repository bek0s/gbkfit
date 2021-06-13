#pragma once

#include "gbkfit/host/common.hpp"
#include "gbkfit/host/fftutils.hpp"

namespace gbkfit::host {

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
            Ptr scube_r, Ptr scube_c,
            Ptr wcube_r, Ptr wcube_c,
            Ptr psf3d_r, Ptr psf3d_c) const;

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
    mutable T* m_scube_r;
    mutable T* m_wcube_r;
    mutable T* m_psf3d_r;
    mutable T* m_scube_c;
    mutable T* m_wcube_c;
    mutable T* m_psf3d_c;
    mutable typename fftw3<T>::plan m_scube_plan_r2c;
    mutable typename fftw3<T>::plan m_scube_plan_c2r;
    mutable typename fftw3<T>::plan m_wcube_plan_r2c;
    mutable typename fftw3<T>::plan m_wcube_plan_c2r;
    mutable typename fftw3<T>::plan m_psf3d_plan_r2c;
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

} // namespace gbkfit::host
