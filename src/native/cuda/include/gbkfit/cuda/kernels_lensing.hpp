#pragma once

#include <gbkfit/lensing/layers.hpp>

#include "gbkfit/cuda/kernels_common.hpp"

namespace gbkfit::cuda::kernels {

template<typename T> __global__ void
lens_elliptical_deflection(
        int npoints, const T* x, const T* y, LensEllipse<T> ellipse,
        T* alpha_x, T* alpha_y)
{
    // Parallelization: per point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints)
        return;

    gbkfit::lens_elliptical_point(
            x[tid], y[tid], ellipse, alpha_x[tid], alpha_y[tid]);
}

template<typename T> __global__ void
lens_band_deflection(
        int npoints, const T* x, const T* y, LensBands<T> bands,
        T* alpha_x, T* alpha_y)
{
    // Parallelization: per point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints)
        return;

    gbkfit::lens_band_point(
            x[tid], y[tid], bands, alpha_x[tid], alpha_y[tid]);
}

} // namespace gbkfit::cuda::kernels
