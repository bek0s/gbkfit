#pragma once

#include <gbkfit/dmodel/dmodels.hpp>

#include "gbkfit/cuda/kernels_common.hpp"

namespace gbkfit::cuda::kernels {

template<typename T> __global__ void
dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    // Parallelization: per 3d position in the dst dcube
    const int nthreads = dst_size_x * dst_size_y * dst_size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, dst_size_x, dst_size_y);

    gbkfit::dmodel_dcube_downscale(
            x, y, z,
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_cube, dst_cube);
}

template<typename T> __global__ void
dcube_moments(
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff,
        int norders,
        const int* orders,
        T* mmaps_d,
        T* mmaps_m,
        T* mmaps_w)
{
    // Parallelization: per 2d spatial position
    const int nthreads = size_x * size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, size_x);

    // The spectrum of the pixel is read from the cube (pixel tid of each
    // channel): the threads of a warp read adjacent values
    gbkfit::dmodel_mmaps_moments(
            x, y,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d + tid, dcube_w ? dcube_w + tid : nullptr, nthreads,
            cutoff, norders, orders,
            mmaps_d, mmaps_m, mmaps_w);
}

template<bool Hermite, typename T> __global__ void
dmodel_mmaps_gaussian(
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
    // Parallelization: per 2d spatial position
    const int nthreads = size_x * size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, size_x);

    // The spectrum of the pixel is read from the cube, as in dcube_moments
    gbkfit::dmodel_mmaps_gaussian<Hermite>(
            x, y,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d + tid, nthreads,
            cutoff, norders, orders,
            mmaps_d, mmaps_m);
}

template<typename T> __global__ void
dmodel_dcube_convolve_z(
        int size_x, int size_y, int size_z,
        int nk, const T* kernels, const T* src, T* dst)
{
    // Parallelization: per 3d position in the dcube
    const int nthreads = size_x * size_y * size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, size_x, size_y);

    gbkfit::dmodel_dcube_convolve_z(
            x, y, z, size_x, size_y, size_z, nk, kernels, src, dst);
}

template<typename T> __global__ void
dmodel_lens_resample(
        int nx, int ny, int nz, int sx, int sy,
        const T* source_x, const T* source_y,
        const T* source, T* image)
{
    // Parallelization: per 3d position in the image cube
    const int nthreads = nx * ny * nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, nx, ny);

    gbkfit::dmodel_lens_resample(
            x, y, z, nx, ny, sx, sy, source_x, source_y, source, image);
}

template<typename T> __global__ void
dmodel_regions_sum(
        int nregions, int npix, int size_z,
        const int* indptr, const int* indices, const T* weights,
        const T* cube, T* out)
{
    // Parallelization: per region and channel
    const int nthreads = nregions * size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int r, z;
    index_1d_to_2d(r, z, tid, nregions);

    gbkfit::dmodel_regions_sum(
            r, z,
            nregions, npix,
            indptr, indices, weights,
            cube, out);
}

} // namespace gbkfit::cuda::kernels
