#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>
#include <gbkfit/objective/objective.hpp>

namespace gbkfit::cuda::kernels {

template<typename T> inline constexpr void
atomic_add(T* addr, T val)
{
    atomicAdd(addr, val);
}

template<typename T> inline constexpr void
atomic_set(T* addr, T val)
{
    // TODO: revise this
    *addr = val;
}

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
dmodel_dcube_mask(
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    // Parallelization: per 3d position in the dcube
    const int nthreads = size_x * size_y * size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, size_x, size_y);

    gbkfit::dmodel_dcube_mask(
            x, y, z,
            cutoff, apply,
            size_x, size_y, size_z,
            dcube_d, dcube_m, dcube_w);
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

template<typename T> __global__ void
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
    gbkfit::dmodel_mmaps_gaussian(
            x, y,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d + tid, nthreads,
            cutoff, norders, orders,
            mmaps_d, mmaps_m);
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

template<typename T> __global__ void
gmodel_wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_cube,
        T* spec_cube)
{
    // Parallelization: per 2d spatial position
    const int nthreads = spat_size_x * spat_size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, spat_size_x);

    gbkfit::gmodel_wcube_pixel(
            x, y,
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            spat_cube,
            spec_cube);
}

template<typename T> __global__ void
gmodel_mcdisk_evaluate(DiskArgs<T> a, MCDiskArgs<T> mc)
{
    // Parallelization: per cloud
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= mc.nclouds)
        return;

    // Each cloud has its own stream of random numbers
    RNG<T> rng(mc.seed, tid);
    gbkfit::gmodel_mcdisk_evaluate_cloud<atomic_set<T>, atomic_add<T>>(
            rng, tid, a, mc);
}

template<typename T> __global__ void
gmodel_smdisk_evaluate(DiskArgs<T> a)
{
    // Parallelization: per 3d spatial position
    const int nthreads = a.spat_size[0] * a.spat_size[1] * a.spat_size[2];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, a.spat_size[0], a.spat_size[1]);

    gbkfit::gmodel_smdisk_evaluate_spaxel<atomic_add<T>>(x, y, z, a);
}

template<typename T> __global__ void
objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w, const T* mdl_m,
        int size, T weight, T* res)
{
    const int nthreads = size;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    gbkfit::objective_residual(
            tid,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w, mdl_m,
            size, weight, res);
}

} // namespace gbkfit::cuda::kernels
