#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>
#include <gbkfit/gmodel/transfer.hpp>
#include <gbkfit/objective/objective.hpp>

// A kernel parameter that the kernel reads where it is, without a copy
// to local memory: the disk kernels take the address of their arguments,
// and a copy of DiskArgs and GridArgs costs each thread hundreds of bytes
// of stack.
// (__grid_constant__ needs compute capability 7.0; older architectures
// keep the copy.)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define GBKFIT_GRID_CONSTANT
#else
#define GBKFIT_GRID_CONSTANT __grid_constant__
#endif

namespace gbkfit::cuda::kernels {

template<typename T> inline constexpr void
atomic_add(T* addr, T val)
{
    atomicAdd(addr, val);
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

template<typename T, bool HasOpacity, bool HasFields, bool HasLineOpacity>
__global__ void
gmodel_mcdisk_evaluate(
        GBKFIT_GRID_CONSTANT const DiskArgs<T> a,
        GBKFIT_GRID_CONSTANT const GridArgs<T> g,
        GBKFIT_GRID_CONSTANT const MCDiskArgs<T> mc,
        GBKFIT_GRID_CONSTANT const FieldsArgs<T> f)
{
    // Parallelization: per cloud
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= mc.nclouds)
        return;

    // Each cloud has its own stream of random numbers
    RNG<T> rng(mc.seed, tid);
    gbkfit::gmodel_mcdisk_evaluate_cloud<
            atomic_add<T>, HasOpacity, HasFields, HasLineOpacity>(
                    rng, tid, a, g, mc, f);
}

template<typename T> __global__ void
gmodel_mcdisk_fields_moments(DiskArgs<T> a, GridArgs<T> g, FieldsArgs<T> f)
{
    // Parallelization: per voxel
    const long long nthreads = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    const long long tid = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    gbkfit::gmodel_mcdisk_fields_moments(tid, a, g, f);
}

template<typename T, bool HasOpacity, bool HasFields, bool HasLineOpacity>
__global__ void
gmodel_smdisk_evaluate(
        GBKFIT_GRID_CONSTANT const DiskArgs<T> a,
        GBKFIT_GRID_CONSTANT const GridArgs<T> g,
        GBKFIT_GRID_CONSTANT const FieldsArgs<T> f)
{
    // Parallelization: per 3d spatial position, or per 2d spatial position
    // for a thin disk (64-bit indices: there can be more voxels than an int
    // holds)
    const long long nthreads = 1LL * g.spat_size[0] * g.spat_size[1]
            * gbkfit::gmodel_smdisk_size_z(a, g);
    const long long tid = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    const int x = tid % g.spat_size[0];
    const int y = tid / g.spat_size[0] % g.spat_size[1];
    const int z = tid / (1LL * g.spat_size[0] * g.spat_size[1]);

    gbkfit::gmodel_smdisk_evaluate_spaxel<
            atomic_add<T>, HasOpacity, HasFields, HasLineOpacity>(
                    x, y, z, a, g, f);
}

template<typename T, bool HasOpacity, bool HasFields>
__global__ void
gmodel_smcone_evaluate(
        GBKFIT_GRID_CONSTANT const ConeArgs<T> a,
        GBKFIT_GRID_CONSTANT const GridArgs<T> g,
        GBKFIT_GRID_CONSTANT const FieldsArgs<T> f)
{
    // Parallelization: per 3d spatial position (see gmodel_smdisk_evaluate)
    const long long nthreads = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    const long long tid = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    const int x = tid % g.spat_size[0];
    const int y = tid / g.spat_size[0] % g.spat_size[1];
    const int z = tid / (1LL * g.spat_size[0] * g.spat_size[1]);

    gbkfit::gmodel_smcone_evaluate_voxel<atomic_add<T>, HasOpacity, HasFields>(
            x, y, z, a, g, f);
}

template<typename T> __global__ void
gmodel_opacity_accumulate(int size_x, int size_y, int size_z, T* opacity)
{
    // Parallelization: per 2d spatial position (the threads of a warp read
    // adjacent values)
    const int nthreads = size_x * size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, size_x);

    gbkfit::gmodel_opacity_accumulate(x, y, size_x, size_y, size_z, opacity);
}

// The most channels that a thread of the march marches (see
// gmodel_transfer_march): whole channels of the output, of all their
// sub-channels (see TransferArgs::subchannels)
constexpr int TRANSFER_CHUNK = 64;
static_assert(TRANSFER_CHUNK >= TRANSFER_SUBCHANNEL_NUM_MAX);

// The number of channels of the march of a thread
template<typename T> constexpr int
transfer_chunk_size(const TransferArgs<T>& a)
{
    return TRANSFER_CHUNK / a.subchannels * a.subchannels;
}

// The number of chunks of channels of the march of a spectral axis
template<typename T> constexpr int
transfer_chunks(const TransferArgs<T>& a)
{
    const int size = transfer_chunk_size(a);
    return (a.spec_size + size - 1) / size;
}

template<typename T, bool HasLineOpacity> __global__ void
gmodel_transfer_march(TransferArgs<T> a)
{
    // Parallelization: per 2d spatial position and chunk of channels.
    // Each thread reads the fields of each voxel once, and adds their light
    // to the channels of its chunk that their lines reach. The threads of
    // a warp are adjacent spaxels of one chunk: they read adjacent fields,
    // which the warps of the other chunks then read from the cache.
    const int nchunks = transfer_chunks(a);
    const long long nthreads = 1LL * a.spat_size[0] * a.spat_size[1]
            * nchunks;
    const long long tid = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    const int x = tid % a.spat_size[0];
    const int chunk = tid / a.spat_size[0] % nchunks;
    const int y = tid / (1LL * a.spat_size[0] * nchunks);
    const int c0 = chunk * transfer_chunk_size(a);
    const int c1 = std::min(c0 + transfer_chunk_size(a), a.spec_size);

    // (the optical depth of the lines in front and of the voxel, with
    // line opacity)
    constexpr int nline = HasLineOpacity ? TRANSFER_CHUNK : 1;
    T light[TRANSFER_CHUNK] = {};
    T tau[nline];
    T dk[nline] = {};
    gbkfit::gmodel_transfer_march<HasLineOpacity>(
            x, y, c0, c1, a, light, tau, dk);
    gbkfit::gmodel_transfer_add_light(x, y, c0, c1, a, light);
}

template<typename T> __global__ void
gmodel_transfer_opacity_front(TransferArgs<T> a)
{
    // Parallelization: per 2d spatial position
    const int nthreads = a.spat_size[0] * a.spat_size[1];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    const int x = tid % a.spat_size[0];
    const int y = tid / a.spat_size[0];
    gbkfit::gmodel_transfer_opacity_front(x, y, a);
}

template<typename T> __global__ void
gmodel_disk_traits_evaluate(DiskTraitsArgs<T> a)
{
    // Parallelization: per point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= a.npoints)
        return;

    gbkfit::gmodel_disk_traits_point(tid, a);
}

template<typename T> __global__ void
objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T missing, T limit, T limit_error,
        T scale, T jitter, T* res)
{
    const int nthreads = size;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    gbkfit::objective_residual(
            tid,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w,
            size, weight, missing, limit, limit_error,
            scale, jitter, res);
}

template<typename T> __global__ void
objective_amplitude_sums(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        T scale, T jitter, int nchannels, int nspectra, double* a, double* b)
{
    const int nthreads = nspectra;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    // (one thread for each spectrum: the threads of a warp read adjacent
    // values of each channel)
    double sum_a = 0, sum_b = 0;
    for (int c = 0; c < nchannels; ++c) {
        gbkfit::objective_amplitude_terms(
                (long long)c * nspectra + tid,
                obs_d, obs_e, obs_m,
                mdl_d, mdl_w,
                scale, jitter, sum_a, sum_b);
    }
    a[tid] = sum_a;
    b[tid] = sum_b;
}

} // namespace gbkfit::cuda::kernels
