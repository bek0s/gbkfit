#pragma once

#include <gbkfit/gmodel/gmodels.hpp>
#include <gbkfit/gmodel/transfer.hpp>

#include "gbkfit/cuda/kernels_common.hpp"

namespace gbkfit::cuda::kernels {

template<
        typename T, bool HasOpacity, bool HasFields, bool HasLineOpacity,
        bool FullKinematics, bool HasVectors>
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

    // (each cloud has its own stream of random numbers; see
    // gmodel_mcdisk_evaluate_cloud)
    gbkfit::gmodel_mcdisk_evaluate_cloud<
            atomic_add<T>, HasOpacity, HasFields, HasLineOpacity,
            FullKinematics, HasVectors>(tid, a, g, mc, f);
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

template<
        typename T, bool HasOpacity, bool HasFields, bool HasLineOpacity,
        bool FullKinematics>
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
            atomic_add<T>, HasOpacity, HasFields, HasLineOpacity,
            FullKinematics>(x, y, z, a, g, f);
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

template<typename T, bool HasOpacity, bool HasFields>
__global__ void
gmodel_smspheroid_evaluate(
        GBKFIT_GRID_CONSTANT const SpheroidArgs<T> a,
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

    gbkfit::gmodel_smspheroid_evaluate_voxel<
            atomic_add<T>, HasOpacity, HasFields>(x, y, z, a, g, f);
}

template<typename T>
__global__ void
gmodel_smspheroid_evaluate_projected(
        GBKFIT_GRID_CONSTANT const SpheroidArgs<T> a,
        GBKFIT_GRID_CONSTANT const GridArgs<T> g)
{
    // Parallelization: per 2d spatial position
    const int nthreads = g.spat_size[0] * g.spat_size[1];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, g.spat_size[0]);

    gbkfit::gmodel_smspheroid_evaluate_pixel<atomic_add<T>>(x, y, a, g);
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

} // namespace gbkfit::cuda::kernels
