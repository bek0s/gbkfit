#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>
#include <gbkfit/gmodel/transfer.hpp>
#include <gbkfit/objective/objective.hpp>
#include <gbkfit/orbits/orbits.hpp>

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

template<
        typename T, bool HasOpacity, bool HasFields, bool HasLineOpacity,
        bool Anisotropic>
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
            Anisotropic>(tid, a, g, mc, f);
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
        bool Anisotropic>
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
            Anisotropic>(x, y, z, a, g, f);
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
        T scale, T jitter, int nouter, int nchannels, int ninner,
        double* a, double* b)
{
    const int nthreads = nouter * ninner;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) {
        return;
    }

    // (one thread for each spectrum s = o * ninner + i, its value of
    // channel c at (o * nchannels + c) * ninner + i: along the first axis,
    // the threads of a warp read adjacent values of each channel)
    const long long first =
            (long long)(tid / ninner) * nchannels * ninner + tid % ninner;
    double sum_a = 0, sum_b = 0;
    for (int c = 0; c < nchannels; ++c) {
        gbkfit::objective_amplitude_terms(
                first + (long long)c * ninner,
                obs_d, obs_e, obs_m,
                mdl_d, mdl_w,
                scale, jitter, sum_a, sum_b);
    }
    a[tid] = sum_a;
    b[tid] = sum_b;
}

template<typename T> __global__ void
orbits_potential(
        PotentialArgs<T> p, int n, const T* points, T* potential, T* forces)
{
    // Parallelization: per point
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    gbkfit::orbits_potential_point(tid, p, points, potential, forces);
}

// Integrate the count orbits of a chunk, from orbit first, and keep their
// samples (see gbkfit::orbits_trajectory)
template<typename T> __global__ void
orbits_trajectories(
        GBKFIT_GRID_CONSTANT const OrbitArgs<T> a, int first, int count)
{
    // Parallelization: per orbit
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count)
        return;

    gbkfit::orbits_trajectory(
            first + tid, a, a.samples + 6LL * a.nsamples * tid);
}

// The unit of the records of orbits in shared memory (see
// orbits_record): they add up as integers in it, so that their sums do
// not depend on the order of the additions (each record of an orbit, a
// fraction of its light or its time, is at most about 2: its integers
// are at most 2^57 or so, of the 2^63 of a long long)
constexpr double ORBITS_RECORD_UNIT = 0x1p-56;

// Add a record to an integer of records in shared memory (see
// ORBITS_RECORD_UNIT; in its own precision: float64 is slow on most GPUs)
template<typename T> __device__ inline void
orbits_record_add(unsigned long long* addr, T val)
{
    long long record;
    if constexpr (std::is_same_v<T, float>)
        record = __float2ll_rn(val * float(1 / ORBITS_RECORD_UNIT));
    else
        record = __double2ll_rn(val * (1 / ORBITS_RECORD_UNIT));
    atomicAdd(addr, (unsigned long long)record);
}

// Record the orbits of a chunk, from orbit first, from their samples (see
// gbkfit::orbits_trajectory) into their outputs (see gbkfit::OrbitArgs),
// and add them to the spectra: each orbit by a block, whose threads share
// out its samples and their copies, and add its records (and its twin's)
// in shared memory, as integers (see ORBITS_RECORD_UNIT), those of ntile
// targets at a time (as many as it holds), then write them out (one
// record of an orbit in global memory at a time, from the threads of all
// orbits, was bound by the bandwidth of the memory)
template<typename T> __global__ void
orbits_record(GBKFIT_GRID_CONSTANT const OrbitArgs<T> a, int first, int ntile)
{
    // Parallelization: per orbit (block), and per copy of its samples
    // (thread)
    extern __shared__ unsigned long long shared[];
    const int k = first + blockIdx.x;
    const T* samples = a.samples + 6LL * a.nsamples * blockIdx.x;
    const int twins = a.twins ? 2 : 1;
    const long long orbit = (long long)k * twins;
    const int width = a.order + 2;
    const int ncells = a.cells ? a.ncell_r * a.ncell_b : 0;
    const int ntargets = a.nsky ? a.ntargets : 0;
    const int ncopies = 2 * a.nphi;
    T sky[3][3];
    gbkfit::orbits_sky(a, sky);

    // (the cells with the first tile; none without targets)
    for (int tile = 0; tile == 0 || tile < ntargets; tile += ntile)
    {
        gbkfit::OrbitRows<unsigned long long> rows;
        rows.first = tile;
        rows.last = min(tile + ntile, ntargets);
        const int size = a.moments ? (rows.last - rows.first) * width : 0;
        rows.moments = a.moments ? shared : nullptr;
        rows.twin_moments = size;
        rows.cells = a.cells && tile == 0 ? shared + twins * size : nullptr;
        rows.twin_cells = ncells;
        const int nshared = twins * (size + (rows.cells ? ncells : 0));
        for (int i = threadIdx.x; i < nshared; i += blockDim.x)
            shared[i] = 0;
        __syncthreads();

        for (long long i = threadIdx.x; i < (long long)a.nsamples * ncopies;
             i += blockDim.x)
        {
            const int n = int(i / ncopies);
            const int c = int(i % ncopies);
            const T* sample = samples + 6LL * n;
            if (isnan(sample[0]))
                continue;
            if (rows.cells && c == 0)
                gbkfit::orbits_record_cell<orbits_record_add<T>>(
                        sample, a, rows);
            if (a.nsky)
                gbkfit::orbits_record_copy<
                        atomic_add<T>, orbits_record_add<T>>(
                                k, n, c, sample, sample + 3, sky, a, rows);
        }
        __syncthreads();

        // (the records of the orbit, then those of its twin)
        const auto value = [](unsigned long long record) {
            return T((long long)record * ORBITS_RECORD_UNIT);
        };
        for (int i = threadIdx.x; i < twins * size; i += blockDim.x)
        {
            const long long row = (orbit + i / size) * ntargets + rows.first;
            a.moments[row * width + i % size] = value(shared[i]);
        }
        for (int i = threadIdx.x; rows.cells && i < twins * ncells;
             i += blockDim.x)
            a.cells[orbit * ncells + i] = value(rows.cells[i]);
        __syncthreads();
    }
}

} // namespace gbkfit::cuda::kernels
