#pragma once

#include <gbkfit/orbits/orbits.hpp>

#include "gbkfit/cuda/kernels_common.hpp"

namespace gbkfit::cuda::kernels {

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
            if (c == 0 && tile == 0 && (rows.cells || a.cell_moments))
                gbkfit::orbits_record_cell<
                        atomic_add<T>, orbits_record_add<T>>(
                                k, sample, sample + 3, a, rows);
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
