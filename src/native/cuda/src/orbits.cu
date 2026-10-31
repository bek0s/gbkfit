#include <algorithm>

#include "gbkfit/cuda/kernels_orbits.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_orbits.hpp"

namespace gbkfit::cuda {

// The stack of a thread of the orbit kernels (bytes)
constexpr size_t ORBITS_STACK = 4096;

template<typename T> void
OrbitsWrapper<T>::orbits_potential(
        const PotentialArgs<T>& p, int n, const T* points,
        T* potential, T* forces)
{
    launch("orbits_potential", n, kernels::orbits_potential<T>,
            p, n, points, potential, forces);
}

// A chunk at a time: its orbits integrated, one per thread, in blocks of a
// warp (an orbit is long, and a library has thousands, not millions, so
// small blocks spread them over all the multiprocessors), then recorded,
// one per block (see kernels::orbits_record), with as many targets at a
// time as the shared memory of a block holds (of a long long each)
template<typename T> void
OrbitsWrapper<T>::orbits_integrate(const OrbitArgs<T>& a)
{
    // (the stages of a step of DOP853 of each thread, 13 of 6 values, need
    // more than the 1 KiB of stack of a thread by default)
    size_t stack;
    check(cudaDeviceGetLimit(&stack, cudaLimitStackSize),
            "cudaDeviceGetLimit");
    if (stack < ORBITS_STACK)
        check(cudaDeviceSetLimit(cudaLimitStackSize, ORBITS_STACK),
                "cudaDeviceSetLimit");
    int device, max_shared;
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaDeviceGetAttribute(
            &max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
            "cudaDeviceGetAttribute");
    const int twins = a.twins ? 2 : 1;
    const long long ncells = a.cells ? a.ncell_r * a.ncell_b : 0;
    const long long width = a.moments ? a.order + 2 : 0;
    const int ntargets = a.nsky ? a.ntargets : 0;
    int ntile = std::max(ntargets, 1);
    if (width) {
        const long long room = max_shared / 8 - twins * ncells;
        ntile = int(std::min<long long>(ntile, room / (twins * width)));
        if (ntile < 1) {
            throw std::runtime_error(
                    "orbits_integrate failed: the cells of an orbit do not "
                    "fit in the shared memory of a block");
        }
    }
    const size_t shared = 8 * twins * (ntile * width + ncells);
    for (int first = 0; first < a.norbits; first += a.nchunk) {
        const int count = std::min(a.nchunk, a.norbits - first);
        // (none if the samples are those of the last integration)
        if (!a.record_only) {
            launch_blocks("orbits_trajectories", count, 32,
                    kernels::orbits_trajectories<T>, a, first, count);
        }
        launch_grid("orbits_record", count, BLOCK_SIZE, shared,
                kernels::orbits_record<T>, a, first, ntile);
    }
}

template struct OrbitsWrapper<float>;
template struct OrbitsWrapper<double>;

} // namespace gbkfit::cuda
