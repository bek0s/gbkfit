#pragma once

// The launch of the kernels, shared by the wrappers of each family (see
// wrapper.hpp)

#include <limits>
#include <stdexcept>
#include <string>

namespace gbkfit::cuda {

constexpr int BLOCK_SIZE = 256;

// Turn a cuda error into an exception, which nanobind raises in Python
inline void
check(cudaError_t error, const char* what)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
                std::string(what) + " failed: " + cudaGetErrorString(error));
    }
}

// Launch a kernel with one thread per item (n items, which can be more
// than an int holds), in blocks of the given size, wait for it to finish,
// and raise any error of its launch or execution
template<typename Kernel, typename... Args> void
launch_blocks(
        const char* name, long long n, int block_size, Kernel kernel,
        Args... args)
{
    if (n <= 0)
        return;
    const long long nblocks = (n + block_size - 1) / block_size;
    if (nblocks > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
                std::string(name) + " failed: too many items ("
                + std::to_string(n) + ")");
    }
    const dim3 bsize(block_size);
    const dim3 gsize(nblocks);
    kernel<<<gsize, bsize>>>(args...);
    check(cudaGetLastError(), name);
    check(cudaDeviceSynchronize(), name);
}

// Launch a kernel in nblocks blocks of the given size, each with the given
// bytes of shared memory, wait for it to finish, and raise any error of
// its launch or execution
template<typename Kernel, typename... Args> void
launch_grid(
        const char* name, int nblocks, int block_size, size_t shared,
        Kernel kernel, Args... args)
{
    if (nblocks <= 0)
        return;
    if (shared > 48 * 1024) {
        check(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(shared)), name);
    }
    kernel<<<nblocks, block_size, shared>>>(args...);
    check(cudaGetLastError(), name);
    check(cudaDeviceSynchronize(), name);
}

// Launch a kernel with one thread per item, in blocks of BLOCK_SIZE (see
// launch_blocks)
template<typename Kernel, typename... Args> void
launch(const char* name, long long n, Kernel kernel, Args... args)
{
    launch_blocks(name, n, BLOCK_SIZE, kernel, args...);
}

} // namespace gbkfit::cuda
