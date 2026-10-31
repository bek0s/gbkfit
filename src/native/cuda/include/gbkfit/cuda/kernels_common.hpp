#pragma once

// What the kernels of every family share (see kernels_*.hpp)

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

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

} // namespace gbkfit::cuda::kernels
