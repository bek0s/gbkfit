#pragma once

#include <curand_kernel.h>
#include <thrust/random.h>

// This is on purpose under the namespace ::gbkfit and not ::gbkfit::cuda
// This hack allow us to pass any Random Number Generator type down to all the
// functions that need one without using a template parameter.

namespace gbkfit {
#if 0
template<typename T>
struct RNG
{
    __device__
    RNG(unsigned int tid) {
        curand_init(tid, tid, 0, &state);
    }

    __device__ T
    operator ()(void) {
        return curand_uniform(&state);
    }

    curandState state;
};
#else
template<typename T>
struct RNG
{
    __device__
    RNG(unsigned int tid)
        : gen(tid)
        , dis(0, 1) { gen.discard(tid); }

    __device__ T
    operator ()(void) {
        return dis(gen);
    }

    thrust::default_random_engine gen;
    thrust::uniform_real_distribution<T> dis;
};
#endif
} // namespace gbkfit
