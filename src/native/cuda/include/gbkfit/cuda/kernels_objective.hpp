#pragma once

#include <gbkfit/objective/objective.hpp>

#include "gbkfit/cuda/kernels_common.hpp"

namespace gbkfit::cuda::kernels {

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

} // namespace gbkfit::cuda::kernels
