#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include "gbkfit/cuda/kernels_objective.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_objective.hpp"

namespace gbkfit::cuda {

template<typename T> void
ObjectiveWrapper<T>::objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T missing, T limit, T limit_error,
        T scale, T jitter, T* res)
{
    const int n = size;
    launch("objective_residual", n, kernels::objective_residual<T>,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w,
            size, weight, missing, limit, limit_error,
            scale, jitter, res);
}

// The term of a residual sum, in double precision
template<typename T>
struct ResidualSumTerm
{
    bool squared;

    __host__ __device__ double
    operator()(T residual) const
    {
        const double r = residual;
        return squared ? r * r : fabs(r);
    }
};

template<typename T> void
ObjectiveWrapper<T>::objective_residual_sum(
        const T* residual, int size, bool squared, double* sum)
{
    // Accumulate in double precision: a cube can have millions of terms
    const double result = thrust::transform_reduce(
            thrust::device, residual, residual + size,
            ResidualSumTerm<T>{squared}, 0.0, thrust::plus<double>());
    check(cudaMemcpy(sum, &result, sizeof(double), cudaMemcpyHostToDevice),
          "objective_residual_sum");
}

template<typename T> void
ObjectiveWrapper<T>::objective_amplitude_sums(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        T scale, T jitter, int nouter, int nchannels, int ninner,
        double* a, double* b)
{
    const int n = nouter * ninner;
    launch("objective_amplitude_sums", n,
            kernels::objective_amplitude_sums<T>,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w,
            scale, jitter, nouter, nchannels, ninner, a, b);
}

// The term of value i of a noise residual (see
// gbkfit::objective_noise_term)
template<typename T>
struct NoiseSumTerm
{
    const T* obs_e;
    const T* obs_m;
    const T* mdl_d;
    const T* mdl_w;
    T weight, scale, jitter, scale_minimum;

    __host__ __device__ double
    operator()(int i) const
    {
        return gbkfit::objective_noise_term(
                i, obs_e, obs_m, mdl_d, mdl_w,
                weight, scale, jitter, scale_minimum);
    }
};

template<typename T> void
ObjectiveWrapper<T>::objective_noise_sum(
        const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T scale, T jitter, T scale_minimum,
        double* sum)
{
    // Accumulate in double precision: a cube can have millions of terms
    const double result = thrust::transform_reduce(
            thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(size),
            NoiseSumTerm<T>{
                obs_e, obs_m, mdl_d, mdl_w,
                weight, scale, jitter, scale_minimum},
            0.0, thrust::plus<double>());
    check(cudaMemcpy(sum, &result, sizeof(double), cudaMemcpyHostToDevice),
          "objective_noise_sum");
}

template struct ObjectiveWrapper<float>;
template struct ObjectiveWrapper<double>;

} // namespace gbkfit::cuda
