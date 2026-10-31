#pragma once

namespace gbkfit::cuda {

// The kernels of the objective (see gbkfit/objective)
template<typename T>
struct ObjectiveWrapper
{
    static void
    objective_residual(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            int size, T weight, T missing, T limit, T limit_error,
            T scale, T jitter, T* res);

    static void
    objective_residual_sum(
            const T* residual, int size, bool squared, double* sum);

    static void
    objective_amplitude_sums(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            T scale, T jitter, int nouter, int nchannels, int ninner,
            double* a, double* b);

    static void
    objective_noise_sum(
            const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            int size, T weight, T scale, T jitter, T scale_minimum,
            double* sum);
};

} // namespace gbkfit::cuda
