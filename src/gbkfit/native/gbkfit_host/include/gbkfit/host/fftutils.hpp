#pragma once

#include <fftw3.h>

namespace gbkfit {

namespace {

template<typename T>
struct wrapper;

template<>
struct wrapper<float>
{
    using plan = fftwf_plan;
    using complex = fftwf_complex;
    static constexpr auto init_threads = fftwf_init_threads;
    static constexpr auto cleanup_threads = fftwf_cleanup_threads;
    static constexpr auto plan_with_nthreads = fftwf_plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = fftwf_plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = fftwf_plan_dft_r2c_3d;
    static constexpr auto destroy_plan = fftwf_destroy_plan;
    static constexpr auto execute = fftwf_execute;
};

template<>
struct wrapper<double>
{
    using plan = fftw_plan;
    using complex = fftw_complex;
    static constexpr auto init_threads = fftw_init_threads;
    static constexpr auto cleanup_threads = fftw_cleanup_threads;
    static constexpr auto plan_with_nthreads = fftw_plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = fftw_plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = fftw_plan_dft_r2c_3d;
    static constexpr auto destroy_plan = fftw_destroy_plan;
    static constexpr auto execute = fftw_execute;
};

}

template<typename T>
struct fftw3
{
public:

    using plan = typename wrapper<T>::plan;
    using complex = typename wrapper<T>::complex;
    static constexpr auto plan_with_nthreads = wrapper<T>::plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = wrapper<T>::plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = wrapper<T>::plan_dft_r2c_3d;
    static constexpr auto destroy_plan = wrapper<T>::destroy_plan;
    static constexpr auto execute = wrapper<T>::execute;

    static constexpr void init_threads(void)
    {
        if (!refcount) {
            wrapper<T>::init_threads();
        }
        refcount++;
    }

    static constexpr void cleanup_threads(void)
    {
        if (refcount) {
            refcount--;
            if (!refcount) {
                // No idea why this causes segmentation fault
            //  wrapper<T>::cleanup_threads();
            }
        }
    }

private:

    static int refcount;
};

template<typename T>
int fftw3<T>::refcount = 0;

} // namespace gbkfit
