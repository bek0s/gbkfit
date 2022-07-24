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
    using real = float;
    using complex = fftwf_complex;
    static constexpr auto free = fftwf_free;
    static constexpr auto malloc = fftwf_malloc;
    static constexpr auto alloc_real = fftwf_alloc_real;
    static constexpr auto alloc_complex = fftwf_alloc_complex;
    static constexpr auto alignment_of = fftwf_alignment_of;
    static constexpr auto init_threads = fftwf_init_threads;
    static constexpr auto cleanup_threads = fftwf_cleanup_threads;
    static constexpr auto plan_with_nthreads = fftwf_plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = fftwf_plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = fftwf_plan_dft_r2c_3d;
    static constexpr auto destroy_plan = fftwf_destroy_plan;
    static constexpr auto execute = fftwf_execute;
    static constexpr auto execute_dft_r2c = fftwf_execute_dft_r2c;
    static constexpr auto execute_dft_c2r = fftwf_execute_dft_c2r;
};

template<>
struct wrapper<double>
{
    using plan = fftw_plan;
    using real = double;
    using complex = fftw_complex;
    static constexpr auto free = fftw_free;
    static constexpr auto malloc = fftw_malloc;
    static constexpr auto alloc_real = fftw_alloc_real;
    static constexpr auto alloc_complex = fftw_alloc_complex;
    static constexpr auto alignment_of = fftw_alignment_of;
    static constexpr auto init_threads = fftw_init_threads;
    static constexpr auto cleanup_threads = fftw_cleanup_threads;
    static constexpr auto plan_with_nthreads = fftw_plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = fftw_plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = fftw_plan_dft_r2c_3d;
    static constexpr auto destroy_plan = fftw_destroy_plan;
    static constexpr auto execute = fftw_execute;
    static constexpr auto execute_dft_r2c = fftw_execute_dft_r2c;
    static constexpr auto execute_dft_c2r = fftw_execute_dft_c2r;
};

} // namespace

template<typename T>
struct fftw3
{
public:

    using plan = typename wrapper<T>::plan;
    using real = typename wrapper<T>::real;
    using complex = typename wrapper<T>::complex;
    static constexpr auto free = wrapper<T>::free;
    static constexpr auto malloc = wrapper<T>::malloc;
    static constexpr auto alloc_real = wrapper<T>::alloc_real;
    static constexpr auto alloc_complex = wrapper<T>::alloc_complex;
    static constexpr auto alignment_of = wrapper<T>::alignment_of;
    static constexpr auto plan_with_nthreads = wrapper<T>::plan_with_nthreads;
    static constexpr auto plan_dft_c2r_3d = wrapper<T>::plan_dft_c2r_3d;
    static constexpr auto plan_dft_r2c_3d = wrapper<T>::plan_dft_r2c_3d;
    static constexpr auto destroy_plan = wrapper<T>::destroy_plan;
    static constexpr auto execute = wrapper<T>::execute;
    static constexpr auto execute_dft_r2c = wrapper<T>::execute_dft_r2c;
    static constexpr auto execute_dft_c2r = wrapper<T>::execute_dft_c2r;

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

    static inline int refcount = 0;
};

} // namespace gbkfit
