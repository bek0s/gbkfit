
#include "gbkfit/drivers/host/dmodels.hpp"
#include "gbkfit/drivers/host/kernels_main.hpp"

namespace gbkfit { namespace host {

template<typename T>
DModelDCube<T>::DModelDCube(void)
    : m_size_lo{0, 0, 0}
    , m_size_hi{0, 0, 0}
    , m_edge_hi{0, 0, 0}
    , m_scale{0, 0, 0}
    , m_scube_lo{nullptr}
    , m_scube_hi{nullptr}
    , m_scube_hi_fft{nullptr}
    , m_psf3d_hi_fft{nullptr}
    , m_scube_fft_plan_r2c{nullptr}
    , m_scube_fft_plan_c2r{nullptr}
    , m_psf3d_fft_plan_r2c{nullptr}
{
}

template<typename T>
DModelDCube<T>::~DModelDCube()
{
    cleanup();
}

template<typename T> void
DModelDCube<T>::prepare(
        int size_lo_x, int size_lo_y, int size_lo_z,
        int size_hi_x, int size_hi_y, int size_hi_z,
        int edge_hi_x, int edge_hi_y, int edge_hi_z,
        int scale_x, int scale_y, int scale_z,
        Ptr scube_lo,
        Ptr scube_hi, Ptr scube_hi_fft,
        Ptr psf3d_hi, Ptr psf3d_hi_fft)
{
    cleanup();

    m_size_lo = {size_lo_x, size_lo_y, size_lo_z};
    m_size_hi = {size_hi_x, size_hi_y, size_hi_z};
    m_edge_hi = {edge_hi_x, edge_hi_y, edge_hi_z};
    m_scale = {scale_x, scale_y, scale_z};
    m_scube_lo = reinterpret_cast<T*>(scube_lo);
    m_scube_hi = reinterpret_cast<T*>(scube_hi);
    m_scube_hi_fft = reinterpret_cast<T*>(scube_hi_fft);
    m_psf3d_hi_fft = reinterpret_cast<T*>(psf3d_hi_fft);

    if (psf3d_hi && psf3d_hi_fft)
    {
        T* psf3d_hi_ptr = reinterpret_cast<T*>(psf3d_hi);

        int n0 = m_size_hi[2], n1 = m_size_hi[1], n2 = m_size_hi[0];

        std::vector<T> tmp(n0*n1*n2);

        fftw3<T>::init_threads();
        fftw3<T>::plan_with_nthreads(std::thread::hardware_concurrency());

        std::copy_n(m_scube_hi, n0*n1*n2, tmp.data());
        m_scube_fft_plan_r2c = fftw3<T>::plan_dft_r2c_3d(
                n0, n1, n2,
                m_scube_hi,
                reinterpret_cast<typename fftw3<T>::complex*>(m_scube_hi_fft),
                FFTW_ESTIMATE);
        m_scube_fft_plan_c2r = fftw3<T>::plan_dft_c2r_3d(
                n0, n1, n2,
                reinterpret_cast<typename fftw3<T>::complex*>(m_scube_hi_fft),
                m_scube_hi,
                FFTW_ESTIMATE);
        std::copy_n(tmp.data(), n0*n1*n2, m_scube_hi);

        std::copy_n(psf3d_hi_ptr, n0*n1*n2, tmp.data());
        m_psf3d_fft_plan_r2c = fftw3<T>::plan_dft_r2c_3d(
                n0, n1, n2,
                psf3d_hi_ptr,
                reinterpret_cast<typename fftw3<T>::complex*>(m_psf3d_hi_fft),
                FFTW_ESTIMATE);
        std::copy_n(tmp.data(), n0*n1*n2, psf3d_hi_ptr);
        fftw3<T>::execute(reinterpret_cast<typename fftw3<T>::plan>(
                m_psf3d_fft_plan_r2c));
    }
}

template<typename T> void
DModelDCube<T>::cleanup(void)
{
    if (m_scube_fft_plan_r2c) {
        fftw3<T>::destroy_plan(m_scube_fft_plan_r2c);
    }
    if (m_scube_fft_plan_c2r) {
        fftw3<T>::destroy_plan(m_scube_fft_plan_c2r);
    }
    if (m_psf3d_fft_plan_r2c) {
        fftw3<T>::destroy_plan(m_psf3d_fft_plan_r2c);
    }

    fftw3<T>::cleanup_threads();

    m_size_lo = {0, 0, 0};
    m_size_hi = {0, 0, 0};
    m_edge_hi = {0, 0, 0};
    m_scale = {0, 0, 0};
    m_scube_lo = nullptr;
    m_scube_hi = nullptr;
    m_scube_hi_fft = nullptr;
    m_psf3d_hi_fft = nullptr;
    m_scube_fft_plan_r2c = nullptr;
    m_scube_fft_plan_c2r = nullptr;
    m_psf3d_fft_plan_r2c = nullptr;
}

template<typename T> void
DModelDCube<T>::convolve(void) const
{
    int n0 = m_size_hi[2], n1 = m_size_hi[1], n2 = m_size_hi[0];

    fftw3<T>::execute(m_scube_fft_plan_r2c);

    const int size = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);

    kernels::complex_multiply_and_scale(
            reinterpret_cast<std::complex<T>*>(m_scube_hi_fft),
            reinterpret_cast<std::complex<T>*>(m_psf3d_hi_fft),
            size, nfactor);

    fftw3<T>::execute(m_scube_fft_plan_c2r);
}

template<typename T> void
DModelDCube<T>::downscale(void) const
{
    kernels::dcube_downscale(
            m_scale[0], m_scale[1], m_scale[2],
            m_edge_hi[0], m_edge_hi[1], m_edge_hi[2],
            m_size_hi[0], m_size_hi[1], m_size_hi[2],
            m_size_lo[0], m_size_lo[1], m_size_lo[2],
            m_scube_hi,
            m_scube_lo);
}

template<typename T>
DModelMMaps<T>::DModelMMaps(void)
    : m_spat_size_x{0}
    , m_spat_size_y{0}
    , m_spec_size{0}
    , m_spec_step{0}
    , m_spec_zero{0}
    , m_nanval{0}
    , m_scube{nullptr}
    , m_mmaps{nullptr}
    , m_mmaps_count{0}
    , m_mmaps_orders{nullptr}
{
}

template<typename T> void
DModelMMaps<T>::prepare(
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero,
        T nanval,
        Ptr scube,
        Ptr mmaps,
        int mmaps_count,
        Ptr mmaps_orders)
{
    m_spat_size_x = spat_size_x;
    m_spat_size_y = spat_size_y;
    m_spec_size = spec_size;
    m_spec_step = spec_step;
    m_spec_zero = spec_zero;
    m_nanval = nanval;
    m_scube = reinterpret_cast<T*>(scube);
    m_mmaps = reinterpret_cast<T*>(mmaps);
    m_mmaps_count = mmaps_count;
    m_mmaps_orders = reinterpret_cast<int*>(mmaps_orders);
}

template<typename T> void
DModelMMaps<T>::moments(void) const
{
    kernels::dcube_moments(
            m_spat_size_x, m_spat_size_y,
            m_spec_size,
            m_spec_step,
            m_spec_zero,
            m_nanval,
            m_scube,
            m_mmaps,
            m_mmaps_count,
            m_mmaps_orders);
}

#define INSTANTIATE(T)\
    template struct DModelDCube<T>;\
    template struct DModelMMaps<T>;
    INSTANTIATE(float)
#undef INSTANTIATE

}} // namespace gbkfit::host
