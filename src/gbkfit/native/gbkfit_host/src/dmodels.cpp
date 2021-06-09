
#include "gbkfit/host/dmodels.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T>
DModelDCube<T>::DModelDCube(void)
    : m_size{0, 0, 0}
    , m_scube{nullptr}
    , m_wcube{nullptr}
    , m_scube_fft{nullptr}
    , m_wcube_fft{nullptr}
    , m_psf3d_fft{nullptr}
    , m_scube_fft_plan_r2c{nullptr}
    , m_scube_fft_plan_c2r{nullptr}
    , m_wcube_fft_plan_r2c{nullptr}
    , m_wcube_fft_plan_c2r{nullptr}
    , m_psf3d_fft_plan_r2c{nullptr}
{
}

template<typename T>
DModelDCube<T>::~DModelDCube()
{
    cleanup();
}

template<typename T> void
DModelDCube<T>::cleanup(void) const
{
    if (m_scube_fft_plan_r2c) {
        fftw3<T>::destroy_plan(m_scube_fft_plan_r2c);
    }
    if (m_scube_fft_plan_c2r) {
        fftw3<T>::destroy_plan(m_scube_fft_plan_c2r);
    }
    if (m_wcube_fft_plan_r2c) {
        fftw3<T>::destroy_plan(m_wcube_fft_plan_r2c);
    }
    if (m_wcube_fft_plan_c2r) {
        fftw3<T>::destroy_plan(m_wcube_fft_plan_c2r);
    }
    if (m_psf3d_fft_plan_r2c) {
        fftw3<T>::destroy_plan(m_psf3d_fft_plan_r2c);
    }

    fftw3<T>::cleanup_threads();

    m_size = {0, 0, 0};
    m_scube = nullptr;
    m_wcube = nullptr;
    m_scube_fft = nullptr;
    m_wcube_fft = nullptr;
    m_psf3d_fft = nullptr;
    m_scube_fft_plan_r2c = nullptr;
    m_scube_fft_plan_c2r = nullptr;
    m_wcube_fft_plan_r2c = nullptr;
    m_wcube_fft_plan_c2r = nullptr;
    m_psf3d_fft_plan_r2c = nullptr;
}

template<typename T> void
DModelDCube<T>::convolve(
        std::array<int, 3> size,
        Ptr scube, Ptr scube_fft,
        Ptr wcube, Ptr wcube_fft,
        Ptr psf3d, Ptr psf3d_fft) const
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    T* scube_ptr = reinterpret_cast<T*>(scube);
    T* wcube_ptr = reinterpret_cast<T*>(wcube);
    T* psf3d_ptr = reinterpret_cast<T*>(psf3d);
    T* scube_fft_ptr = reinterpret_cast<T*>(scube_fft);
    T* wcube_fft_ptr = reinterpret_cast<T*>(wcube_fft);
    T* psf3d_fft_ptr = reinterpret_cast<T*>(psf3d_fft);

    if (m_scube != scube_ptr || m_scube_fft != scube_fft_ptr ||
        m_wcube != wcube_ptr || m_wcube_fft != wcube_fft_ptr ||
        m_psf3d != psf3d_ptr || m_psf3d_fft != psf3d_fft_ptr || m_size != size)
    {
        cleanup();

        m_size = size;
        m_scube = scube_ptr;
        m_wcube = wcube_ptr;
        m_psf3d = psf3d_ptr;
        m_scube_fft = scube_fft_ptr;
        m_wcube_fft = wcube_fft_ptr;
        m_psf3d_fft = psf3d_fft_ptr;

        fftw3<T>::init_threads();
        fftw3<T>::plan_with_nthreads(omp_get_num_procs());

        std::vector<T> tmp(n0*n1*n2);

        if (scube_ptr)
        {
            std::copy_n(scube_ptr, n0*n1*n2, tmp.data());
            m_scube_fft_plan_r2c = fftw3<T>::plan_dft_r2c_3d(
                    n0, n1, n2,
                    scube_ptr,
                    reinterpret_cast<typename fftw3<T>::complex*>(
                            scube_fft_ptr),
                    FFTW_ESTIMATE);
            m_scube_fft_plan_c2r = fftw3<T>::plan_dft_c2r_3d(
                    n0, n1, n2,
                    reinterpret_cast<typename fftw3<T>::complex*>(
                            scube_fft_ptr),
                    scube_ptr,
                    FFTW_ESTIMATE);
            std::copy_n(tmp.data(), n0*n1*n2, scube_ptr);
        }

        if (wcube_ptr)
        {
            std::copy_n(wcube_ptr, n0*n1*n2, tmp.data());
            m_wcube_fft_plan_r2c = fftw3<T>::plan_dft_r2c_3d(
                    n0, n1, n2,
                    wcube_ptr,
                    reinterpret_cast<typename fftw3<T>::complex*>(
                            wcube_fft_ptr),
                    FFTW_ESTIMATE);
            m_wcube_fft_plan_c2r = fftw3<T>::plan_dft_c2r_3d(
                    n0, n1, n2,
                    reinterpret_cast<typename fftw3<T>::complex*>(
                            wcube_fft_ptr),
                    wcube_ptr,
                    FFTW_ESTIMATE);
            std::copy_n(tmp.data(), n0*n1*n2, wcube_ptr);
        }

        std::copy_n(psf3d_ptr, n0*n1*n2, tmp.data());
        m_psf3d_fft_plan_r2c = fftw3<T>::plan_dft_r2c_3d(
                n0, n1, n2,
                psf3d_ptr,
                reinterpret_cast<typename fftw3<T>::complex*>(psf3d_fft_ptr),
                FFTW_ESTIMATE);
        std::copy_n(tmp.data(), n0*n1*n2, psf3d_ptr);
        fftw3<T>::execute(reinterpret_cast<typename fftw3<T>::plan>(
                m_psf3d_fft_plan_r2c));
    }

    if (scube_ptr)
    {
        fftw3<T>::execute(m_scube_fft_plan_r2c);

        const int n = n0 * n1 * (n2 / 2 + 1);
        const T nfactor = T{1} / (n0 * n1 * n2);

        kernels::dmodel_dcube_complex_multiply_and_scale<T>(
                reinterpret_cast<typename fftw3<T>::complex*>(scube_fft_ptr),
                reinterpret_cast<typename fftw3<T>::complex*>(psf3d_fft_ptr),
                n, nfactor);

        fftw3<T>::execute(m_scube_fft_plan_c2r);
    }

    if (wcube_ptr)
    {
        fftw3<T>::execute(m_wcube_fft_plan_r2c);

        const int n = n0 * n1 * (n2 / 2 + 1);
        const T nfactor = T{1} / (n0 * n1 * n2);

        kernels::dmodel_dcube_complex_multiply_and_scale<T>(
                reinterpret_cast<typename fftw3<T>::complex*>(wcube_fft_ptr),
                reinterpret_cast<typename fftw3<T>::complex*>(psf3d_fft_ptr),
                n, nfactor);

        fftw3<T>::execute(m_wcube_fft_plan_c2r);
    }
}

template<typename T> void
DModelDCube<T>::downscale(
        std::array<int, 3> scale,
        std::array<int, 3> offset,
        std::array<int, 3> src_size,
        std::array<int, 3> dst_size,
        Ptr src_cube, Ptr dst_cube) const
{
    kernels::dmodel_dcube_downscale(
            scale[0], scale[1], scale[2],
            offset[0], offset[1], offset[2],
            src_size[0], src_size[1], src_size[2],
            dst_size[0], dst_size[1], dst_size[2],
            reinterpret_cast<T*>(src_cube),
            reinterpret_cast<T*>(dst_cube));
}

template<typename T> void
DModelDCube<T>::make_mask(
        bool mask_spat, bool mask_spec, T mask_coef,
        std::array<int, 3> size,
        Ptr cube, Ptr mask) const
{
    kernels::dmodel_dcube_make_mask(
            mask_spat, mask_spec, mask_coef,
            size[0], size[1], size[2],
            reinterpret_cast<T*>(cube),
            reinterpret_cast<T*>(mask));
}

template<typename T> void
DModelMMaps<T>::moments(
        std::array<int, 3> size,
        std::array<T, 3> step,
        std::array<T, 3> zero,
        Ptr scube,
        Ptr mmaps,
        Ptr masks,
        Ptr orders,
        int norders) const
{
    kernels::dmodel_dcube_moments(
            size[0], size[1], size[2],
            step[0], step[1], step[2],
            zero[0], zero[1], zero[2],
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(mmaps),
            reinterpret_cast<T*>(masks),
            reinterpret_cast<int*>(orders),
            norders);
}

#define INSTANTIATE(T)\
    template struct DModelDCube<T>;\
    template struct DModelMMaps<T>;
    INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::host
