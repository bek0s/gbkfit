
#include "gbkfit/host/dmodels.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T>
DModelDCube<T>::DModelDCube(void)
    : m_size{0, 0, 0}
    , m_scube_r{nullptr}
    , m_wcube_r{nullptr}
    , m_scube_c{nullptr}
    , m_wcube_c{nullptr}
    , m_psf3d_c{nullptr}
    , m_scube_plan_r2c{nullptr}
    , m_scube_plan_c2r{nullptr}
    , m_wcube_plan_r2c{nullptr}
    , m_wcube_plan_c2r{nullptr}
    , m_psf3d_plan_r2c{nullptr}
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
    if (m_scube_plan_r2c) {
        fftw3<T>::destroy_plan(m_scube_plan_r2c);
    }
    if (m_scube_plan_c2r) {
        fftw3<T>::destroy_plan(m_scube_plan_c2r);
    }
    if (m_wcube_plan_r2c) {
        fftw3<T>::destroy_plan(m_wcube_plan_r2c);
    }
    if (m_wcube_plan_c2r) {
        fftw3<T>::destroy_plan(m_wcube_plan_c2r);
    }
    if (m_psf3d_plan_r2c) {
        fftw3<T>::destroy_plan(m_psf3d_plan_r2c);
    }

    fftw3<T>::cleanup_threads();

    m_size = {0, 0, 0};
    m_scube_r = nullptr;
    m_wcube_r = nullptr;
    m_scube_c = nullptr;
    m_wcube_c = nullptr;
    m_psf3d_c = nullptr;
    m_scube_plan_r2c = nullptr;
    m_scube_plan_c2r = nullptr;
    m_wcube_plan_r2c = nullptr;
    m_wcube_plan_c2r = nullptr;
    m_psf3d_plan_r2c = nullptr;
}

namespace {

template<typename T> void
_make_plan_r2c(
        typename fftw3<T>::plan& plan_r2c,
        std::array<int, 3> size, T* cube_r, T* cube_c)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    std::vector<T> tmp(n0*n1*n2);
    std::copy_n(cube_r, n0*n1*n2, tmp.data());
    plan_r2c = fftw3<T>::plan_dft_r2c_3d(
            n0, n1, n2,
            cube_r,
            reinterpret_cast<typename fftw3<T>::complex*>(cube_c),
            FFTW_ESTIMATE);
    std::copy_n(tmp.data(), n0*n1*n2, cube_r);
}

template<typename T> void
_make_plan_c2r(
        typename fftw3<T>::plan& plan_c2r,
        std::array<int, 3> size, T* cube_r, T* cube_c)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    std::vector<T> tmp(n0*n1*n2);
    std::copy_n(cube_r, n0*n1*n2, tmp.data());
    plan_c2r = fftw3<T>::plan_dft_c2r_3d(
            n0, n1, n2,
            reinterpret_cast<typename fftw3<T>::complex*>(cube_c),
            cube_r,
            FFTW_ESTIMATE);
    std::copy_n(tmp.data(), n0*n1*n2, cube_r);
}

template<typename T> void
_complex_multiply(std::array<int, 3> size, T* cube_c, T* psf_c)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    const int n = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);
    kernels::dmodel_dcube_complex_multiply_and_scale<T>(
            reinterpret_cast<typename fftw3<T>::complex*>(cube_c),
            reinterpret_cast<typename fftw3<T>::complex*>(psf_c),
            n, nfactor);
}

} // namespace

template<typename T> void
DModelDCube<T>::convolve(
        std::array<int, 3> size,
        Ptr scube_r, Ptr scube_c,
        Ptr wcube_r, Ptr wcube_c,
        Ptr psf3d_r, Ptr psf3d_c) const
{
    T* scube_r_ptr = reinterpret_cast<T*>(scube_r);
    T* wcube_r_ptr = reinterpret_cast<T*>(wcube_r);
    T* psf3d_r_ptr = reinterpret_cast<T*>(psf3d_r);
    T* scube_c_ptr = reinterpret_cast<T*>(scube_c);
    T* wcube_c_ptr = reinterpret_cast<T*>(wcube_c);
    T* psf3d_c_ptr = reinterpret_cast<T*>(psf3d_c);

    if (m_scube_r != scube_r_ptr || m_scube_c != scube_c_ptr ||
        m_wcube_r != wcube_r_ptr || m_wcube_c != wcube_c_ptr ||
        m_psf3d_r != psf3d_r_ptr || m_psf3d_c != psf3d_c_ptr || m_size != size)
    {
        cleanup();

        m_size = size;
        m_scube_r = scube_r_ptr;
        m_wcube_r = wcube_r_ptr;
        m_psf3d_r = psf3d_r_ptr;
        m_scube_c = scube_c_ptr;
        m_wcube_c = wcube_c_ptr;
        m_psf3d_c = psf3d_c_ptr;

        fftw3<T>::init_threads();
        fftw3<T>::plan_with_nthreads(omp_get_num_procs());

        if (scube_c_ptr) {
            _make_plan_r2c(m_scube_plan_r2c, size, scube_r_ptr, scube_c_ptr);
            _make_plan_c2r(m_scube_plan_c2r, size, scube_r_ptr, scube_c_ptr);
        }
        if (wcube_c_ptr) {
            _make_plan_r2c(m_wcube_plan_r2c, size, wcube_r_ptr, wcube_c_ptr);
            _make_plan_c2r(m_wcube_plan_c2r, size, wcube_r_ptr, wcube_c_ptr);
        }
        _make_plan_r2c(m_psf3d_plan_r2c, size, psf3d_r_ptr, psf3d_c_ptr);
        fftw3<T>::execute(m_psf3d_plan_r2c);
    }

    if (scube_c_ptr) {
        fftw3<T>::execute(m_scube_plan_r2c);
        _complex_multiply(size, scube_c_ptr, psf3d_c_ptr);
        fftw3<T>::execute(m_scube_plan_c2r);
    }
    if (wcube_c_ptr) {
        fftw3<T>::execute(m_wcube_plan_r2c);
        _complex_multiply(size, wcube_c_ptr, psf3d_c_ptr);
        fftw3<T>::execute(m_wcube_plan_c2r);
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
