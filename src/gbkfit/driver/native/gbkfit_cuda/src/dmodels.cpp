
#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T>
DModelDCube<T>::DModelDCube(void)
    : m_size{0, 0, 0}
    , m_scube_r{nullptr}
    , m_wcube_r{nullptr}
    , m_scube_c{nullptr}
    , m_wcube_c{nullptr}
    , m_psf3d_c{nullptr}
    , m_scube_plan_r2c{0}
    , m_scube_plan_c2r{0}
    , m_wcube_plan_r2c{0}
    , m_wcube_plan_c2r{0}
    , m_psf3d_plan_r2c{0}
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
        cufftDestroy(m_scube_plan_r2c);
    }
    if (m_scube_plan_c2r) {
        cufftDestroy(m_scube_plan_c2r);
    }
    if (m_wcube_plan_r2c) {
        cufftDestroy(m_wcube_plan_r2c);
    }
    if (m_wcube_plan_c2r) {
        cufftDestroy(m_wcube_plan_c2r);
    }
    if (m_psf3d_plan_r2c) {
        cufftDestroy(m_psf3d_plan_r2c);
    }

    m_size = {0, 0, 0};
    m_scube_r = nullptr;
    m_wcube_r = nullptr;
    m_scube_c = nullptr;
    m_wcube_c = nullptr;
    m_psf3d_c = nullptr;
    m_scube_plan_r2c = 0;
    m_scube_plan_c2r = 0;
    m_wcube_plan_r2c = 0;
    m_wcube_plan_c2r = 0;
    m_psf3d_plan_r2c = 0;
}

namespace {

template <typename T> void
_make_plan_r2c(cufftHandle& plan_r2c, std::array<int, 3> size)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    cufftPlan3d(&plan_r2c, n0, n1, n2, cufft<T>::R2C);
}

template <typename T> void
_make_plan_c2r(cufftHandle& plan_c2r, std::array<int, 3> size)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    cufftPlan3d(&plan_c2r, n0, n1, n2, cufft<T>::C2R);
}

template <typename T> void
_exec_plan_r2c(cufftHandle& plan_r2c, T* cube_r, T* cube_c)
{
    cufft<T>::execR2C(
            plan_r2c,
            reinterpret_cast<typename cufft<T>::real*>(cube_r),
            reinterpret_cast<typename cufft<T>::complex*>(cube_c));
}

template <typename T> void
_exec_plan_c2r(cufftHandle& plan_c2r, T* cube_r, T* cube_c)
{
    cufft<T>::execC2R(
            plan_c2r,
            reinterpret_cast<typename cufft<T>::complex*>(cube_c),
            reinterpret_cast<typename cufft<T>::real*>(cube_r));
}

template<typename T> void
_complex_multiply(std::array<int, 3> size, T* cube_c, T* psf_c)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    const int n = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);
    Wrapper<T>::dmodel_dcube_complex_multiply_and_scale(
            reinterpret_cast<typename cufft<T>::complex*>(cube_c),
            reinterpret_cast<typename cufft<T>::complex*>(psf_c),
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

        if (scube_c_ptr) {
            _make_plan_r2c<T>(m_scube_plan_r2c, size);
            _make_plan_c2r<T>(m_scube_plan_c2r, size);
        }
        if (wcube_c_ptr) {
            _make_plan_r2c<T>(m_wcube_plan_r2c, size);
            _make_plan_c2r<T>(m_wcube_plan_c2r, size);
        }

        _make_plan_r2c<T>(m_psf3d_plan_r2c, size);
        _exec_plan_r2c(m_psf3d_plan_r2c, psf3d_r_ptr, psf3d_c_ptr);
    }

    if (scube_c_ptr) {
        _exec_plan_r2c(m_scube_plan_r2c, scube_r_ptr, scube_c_ptr);
        _complex_multiply(size, scube_c_ptr, psf3d_c_ptr);
        _exec_plan_c2r(m_scube_plan_c2r, scube_r_ptr, scube_c_ptr);
    }
    if (wcube_c_ptr) {
        _exec_plan_r2c(m_wcube_plan_r2c, wcube_r_ptr, wcube_c_ptr);
        _complex_multiply(size, wcube_c_ptr, psf3d_c_ptr);
        _exec_plan_c2r(m_wcube_plan_c2r, wcube_r_ptr, wcube_c_ptr);
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
    Wrapper<T>::dmodel_dcube_downscale(
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
    Wrapper<T>::dmodel_dcube_make_mask(
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
    Wrapper<T>::dmodel_mmaps_moments(
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

} // namespace gbkfit::cuda
