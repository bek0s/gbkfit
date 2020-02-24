
#include "gbkfit/drivers/cuda/dmodels.hpp"
#include "gbkfit/drivers/cuda/wrapper.hpp"

namespace gbkfit { namespace cuda {

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
    , m_scube_fft_plan_r2c{0}
    , m_scube_fft_plan_c2r{0}
    , m_psf3d_fft_plan_r2c{0}
{
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

        cufftPlan3d(&m_scube_fft_plan_r2c, n0, n1, n2, cufft<T>::cufftTypeR2C);

        cufftPlan3d(&m_scube_fft_plan_c2r, n0, n1, n2, cufft<T>::cufftTypeC2R);

        cufftPlan3d(&m_psf3d_fft_plan_r2c, n0, n1, n2, cufft<T>::cufftTypeR2C);

        cufft<T>::cufftExecR2C(
                m_psf3d_fft_plan_r2c,
                reinterpret_cast<typename cufft<T>::real*>(psf3d_hi_ptr),
                reinterpret_cast<typename cufft<T>::complex*>(m_psf3d_hi_fft));
    }
}

template<typename T> void
DModelDCube<T>::cleanup(void)
{
    if (m_scube_fft_plan_r2c) {
        cufftDestroy(m_scube_fft_plan_r2c);
    }
    if (m_scube_fft_plan_c2r) {
        cufftDestroy(m_scube_fft_plan_c2r);
    }
    if (m_psf3d_fft_plan_r2c) {
        cufftDestroy(m_psf3d_fft_plan_r2c);
    }

    m_size_lo = {0, 0, 0};
    m_size_hi = {0, 0, 0};
    m_edge_hi = {0, 0, 0};
    m_scale = {0, 0, 0};
    m_scube_lo = nullptr;
    m_scube_hi = nullptr;
    m_scube_hi_fft = nullptr;
    m_psf3d_hi_fft = nullptr;
    m_scube_fft_plan_r2c = 0;
    m_scube_fft_plan_c2r = 0;
    m_psf3d_fft_plan_r2c = 0;
}

template<typename T> void
DModelDCube<T>::convolve(void) const
{
    int n0 = m_size_hi[2], n1 = m_size_hi[1], n2 = m_size_hi[0];

    cufft<T>::cufftExecR2C(
            m_scube_fft_plan_r2c,
            reinterpret_cast<typename cufft<T>::real*>(m_scube_hi),
            reinterpret_cast<typename cufft<T>::complex*>(m_scube_hi_fft));

    const int size = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);

    Wrapper<T>::dmodel_dcube_complex_multiply_and_scale(
            reinterpret_cast<typename cufft<T>::complex*>(m_scube_hi_fft),
            reinterpret_cast<typename cufft<T>::complex*>(m_psf3d_hi_fft),
            size, nfactor);

    cufft<T>::cufftExecC2R(
            m_scube_fft_plan_c2r,
            reinterpret_cast<typename cufft<T>::complex*>(m_scube_hi_fft),
            reinterpret_cast<typename cufft<T>::real*>(m_scube_hi));
}

template<typename T> void
DModelDCube<T>::downscale(void) const
{
}

#define INSTANTIATE(T)\
    template struct DModelDCube<T>;
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}} // namespace gbkfit::cuda
