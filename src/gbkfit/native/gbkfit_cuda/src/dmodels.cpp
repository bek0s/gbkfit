
#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T>
DModelDCube<T>::DModelDCube(void)
    : m_size{0, 0, 0}
    , m_scube{nullptr}
    , m_scube_fft{nullptr}
    , m_psf3d_fft{nullptr}
    , m_scube_fft_plan_r2c{0}
    , m_scube_fft_plan_c2r{0}
    , m_psf3d_fft_plan_r2c{0}
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
        cufftDestroy(m_scube_fft_plan_r2c);
    }
    if (m_scube_fft_plan_c2r) {
        cufftDestroy(m_scube_fft_plan_c2r);
    }
    if (m_psf3d_fft_plan_r2c) {
        cufftDestroy(m_psf3d_fft_plan_r2c);
    }

    m_size = {0, 0, 0};
    m_scube = nullptr;
    m_scube_fft = nullptr;
    m_psf3d_fft = nullptr;
    m_scube_fft_plan_r2c = 0;
    m_scube_fft_plan_c2r = 0;
    m_psf3d_fft_plan_r2c = 0;
}

template<typename T> void
DModelDCube<T>::convolve(
        std::array<int, 3> size,
        Ptr scube, Ptr scube_fft,
        Ptr psf3d, Ptr psf3d_fft) const
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    T* scube_ptr = reinterpret_cast<T*>(scube);
    T* psf3d_ptr = reinterpret_cast<T*>(psf3d);
    T* scube_fft_ptr = reinterpret_cast<T*>(scube_fft);
    T* psf3d_fft_ptr = reinterpret_cast<T*>(psf3d_fft);

    if (m_scube != scube_ptr || m_scube_fft != scube_fft_ptr ||
        m_psf3d != psf3d_ptr || m_psf3d_fft != psf3d_fft_ptr || m_size != size)
    {
        cleanup();

        m_size = size;
        m_scube = scube_ptr;
        m_psf3d = psf3d_ptr;
        m_scube_fft = scube_fft_ptr;
        m_psf3d_fft = psf3d_fft_ptr;

        cufftPlan3d(&m_scube_fft_plan_r2c, n0, n1, n2, cufft<T>::R2C);

        cufftPlan3d(&m_scube_fft_plan_c2r, n0, n1, n2, cufft<T>::C2R);

        cufftPlan3d(&m_psf3d_fft_plan_r2c, n0, n1, n2, cufft<T>::R2C);

        cufft<T>::execR2C(
                m_psf3d_fft_plan_r2c,
                reinterpret_cast<typename cufft<T>::real*>(psf3d_ptr),
                reinterpret_cast<typename cufft<T>::complex*>(psf3d_fft));
    }

    cufft<T>::execR2C(
            m_scube_fft_plan_r2c,
            reinterpret_cast<typename cufft<T>::real*>(scube_ptr),
            reinterpret_cast<typename cufft<T>::complex*>(scube_fft_ptr));

    const int n = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);

    Wrapper<T>::dmodel_dcube_complex_multiply_and_scale(
            reinterpret_cast<typename cufft<T>::complex*>(scube_fft_ptr),
            reinterpret_cast<typename cufft<T>::complex*>(psf3d_fft_ptr),
            n, nfactor);

    cufft<T>::execC2R(
            m_scube_fft_plan_c2r,
            reinterpret_cast<typename cufft<T>::complex*>(scube_fft_ptr),
            reinterpret_cast<typename cufft<T>::real*>(scube_ptr));
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
DModelMMaps<T>::moments(
        std::array<int, 3> size,
        std::array<T, 3> step,
        std::array<T, 3> zero,
        Ptr scube,
        Ptr mmaps,
        Ptr orders,
        int norders) const
{
    Wrapper<T>::dmodel_mmaps_moments(
            size[0], size[1], size[2],
            step[0], step[1], step[2],
            zero[0], zero[1], zero[2],
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(mmaps),
            reinterpret_cast<int*>(orders),
            norders);
}

#define INSTANTIATE(T)\
    template struct DModelDCube<T>;\
    template struct DModelMMaps<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::cuda
