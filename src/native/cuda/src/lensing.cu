#include "gbkfit/cuda/kernels_lensing.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_lensing.hpp"

namespace gbkfit::cuda {

// The threads of a block of the kernels of the layers, which take many
// registers (their loops of complex arithmetic in float64)
constexpr int LENS_BLOCK_SIZE = 128;

template<typename T> void
LensingWrapper<T>::lens_elliptical_deflection(
        int npoints, const T* x, const T* y,
        const LensEllipse<T>& ellipse, T* alpha_x, T* alpha_y)
{
    launch_blocks(
            "lens_elliptical_deflection", npoints, LENS_BLOCK_SIZE,
            kernels::lens_elliptical_deflection<T>,
            npoints, x, y, ellipse, alpha_x, alpha_y);
}

template<typename T> void
LensingWrapper<T>::lens_band_deflection(
        int npoints, const T* x, const T* y,
        const LensBands<T>& bands, T* alpha_x, T* alpha_y)
{
    launch_blocks(
            "lens_band_deflection", npoints, LENS_BLOCK_SIZE,
            kernels::lens_band_deflection<T>,
            npoints, x, y, bands, alpha_x, alpha_y);
}

template struct LensingWrapper<float>;
template struct LensingWrapper<double>;

} // namespace gbkfit::cuda
