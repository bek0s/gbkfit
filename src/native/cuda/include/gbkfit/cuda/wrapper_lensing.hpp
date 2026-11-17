#pragma once

#include <gbkfit/lensing/layers.hpp>

namespace gbkfit::cuda {

// The kernels of the deflections of the layers of the mass of galaxies on
// the sky (see gbkfit/lensing)
template<typename T>
struct LensingWrapper
{
    static void
    lens_elliptical_deflection(
            int npoints, const T* x, const T* y,
            const LensEllipse<T>& ellipse, T* alpha_x, T* alpha_y);

    static void
    lens_band_deflection(
            int npoints, const T* x, const T* y,
            const LensBands<T>& bands, T* alpha_x, T* alpha_y);
};

} // namespace gbkfit::cuda
