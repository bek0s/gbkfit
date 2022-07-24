#pragma once

#include "gbkfit/constants.hpp"
#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

template<typename T> constexpr void
gmodel_wcube_pixel(
        int x, int y,
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_wcube,
        T* spec_wcube)
{
    T sum = 0;
    T mean = 0;
    T maximum = 0;

    // Find the maximum value and sum at position (x, y) of the spatial cube
    for(int z = 0; z < spat_size_z; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        T value = spat_wcube[idx];
        sum += value;
        maximum = std::max(maximum, value);
    }

    // Calculate mean weight
    mean = sum / spat_size_z;

    // Assign the same mean weight across the entire spectrum
    for(int z = 0; z < spec_size_z; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        spec_wcube[idx] = maximum > 0 ? mean / maximum : 0;
    }
}

template<auto AtomicAddFunT, typename T> void constexpr
gmodel_image_evaluate(T* image, int x, int y, T rvalue, int spat_size_x)
{
    const int idx = index_2d_to_1d(x, y, spat_size_x);
    AtomicAddFunT(&image[idx], rvalue);
}

template<auto AtomicAddFunT, typename T> void constexpr
gmodel_scube_evaluate(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero)
{
    // Calculate a spectral range that encloses most of the flux.
    T zmin = vvalue - dvalue * LINE_WIDTH_MULTIPLIER<T>;
    T zmax = vvalue + dvalue * LINE_WIDTH_MULTIPLIER<T>;
    int zmin_idx = std::max<T>(std::rint(
            (zmin - spec_zero)/spec_step), 0);
    int zmax_idx = std::min<T>(std::rint(
            (zmax - spec_zero)/spec_step), spec_size - 1);

    // Evaluate the spectral line within the range specified above
    // Evaluating only within the range can result in huge speed increase
    for (int z = zmin_idx; z <= zmax_idx; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        T zvel = spec_zero + z * spec_step;
        T flux = rvalue * gauss_1d_pdf<T>(zvel, vvalue, dvalue); // * spec_step;
        AtomicAddFunT(&scube[idx], flux);
    }
}

} // namespace
