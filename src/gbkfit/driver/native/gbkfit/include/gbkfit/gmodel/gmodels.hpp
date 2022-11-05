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

} // namespace
