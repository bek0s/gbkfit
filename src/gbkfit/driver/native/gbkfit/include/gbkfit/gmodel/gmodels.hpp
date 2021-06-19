#pragma once

namespace gbkfit {

template<typename T> constexpr void
gmodel_weight_scube_pixel(
        int x, int y,
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size,
        const T* src, T* dst)
{
    T max_ = 0;

    for(int z = 0; z < spat_size_z; ++z)
    {
        int spat_idx =
                  x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        max_ = std::max(max_, src[spat_idx]);
    }

    for(int z = 0; z < spat_size_z; ++z)
    {
        int spat_idx =
                  x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        for(int v = 0; v < spec_size; ++v)
        {
            int spec_idx =
                      x
                    + y * spat_size_x
                    + v * spat_size_x * spat_size_y;

            dst[spec_idx] = src[spat_idx] / max_;
        }
    }
}

} // namespace
