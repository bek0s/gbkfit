#pragma once

#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

template<typename T> constexpr void
dmodel_dcube_downscale(
        int x, int y, int z,
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_dcube, T* dst_dcube)
{
    const T nfactor = T{1} / (scale_x * scale_y * scale_z);

    // Src cube 3d index
    int nx = offset_x + x * scale_x;
    int ny = offset_y + y * scale_y;
    int nz = offset_z + z * scale_z;

    // Calculate average value under the current position
    T sum = 0;
    for(int dsz = 0; dsz < scale_z; ++dsz)
    {
    for(int dsy = 0; dsy < scale_y; ++dsy)
    {
    for(int dsx = 0; dsx < scale_x; ++dsx)
    {
        int idx = index_3d_to_1d(
                nx + dsx,
                ny + dsy,
                nz + dsz,
                src_size_x, src_size_y);

        sum += src_dcube[idx];
    }
    }
    }

    // Dst cube 1d index
    int idx = index_3d_to_1d(x, y, z, dst_size_x, dst_size_y);

    dst_dcube[idx] = sum * nfactor;
}

template<typename T> constexpr void
dmodel_dcube_mask(
        int x, int y, int z,
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    // Do not touch the weights for now.
    // Rethink about this in the future.
    (void)dcube_w;

    const int idx = index_3d_to_1d(x, y, z, size_x, size_y);

    T mvalue = 1;

    if (std::fabs(dcube_d[idx]) <= cutoff)
    {
        mvalue = 0;
        if (apply) {
            dcube_d[idx] = NAN;
        }
    }

    if (dcube_m) {
        dcube_m[idx] = mvalue;
    }
}

template<typename T> constexpr void
dmodel_mmaps_moments(
        int x, int y,
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* dcube_d, const T* dcube_w,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m, T* mmaps_w)
{
    // Spatial step is not needed for now,
    // but who knows? Maybe will use them in the future.
    (void)step_x;
    (void)step_y;
    (void)zero_x;
    (void)zero_y;

    // Index of the current spatial position
    const int idx_2d = index_2d_to_1d(x, y, size_x);

    // Moment orders are assumed to be sorted
    const int max_order = orders[norders - 1];

    // Tracks the moment we are current processing
    int m = 0;

    //
    // Moment 0
    //

    T m0=0, m0_sum=0;
    for (int z = 0; z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        m0_sum += i * step_z;
    }

    // Check if we need to mask this spatial position
    bool masked = mmaps_m[idx_2d] = std::abs(m0_sum) <= cutoff;

    // Moment is valid only if not masked
    m0 = masked ? NAN : m0_sum;

    // Store moment if it was requested
    if (orders[m] == 0)
    {
        const int idx = index_3d_to_1d(x, y, m, size_x, size_y);
        mmaps_d[idx] = m0;
        m++;
    }

    //
    // Weight
    //

    T w_sum = 0;
    for (int z = 0; dcube_w && !masked && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T w = dcube_w[idx];
        w_sum += w * i * step_z / m0;
    }

    // Weight is valid only if not masked
    w_sum = masked ? NAN : w_sum;

    if (dcube_w)
    {
        mmaps_w[idx_2d] = w_sum;
    }

    // Max order reached
    if (max_order == 0) {
        return;
    }

    //
    // Moment 1
    //

    T m1=0, m1_sum=0;
    for (int z = 0; !masked && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T v = zero_z + z * step_z;
        m1_sum += i * v * step_z;
    }

    // Moment is valid only if not masked
    m1 = masked ? NAN : m1_sum / m0;

    // Only output requested moments
    if (orders[m] == 1)
    {
        const int idx = index_3d_to_1d(x, y, m, size_x, size_y);
        mmaps_d[idx] = m1;
        m++;
    }

    // Max order reached
    if (max_order == 1) {
        return;
    }

    //
    // Moment 2
    //

    T m2=0, m2_sum=0;
    for (int z = 0; !masked && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T v = zero_z + z * step_z;
        m2_sum += i * std::pow(v - m1, 2) * step_z;
    }

    // Moment is valid only if not masked
    m2 = masked ? NAN : std::sqrt(m2_sum / m0);

    // Only output requested moments
    if (orders[m] == 2)
    {
        const int idx = index_3d_to_1d(x, y, m, size_x, size_y);
        mmaps_d[idx] = m2;
        m++;
    }

    //
    // Higher order moments
    //

    for(; m < norders; ++m)
    {
        T mn=0, mn_sum=0;
        for (int z = 0; !masked && z < size_z; ++z)
        {
            const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
            T flx = dcube_d[idx];
            T vel = zero_z + z * step_z;
            mn_sum += flx * std::pow(vel - m1, orders[m]); // * step_z;
        }

        // Moment is valid only if not masked
        mn = masked ? NAN : mn_sum / m0;

        const int idx = index_3d_to_1d(x, y, m, size_x, size_y);
        mmaps_d[idx] = mn;
    }
}

} // namespace gbkfit
