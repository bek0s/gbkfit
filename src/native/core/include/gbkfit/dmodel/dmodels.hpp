#pragma once

#include "gbkfit/math/math.hpp"
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
    bool valid = mmaps_m[idx_2d] = std::abs(m0_sum) > cutoff;

    // Moment is valid only if not masked
    m0 = valid ? m0_sum : NAN;

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
    for (int z = 0; dcube_w && valid && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T w = dcube_w[idx];
        w_sum += w * i * step_z / m0;
    }

    // Weight is valid only if not masked
    w_sum = valid ? w_sum : NAN;

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
    for (int z = 0; valid && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T v = zero_z + z * step_z;
        m1_sum += i * v * step_z;
    }

    // Moment is valid only if not masked
    m1 = valid ? m1_sum / m0 : NAN;

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
    for (int z = 0; valid && z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        T i = dcube_d[idx];
        T v = zero_z + z * step_z;
        m2_sum += i * std::pow(v - m1, 2) * step_z;
    }

    // Moment is valid only if not masked
    m2 = valid ? std::sqrt(m2_sum / m0) : NAN;

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
        for (int z = 0; valid && z < size_z; ++z)
        {
            const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
            T flx = dcube_d[idx];
            T vel = zero_z + z * step_z;
            mn_sum += flx * std::pow(vel - m1, orders[m]) * step_z;
        }

        // Moment is valid only if not masked
        mn = valid ? mn_sum / m0 : NAN;

        const int idx = index_3d_to_1d(x, y, m, size_x, size_y);
        mmaps_d[idx] = mn;
    }
}

// The sum of the squared residuals of the Gaussian a exp(-(k - c)^2 / (2
// s^2)) (channel units) and the spectrum (x, y) of a cube
template<typename T> constexpr T
dmodel_gaussian_sse(
        int x, int y, int size_x, int size_y, int size_z,
        const T* dcube_d, T a, T c, T s)
{
    T sse = 0;
    for (int z = 0; z < size_z; ++z)
    {
        const T u = (z - c) / s;
        const T r = a * std::exp(T{-0.5} * u * u)
                - dcube_d[index_3d_to_1d(x, y, z, size_x, size_y)];
        sse += r * r;
    }
    return sse;
}

// The moment maps of a Gaussian fitted to the spectrum (x, y) of a cube
// by least squares (Levenberg-Marquardt), as moments: its flux (order 0),
// centre (1) and dispersion (2), the only orders. The spectra whose
// moment 0 is not above the cutoff, and those whose fit fails, are
// masked (NaN, and 0 in mmaps_m), as with the moments. The fit is of a
// Gaussian sampled at the centres of the channels, in channel units,
// and starts from the moments.
template<typename T> constexpr void
dmodel_mmaps_gaussian(
        int x, int y,
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
    constexpr int MAX_ITERATIONS = 100;
    constexpr T TOLERANCE = T{1e-6};
    const int idx_2d = index_2d_to_1d(x, y, size_x);

    // The moments, in channel units
    T m0 = 0, m1 = 0, m2 = 0;
    for (int z = 0; z < size_z; ++z)
    {
        const T i = dcube_d[index_3d_to_1d(x, y, z, size_x, size_y)];
        m0 += i;
        m1 += i * z;
    }
    m1 /= m0;
    for (int z = 0; z < size_z; ++z)
    {
        const T i = dcube_d[index_3d_to_1d(x, y, z, size_x, size_y)];
        m2 += i * (z - m1) * (z - m1);
    }
    m2 = std::sqrt(std::max(m2 / m0, T{0.25}));

    // The fit: amplitude a, centre c and dispersion s, from the moments
    T a = m0 / (m2 * std::sqrt(2 * PI<T>));
    T c = m1;
    T s = m2;
    bool valid = std::abs(m0 * step_z) > cutoff && std::isfinite(a);
    T sse = valid
            ? dmodel_gaussian_sse(x, y, size_x, size_y, size_z, dcube_d,
                                  a, c, s)
            : T{0};
    T lambda = T{1e-3};
    for (int iteration = 0; valid && iteration < MAX_ITERATIONS; ++iteration)
    {
        // The normal equations J^T J and J^T r of the residuals
        T jj[3][3] = {{0}}, jr[3] = {0};
        for (int z = 0; z < size_z; ++z)
        {
            const T u = (z - c) / s;
            const T g = std::exp(T{-0.5} * u * u);
            const T r = a * g
                    - dcube_d[index_3d_to_1d(x, y, z, size_x, size_y)];
            const T j[3] = {g, a * g * u / s, a * g * u * u / s};
            for (int p = 0; p < 3; ++p) {
                jr[p] += j[p] * r;
                for (int q = 0; q < 3; ++q)
                    jj[p][q] += j[p] * j[q];
            }
        }
        // Solve (J^T J + lambda diag(J^T J)) step = -J^T r (Cramer)
        T m[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
                m[p][q] = jj[p][q] * (p == q ? 1 + lambda : 1);
        const T det =
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (!(std::abs(det) > 0)) {
            break;
        }
        T delta[3];
        for (int p = 0; p < 3; ++p) {
            T n[3][3];
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    n[i][k] = k == p ? -jr[i] : m[i][k];
            delta[p] = (
                    n[0][0] * (n[1][1] * n[2][2] - n[1][2] * n[2][1])
                  - n[0][1] * (n[1][0] * n[2][2] - n[1][2] * n[2][0])
                  + n[0][2] * (n[1][0] * n[2][1] - n[1][1] * n[2][0])) / det;
        }
        const T a_new = a + delta[0];
        const T c_new = c + delta[1];
        const T s_new = s + delta[2];
        const T sse_new = s_new > 0
                ? dmodel_gaussian_sse(x, y, size_x, size_y, size_z, dcube_d,
                                      a_new, c_new, s_new)
                : NAN;
        if (sse_new < sse) {
            const bool converged = sse - sse_new <= TOLERANCE * sse;
            a = a_new;
            c = c_new;
            s = s_new;
            sse = sse_new;
            lambda /= 10;
            if (converged) {
                break;
            }
        } else {
            lambda *= 10;
            if (lambda > T{1e10}) {
                break;
            }
        }
    }
    valid = valid && std::isfinite(a) && std::isfinite(c) && s > 0;
    mmaps_m[idx_2d] = valid;

    // The moments of the Gaussian, in the units of the spectral axis
    for (int m = 0; m < norders; ++m)
    {
        T value = NAN;
        if (valid && orders[m] == 0)
            value = a * s * std::sqrt(2 * PI<T>) * step_z;
        else if (valid && orders[m] == 1)
            value = zero_z + c * step_z;
        else if (valid && orders[m] == 2)
            value = s * step_z;
        mmaps_d[index_3d_to_1d(x, y, m, size_x, size_y)] = value;
    }
}

// The weighted sum of the pixels of region r in channel z of a cube with
// npix pixels in each channel, into out[z][r] (out has nregions values
// in each channel). The region has the pixels indices[k] of the channel,
// with the weights weights[k], for k from indptr[r] to indptr[r + 1] - 1.
template<typename T> constexpr void
dmodel_regions_sum(
        int r, int z,
        int nregions, int npix,
        const int* indptr, const int* indices, const T* weights,
        const T* cube, T* out)
{
    // Accumulate in double precision: a region can have many pixels
    const T* channel = cube + static_cast<long>(z) * npix;
    double sum = 0;
    for (int k = indptr[r]; k < indptr[r + 1]; ++k)
    {
        sum += static_cast<double>(weights[k]) * channel[indices[k]];
    }
    out[z * nregions + r] = static_cast<T>(sum);
}

} // namespace gbkfit
