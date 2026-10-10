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

// The voxel (x, y, z) of the convolution of the spectra of src along z
// with a kernel for each channel (e.g. an LSF that varies with
// wavelength), into dst: the light of the channel zs of src goes to the
// channels zs + k - nk / 2, with the weights kernels[zs * nk + k] (k <
// nk). Light from beyond the cube is 0.
template<typename T> constexpr void
dmodel_dcube_convolve_z(
        int x, int y, int z,
        int size_x, int size_y, int size_z,
        int nk, const T* kernels, const T* src, T* dst)
{
    const int half = nk / 2;
    T sum = 0;
    for (int k = 0; k < nk; ++k)
    {
        // The channel of src whose light reaches z with the weight k
        const int zs = z - k + half;
        if (zs < 0 || zs >= size_z)
            continue;
        sum += kernels[zs * nk + k]
                * src[index_3d_to_1d(x, y, zs, size_x, size_y)];
    }
    dst[index_3d_to_1d(x, y, z, size_x, size_y)] = sum;
}

// The row (y, z) of dmodel_dcube_convolve_z, summed weight by weight over
// the contiguous pixels of the rows of src (for the host, where this is
// much faster than voxel by voxel)
template<typename T> constexpr void
dmodel_dcube_convolve_z_row(
        int y, int z,
        int size_x, int size_y, int size_z,
        int nk, const T* kernels, const T* src, T* dst)
{
    const int half = nk / 2;
    T* out = dst + index_3d_to_1d(0, y, z, size_x, size_y);
    for (int x = 0; x < size_x; ++x)
        out[x] = 0;
    for (int k = 0; k < nk; ++k)
    {
        const int zs = z - k + half;
        if (zs < 0 || zs >= size_z)
            continue;
        const T w = kernels[zs * nk + k];
        if (w == 0)
            continue;
        const T* in = src + index_3d_to_1d(0, y, zs, size_x, size_y);
        for (int x = 0; x < size_x; ++x)
            out[x] += w * in[x];
    }
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

// The moments of the given orders (sorted) of the spectrum of the pixel
// (x, y) of a cube: its flux (order 0), mean velocity (1), dispersion
// (2), and central moments (above 2); with the weights of the spectrum,
// their flux-weighted mean (mmaps_w). The spectra whose moment 0 is not
// above the cutoff are masked (NaN, and 0 in mmaps_m), and with orders
// from 2, those whose variance is negative (e.g. the faint ringing of a
// convolution), which have no dispersion. The channels of the spectrum
// (and of its weights) are stride values apart: the spectrum can be in
// the cube or in a contiguous copy. The sums are in channels from the
// centre of the axis, so that their precision does not depend on its
// velocities.
//
// Precision: in float32, the moments of the faint wings of a convolved
// cube (below about 1e-4 of its peak) are off by up to a few km/s. The
// error is in the cube, from the rounding of the float32 FFT convolution
// (about 1e-7 of the peak), not in these sums: float64 sums over the
// float32 cube give the same moments (tested 2026-10-10). A float64
// model, or a float64 convolution of float32 models, would fix them;
// a mask_cutoff masks them.
template<typename T> constexpr void
dmodel_mmaps_moments(
        int x, int y,
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* spectrum_d, const T* spectrum_w, int stride,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m, T* mmaps_w)
{
    const int idx_2d = index_2d_to_1d(x, y, size_x);

    // Moment orders are assumed to be sorted
    const int max_order = orders[norders - 1];

    // The flux and the mean channel, from the centre of the axis
    const T centre = T(size_z - 1) / 2;
    T flux_sum = 0, mean_sum = 0;
    for (int z = 0; z < size_z; ++z)
    {
        T i = spectrum_d[z * stride];
        flux_sum += i;
        mean_sum += i * (z - centre);
    }
    const T m0 = flux_sum * step_z;
    const T mean = mean_sum / flux_sum;
    bool valid = std::abs(m0) > cutoff;

    // The variance about the mean channel
    T variance = 0;
    if (valid && max_order >= 2)
    {
        T variance_sum = 0;
        for (int z = 0; z < size_z; ++z)
        {
            T i = spectrum_d[z * stride];
            variance_sum += i * (z - centre - mean) * (z - centre - mean);
        }
        variance = variance_sum / flux_sum;
        valid = variance >= 0;
    }
    mmaps_m[idx_2d] = valid;

    // Weight
    T w_sum = 0;
    for (int z = 0; spectrum_w && valid && z < size_z; ++z)
    {
        T i = spectrum_d[z * stride];
        T w = spectrum_w[z * stride];
        w_sum += w * i;
    }
    if (spectrum_w)
    {
        mmaps_w[idx_2d] = valid ? w_sum / flux_sum : NAN;
    }

    // The moments of the given orders, valid only if not masked, in the
    // units of the spectral axis
    for (int m = 0; m < norders; ++m)
    {
        const int order = orders[m];
        T value = NAN;
        if (valid && order == 0)
        {
            value = m0;
        }
        else if (valid && order == 1)
        {
            value = (zero_z + centre * step_z) + mean * step_z;
        }
        else if (valid && order == 2)
        {
            value = std::sqrt(variance) * std::abs(step_z);
        }
        else if (valid)
        {
            T mn_sum = 0;
            for (int z = 0; z < size_z; ++z)
            {
                T i = spectrum_d[z * stride];
                mn_sum += i * ipow(z - centre - mean, order);
            }
            value = mn_sum / flux_sum * ipow(step_z, order);
        }
        mmaps_d[index_3d_to_1d(x, y, m, size_x, size_y)] = value;
    }
}

// The sum of the squared residuals of the Gaussian a exp(-(k - c)^2 / (2
// s^2)) (channel units) and a spectrum of size_z channels, stride values
// apart
template<typename T> constexpr T
dmodel_gaussian_sse(
        int size_z, const T* spectrum, int stride, T a, T c, T s)
{
    T sse = 0;
    for (int z = 0; z < size_z; ++z)
    {
        const T u = (z - c) / s;
        const T r = a * std::exp(T{-0.5} * u * u) - spectrum[z * stride];
        sse += r * r;
    }
    return sse;
}

// The moment maps of a Gaussian fitted to the spectrum of the pixel (x,
// y) of a cube by least squares (Levenberg-Marquardt), as moments: its
// flux (order 0), centre (1) and dispersion (2), the only orders. The
// spectra whose moment 0 is not above the cutoff, and those whose fit
// fails, are masked (NaN, and 0 in mmaps_m), as with the moments. The fit
// is of a Gaussian sampled at the centres of the channels, in channel
// units, and starts from the moments. The channels of the spectrum are
// stride values apart, as in dmodel_mmaps_moments.
template<typename T> constexpr void
dmodel_mmaps_gaussian(
        int x, int y,
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* spectrum, int stride,
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
        const T i = spectrum[z * stride];
        m0 += i;
        m1 += i * z;
    }
    m1 /= m0;
    for (int z = 0; z < size_z; ++z)
    {
        const T i = spectrum[z * stride];
        m2 += i * (z - m1) * (z - m1);
    }
    m2 = std::sqrt(std::max(m2 / m0, T{0.25}));

    // The fit: amplitude a, centre c and dispersion s, from the moments
    T a = m0 / (m2 * std::sqrt(2 * PI<T>));
    T c = m1;
    T s = m2;
    bool valid = std::abs(m0 * step_z) > cutoff && std::isfinite(a);
    T sse = valid
            ? dmodel_gaussian_sse(size_z, spectrum, stride, a, c, s)
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
            const T r = a * g - spectrum[z * stride];
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
                ? dmodel_gaussian_sse(size_z, spectrum, stride,
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

// The pixel (x, y) of channel z of an image-plane cube (nx by ny pixels),
// lensed from a source-plane cube (sx by sy pixels): the bilinear
// interpolation of the source at the position of the pixel on the source
// plane, in the pixel coordinates of the source (source_x and source_y,
// at index y * nx + x), and 0 outside the source.
template<typename T> constexpr void
dmodel_lens_resample(
        int x, int y, int z,
        int nx, int ny, int sx, int sy,
        const T* source_x, const T* source_y,
        const T* source, T* image)
{
    const int i = index_2d_to_1d(x, y, nx);
    const T px = source_x[i];
    const T py = source_y[i];
    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const T fx = px - x0;
    const T fy = py - y0;
    T value = 0;
    for (int dy = 0; dy < 2; ++dy)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const int xs = x0 + dx;
            const int ys = y0 + dy;
            if (xs < 0 || ys < 0 || xs >= sx || ys >= sy)
                continue;
            const T w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
            value += w * source[index_3d_to_1d(xs, ys, z, sx, sy)];
        }
    }
    image[index_3d_to_1d(x, y, z, nx, ny)] = value;
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
