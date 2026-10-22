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
// above the cutoff are masked (0 in mmaps_m), and with orders from 2,
// those whose variance is negative (e.g. the faint ringing of a
// convolution), which have no dispersion: their moments from order 1 are
// NaN (moment 0, their flux, keeps its value, so that a model without
// flux where the data have some is compared with them; the objective
// gives the NaN moments a penalty). The channels of the spectrum
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
// model, or a float64 convolution of float32 models (the observations'
// fft_dtype, float64 by default), fixes them.
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

    // The weight: its flux-weighted mean (its plain mean in a masked
    // spectrum, whose moment 0 still counts)
    T w_sum = 0, w_plain_sum = 0;
    for (int z = 0; spectrum_w && z < size_z; ++z)
    {
        T i = spectrum_d[z * stride];
        T w = spectrum_w[z * stride];
        w_sum += w * i;
        w_plain_sum += w;
    }
    if (spectrum_w)
    {
        mmaps_w[idx_2d] = valid ? w_sum / flux_sum : w_plain_sum / size_z;
    }

    // The moments of the given orders, in the units of the spectral axis:
    // moment 0 (the flux) always, the others only if not masked (NaN: the
    // model has no velocity where it has no flux)
    for (int m = 0; m < norders; ++m)
    {
        const int order = orders[m];
        T value = NAN;
        if (order == 0)
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

// Whether the Gaussian a exp(-(k - c)^2 / (2 s^2)) (channel units) is
// one of a line of a spectrum of size_z channels: bright (a > 0), centred
// within the channels (from -0.5 to size_z - 0.5), and not wider than
// them (0 < s <= size_z)
template<typename T> constexpr bool
dmodel_gaussian_in_band(int size_z, T a, T c, T s)
{
    return a > 0 && c >= T{-0.5} && c <= size_z - T{0.5}
        && s > 0 && s <= size_z;
}

// The Gaussian a exp(-(k - c)^2 / (2 s^2)) (channel units) fitted to a
// spectrum of size_z channels, stride values apart, by least squares
// (Levenberg-Marquardt) from (a, c, s), which it updates. The steps that
// leave the Gaussians of a line of the spectrum (see
// dmodel_gaussian_in_band) are rejected, as those that do not lower the
// residuals. Return the sum of the squared residuals of the fit, or
// infinity if the start is not one of those Gaussians.
template<typename T> constexpr T
dmodel_gaussian_fit(
        int size_z, const T* spectrum, int stride, T& a, T& c, T& s)
{
    constexpr int MAX_ITERATIONS = 100;
    constexpr T TOLERANCE = T{1e-6};
    if (!dmodel_gaussian_in_band(size_z, a, c, s)) {
        return INFINITY;
    }
    T sse = dmodel_gaussian_sse(size_z, spectrum, stride, a, c, s);
    T lambda = T{1e-3};
    // The normal equations J^T J and J^T r of the residuals, made again
    // only when the fit moves (a rejected step only raises the damping)
    T jj[3][3], jr[3];
    bool moved = true;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
    {
        if (moved)
        {
            for (int p = 0; p < 3; ++p) {
                jr[p] = 0;
                for (int q = 0; q < 3; ++q)
                    jj[p][q] = 0;
            }
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
            moved = false;
        }
        // The damped step, by Cramer's rule
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
        const T sse_new = dmodel_gaussian_in_band(size_z, a_new, c_new, s_new)
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
            moved = true;
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
    return sse;
}

// The Gauss-Hermite polynomials H3 and H4 of u (van der Marel & Franx
// 1993, as pPXF), and their derivatives
template<typename T> constexpr void
dmodel_hermite(T u, T& h3, T& h4, T& dh3, T& dh4)
{
    const T sqrt3 = std::sqrt(T{3});
    const T sqrt6 = std::sqrt(T{6});
    h3 = (2 * u * u * u - 3 * u) / sqrt3;
    h4 = (4 * u * u * u * u - 12 * u * u + 3) / (2 * sqrt6);
    dh3 = sqrt3 * (2 * u * u - 1);
    dh4 = (8 * u * u * u - 12 * u) / sqrt6;
}

// The value at channel z of the Gauss-Hermite series of p = (a, c, s, h3,
// h4), a exp(-u^2 / 2) (1 + h3 H3(u) + h4 H4(u)) with u = (z - c) / s
// (channel units), and its derivatives by p (in j, if given)
template<typename T> constexpr T
dmodel_gauss_hermite(int z, const T (&p)[5], T* j = nullptr)
{
    const T u = (z - p[1]) / p[2];
    const T g = std::exp(T{-0.5} * u * u);
    T h3 = 0, h4 = 0, dh3 = 0, dh4 = 0;
    dmodel_hermite(u, h3, h4, dh3, dh4);
    const T poly = 1 + p[3] * h3 + p[4] * h4;
    if (j)
    {
        // (d/du of the series, over -s: its derivative by c)
        const T dc = p[0] * g * (u * poly - p[3] * dh3 - p[4] * dh4) / p[2];
        j[0] = g * poly;
        j[1] = dc;
        j[2] = dc * u;
        j[3] = p[0] * g * h3;
        j[4] = p[0] * g * h4;
    }
    return p[0] * g * poly;
}

// The sum of the squared residuals of the Gauss-Hermite series of p (see
// dmodel_gauss_hermite) and a spectrum of size_z channels, stride values
// apart
template<typename T> constexpr T
dmodel_gauss_hermite_sse(
        int size_z, const T* spectrum, int stride, const T (&p)[5])
{
    T sse = 0;
    for (int z = 0; z < size_z; ++z)
    {
        const T r = dmodel_gauss_hermite(z, p) - spectrum[z * stride];
        sse += r * r;
    }
    return sse;
}

// The solution x of the N by N symmetric positive definite system m x =
// b, by Cholesky; false if m is not positive definite
template<int N, typename T> constexpr bool
solve_spd(const T (&m)[N][N], const T (&b)[N], T (&x)[N])
{
    T l[N][N] = {};
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k <= i; ++k)
        {
            T sum = m[i][k];
            for (int q = 0; q < k; ++q)
                sum -= l[i][q] * l[k][q];
            if (i != k) {
                l[i][k] = sum / l[k][k];
            } else if (sum > 0) {
                l[i][i] = std::sqrt(sum);
            } else {
                return false;
            }
        }
    }
    T y[N] = {};
    for (int i = 0; i < N; ++i)
    {
        T sum = b[i];
        for (int q = 0; q < i; ++q)
            sum -= l[i][q] * y[q];
        y[i] = sum / l[i][i];
    }
    for (int i = N - 1; i >= 0; --i)
    {
        T sum = y[i];
        for (int q = i + 1; q < N; ++q)
            sum -= l[q][i] * x[q];
        x[i] = sum / l[i][i];
    }
    return true;
}

// The Gauss-Hermite series of p = (a, c, s, h3, h4) (see
// dmodel_gauss_hermite) fitted to a spectrum of size_z channels, stride
// values apart, by least squares (Levenberg-Marquardt) from p, which it
// updates: as dmodel_gaussian_fit, the steps whose Gaussian (a, c, s) is
// not one of a line of the spectrum (see dmodel_gaussian_in_band) are
// rejected, as those that do not lower the residuals. Return the sum of
// the squared residuals of the fit.
template<typename T> constexpr T
dmodel_gauss_hermite_fit(int size_z, const T* spectrum, int stride, T (&p)[5])
{
    constexpr int MAX_ITERATIONS = 100;
    constexpr T TOLERANCE = T{1e-6};
    T sse = dmodel_gauss_hermite_sse(size_z, spectrum, stride, p);
    T lambda = T{1e-3};
    // The normal equations J^T J and J^T r of the residuals, made again
    // only when the fit moves (a rejected step only raises the damping)
    T jj[5][5] = {}, jr[5] = {};
    bool moved = true;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
    {
        if (moved)
        {
            for (int i = 0; i < 5; ++i) {
                jr[i] = 0;
                for (int k = 0; k < 5; ++k)
                    jj[i][k] = 0;
            }
            for (int z = 0; z < size_z; ++z)
            {
                T j[5];
                const T r = dmodel_gauss_hermite(z, p, j)
                        - spectrum[z * stride];
                for (int i = 0; i < 5; ++i) {
                    jr[i] -= j[i] * r;
                    for (int k = 0; k < 5; ++k)
                        jj[i][k] += j[i] * j[k];
                }
            }
            moved = false;
        }
        T m[5][5];
        for (int i = 0; i < 5; ++i)
            for (int k = 0; k < 5; ++k)
                m[i][k] = jj[i][k] * (i == k ? 1 + lambda : 1);
        T delta[5];
        if (!solve_spd(m, jr, delta)) {
            break;
        }
        T q[5];
        for (int i = 0; i < 5; ++i)
            q[i] = p[i] + delta[i];
        const T sse_new = dmodel_gaussian_in_band(size_z, q[0], q[1], q[2])
                ? dmodel_gauss_hermite_sse(size_z, spectrum, stride, q)
                : NAN;
        if (sse_new < sse) {
            const bool converged = sse - sse_new <= TOLERANCE * sse;
            for (int i = 0; i < 5; ++i)
                p[i] = q[i];
            sse = sse_new;
            lambda /= 10;
            moved = true;
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
    return sse;
}

// The start of a Gaussian fit (channel units) at the brightest channel of
// a spectrum of size_z channels, stride values apart: its value (a), the
// channel (c), and the dispersion of the width at half the value (s; the
// crossings of half the value interpolated between the channels, or the
// ends of the spectrum, and at least half a channel)
template<typename T> constexpr void
dmodel_gaussian_peak(
        int size_z, const T* spectrum, int stride, T& a, T& c, T& s)
{
    int peak = 0;
    for (int z = 1; z < size_z; ++z)
        if (spectrum[z * stride] > spectrum[peak * stride])
            peak = z;
    a = spectrum[peak * stride];
    const T half = a / 2;
    T lo = 0, hi = size_z - 1;
    for (int z = peak - 1; z >= 0; --z)
    {
        const T i = spectrum[z * stride];
        if (i < half) {
            lo = z + (half - i) / (spectrum[(z + 1) * stride] - i);
            break;
        }
    }
    for (int z = peak + 1; z < size_z; ++z)
    {
        const T i = spectrum[z * stride];
        if (i < half) {
            hi = z - (half - i) / (spectrum[(z - 1) * stride] - i);
            break;
        }
    }
    // (the FWHM of a Gaussian is 2 sqrt(2 ln 2) = 2.3548 dispersions)
    c = peak;
    s = std::max((hi - lo) / T{2.3548200450309493}, T{0.5});
}

// The moment maps of a Gaussian fitted to the spectrum of the pixel (x,
// y) of a cube by least squares (see dmodel_gaussian_fit), as moments:
// its flux (order 0), centre (1) and dispersion (2), the only orders. The
// fit is of a Gaussian sampled at the centres of the channels, in channel
// units, from two starts: the moments of the spectrum, and its brightest
// channel (see dmodel_gaussian_peak); that of the smaller residuals is
// kept, that from the moments if they are equal. (In noise, the moments
// of the whole spectrum can start the fit far from the line, where it
// falls on a spike of noise or on the wing of a Gaussian beyond the
// spectrum.) The spectra whose moment 0 is not above the cutoff, and
// those that no Gaussian of a line fits (see dmodel_gaussian_in_band),
// are masked as with the moments (moment 0 is then their flux). The
// channels of the spectrum are stride values apart, as in
// dmodel_mmaps_moments.
//
// With Hermite, the Gaussian is the start of a fit of a Gauss-Hermite
// series (see dmodel_gauss_hermite_fit, from h3 = h4 = 0), whose terms
// are the orders 3 (h3) and 4 (h4); its flux, centre and dispersion are
// those of the series (orders 0 to 2).
template<bool Hermite, typename T> constexpr void
dmodel_mmaps_gaussian(
        int x, int y,
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* spectrum, int stride,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
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

    // The fits: amplitude a, centre c and dispersion s
    T a = m0 / (m2 * std::sqrt(2 * PI<T>));
    T c = m1;
    T s = m2;
    bool valid = std::abs(m0 * step_z) > cutoff;
    if (valid)
    {
        const T sse = dmodel_gaussian_fit(size_z, spectrum, stride, a, c, s);
        T a_peak = 0, c_peak = 0, s_peak = 0;
        dmodel_gaussian_peak(
                size_z, spectrum, stride, a_peak, c_peak, s_peak);
        const T sse_peak = dmodel_gaussian_fit(
                size_z, spectrum, stride, a_peak, c_peak, s_peak);
        if (sse_peak < sse) {
            a = a_peak;
            c = c_peak;
            s = s_peak;
        }
        valid = sse_peak < INFINITY || sse < INFINITY;
    }
    mmaps_m[idx_2d] = valid;

    // The Gauss-Hermite terms
    T h3 = 0, h4 = 0;
    if constexpr (Hermite)
    {
        if (valid)
        {
            T p[5] = {a, c, s, 0, 0};
            dmodel_gauss_hermite_fit(size_z, spectrum, stride, p);
            a = p[0];
            c = p[1];
            s = p[2];
            h3 = p[3];
            h4 = p[4];
        }
    }

    // The moments of the Gaussian, in the units of the spectral axis:
    // moment 0 always (the flux, if no Gaussian fits), the others only if
    // not masked (h3 is of the velocities, odd in them: of the other sign
    // if they fall with the channels)
    for (int m = 0; m < norders; ++m)
    {
        T value = NAN;
        T flux = a * s * std::sqrt(2 * PI<T>);
        if constexpr (Hermite)
            flux *= 1 + h4 * std::sqrt(T{6}) / 4;
        if (orders[m] == 0)
            value = valid ? flux * step_z : m0 * step_z;
        else if (valid && orders[m] == 1)
            value = zero_z + c * step_z;
        else if (valid && orders[m] == 2)
            value = s * step_z;
        else if (valid && orders[m] == 3)
            value = step_z < 0 ? -h3 : h3;
        else if (valid && orders[m] == 4)
            value = h4;
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
