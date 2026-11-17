#pragma once

#include <algorithm>
#include <cmath>

// The deflections of the layers of the mass of a galaxy laid out on the
// sky (see gbkfit.model.mass.sky and gbkfit.lensing._layers), at one
// point each, without the factors of K = 4 G / c^2 that the callers
// apply: of an elliptical layer (by the integrals of Keeton 2001) and of
// bands of rings (in closed form). The functions are constexpr, so that
// the kernels of the host and of cuda (of relaxed constexpr) share them.

namespace gbkfit {

// A complex number of plain arithmetic (of no checks for infinities and
// NaN, which these kernels do not meet, unlike std::complex, whose
// multiplication and division call the checked routines of the library)
template<typename T>
struct LensComplex
{
    T re = 0, im = 0;
};

template<typename T> constexpr LensComplex<T>
operator+(LensComplex<T> a, LensComplex<T> b)
{
    return {a.re + b.re, a.im + b.im};
}

template<typename T> constexpr LensComplex<T>
operator-(LensComplex<T> a, LensComplex<T> b)
{
    return {a.re - b.re, a.im - b.im};
}

template<typename T> constexpr LensComplex<T>
operator*(LensComplex<T> a, LensComplex<T> b)
{
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template<typename T> constexpr LensComplex<T>
operator*(T a, LensComplex<T> b)
{
    return {a * b.re, a * b.im};
}

template<typename T> constexpr LensComplex<T>
operator/(LensComplex<T> a, LensComplex<T> b)
{
    const T norm = b.re * b.re + b.im * b.im;
    return {(a.re * b.re + a.im * b.im) / norm,
            (a.im * b.re - a.re * b.im) / norm};
}

// The principal square root (of a cut along the negative reals)
template<typename T> constexpr LensComplex<T>
lens_sqrt(LensComplex<T> z)
{
    const T r = std::sqrt(z.re * z.re + z.im * z.im);
    if (r == 0)
        return {0, 0};
    if (z.re >= 0) {
        const T t = std::sqrt((r + z.re) / 2);
        return {t, z.im / (2 * t)};
    }
    const T t = std::sqrt((r - z.re) / 2);
    return {std::abs(z.im) / (2 * t), std::copysign(t, z.im)};
}

// The principal logarithm
template<typename T> constexpr LensComplex<T>
lens_log(LensComplex<T> z)
{
    return {std::log(std::sqrt(z.re * z.re + z.im * z.im)),
            std::atan2(z.im, z.re)};
}

// The function of a table of the logarithms of its values at the radii
// exp(start + k step), k from 0 to n - 1 (n >= 4), at the logarithm of a
// radius: cubic in ln R between them (4-point Lagrange, of the stencil k -
// 1 to k + 2 of the interval k, clamped to the table), and beyond the
// ends the power laws of the two first and the two last radii (see
// LogTable).
template<typename T> constexpr T
lens_table(const T* values, int n, T start, T step, T log_radius)
{
    const T x = (log_radius - start) / step;
    T value;
    if (x < 0) {
        value = values[0] + x * (values[1] - values[0]);
    } else if (x > n - 1) {
        value = values[n - 1]
                + (x - (n - 1)) * (values[n - 1] - values[n - 2]);
    } else {
        const int i = std::clamp(int(x), 1, n - 3);
        const T t = x - i;
        const T* v = values + i - 1;
        // (the Lagrange weights of the nodes -1, 0, 1, 2 at t)
        const T a = t + 1, b = t - 1, c = t - 2;
        value = (- t * b * c * v[0] + 3 * a * b * c * v[1]
                 - 3 * a * t * c * v[2] + a * t * b * v[3]) / 6;
    }
    return std::exp(value);
}

// The tanh-sinh quadrature of the integrals of an elliptical layer over
// [0, 1], of n nodes, in a table of 5 rows of n values (see
// _layers._elliptical): of each node, w, 1 - w and its weight, and of the
// integrals of a point within the tables of the layer (over all of [0,
// 1], see lens_elliptical_slice), 1 / d and the weight over sqrt(d), d =
// q^2 + (1 - q^2) (1 - w), of the axis ratio q of the layer
template<typename T>
struct LensNodes
{
    int n = 0;
    const T* nodes = nullptr;
    const T* one_minus = nullptr;
    const T* weights = nullptr;
    const T* inverse_d = nullptr;
    const T* weights_d = nullptr;

    constexpr LensNodes(int count, const T* table)
        : n(count), nodes(table), one_minus(table + count),
          weights(table + 2 * count), inverse_d(table + 3 * count),
          weights_d(table + 4 * count) {}
};

// Add u J_0 and v J_1 of a slice of an elliptical layer at the point (u,
// v) of its principal frame to ju and jv (its deflection over pi K, see
// _layers._elliptical), times sigma1^2: of the axis ratio q, the table of
// its surface density in its plane (see lens_table), less ln(sigma1) in
// ln R, whose last radius is a_end / sigma1, and the nodes of the
// tanh-sinh quadrature over [0, 1], mapped onto [0, w_end].
template<typename T> constexpr void
lens_elliptical_slice(
        T u, T v, T q, T log_sigma1, T a_end,
        const T* table, int n, T start, T step,
        const LensNodes<T>& nodes, T& ju, T& jv)
{
    const T e2 = 1 - q * q;
    const T uu = u * u, vv = v * v;
    // (w_end: a(w) = a_end, the root in (0, 1] of e2 u^2 w^2 - (u^2 + v^2
    // + e2 a_end^2) w + a_end^2)
    const T b = uu + vv + e2 * a_end * a_end;
    const T root = std::sqrt(
            std::max(b * b - 4 * e2 * uu * a_end * a_end, T{0}));
    const T w_end = b > 0 ? std::min(T{1}, 2 * a_end * a_end / (b + root))
                          : T{1};
    const T start_sigma1 = start + log_sigma1;
    T j0 = 0, j1 = 0;
    if (w_end == 1)
    {
        for (int k = 0; k < nodes.n; ++k)
        {
            const T a2 = nodes.nodes[k] * (uu + vv * nodes.inverse_d[k]);
            if (!(a2 > 0))
                continue;
            const T term = nodes.weights_d[k] * lens_table(
                    table, n, start_sigma1, step, std::log(a2) / 2);
            j0 += term;
            j1 += term * nodes.inverse_d[k];
        }
    }
    else
    {
        for (int k = 0; k < nodes.n; ++k)
        {
            const T w = w_end * nodes.nodes[k];
            const T d = q * q
                    + e2 * ((1 - w_end) + w_end * nodes.one_minus[k]);
            const T a2 = w * (uu + vv / d);
            if (!(a2 > 0))
                continue;
            const T term = w_end * nodes.weights[k] / std::sqrt(d)
                    * lens_table(table, n, start_sigma1, step,
                                 std::log(a2) / 2);
            j0 += term;
            j1 += term / d;
        }
    }
    ju += u * j0;
    jv += v * j1;
}

// Below this eccentricity, a band takes the series of its integrals (see
// _layers._bands), of this many terms
constexpr double LENS_SERIES_ECCENTRICITY = 0.1;
constexpr int LENS_SERIES_TERMS = 10;

// Add alpha* = alpha_u - i alpha_v over 2 pi K of a band at the point z =
// u + i v of its principal frame to alpha (see _layers._bands): of the
// axis ratio q, between the semi-major axes a1 and a_out, of q
// Sigma_sky(a) = d0 + d1 a + d2 a^2; nothing inside the band.
template<typename T> constexpr void
lens_band_slice(
        T u, T v, T q, T a1, T a_out, T d0, T d1, T d2,
        LensComplex<T>& alpha)
{
    // (the ellipse of the point; seen edge-on, q = 0, that of a point off
    // its line is infinite)
    const T a_ray = q > 0 ? std::sqrt(u * u + v * v / (q * q))
                          : (v == 0 ? std::abs(u) : INFINITY);
    const T a2 = std::min(a_out, a_ray);
    if (!(a2 > a1))
        return;
    const LensComplex<T> z{u, v};
    const LensComplex<T> zz = z * z;
    const T e2 = 1 - q * q;
    const T e = std::sqrt(e2);
    LensComplex<T> j1, j2, j3;
    if (e < T(LENS_SERIES_ECCENTRICITY))
    {
        // 1 / S = sum_k binom(2k, k) / 4^k e^2k a^2k / z^(2k + 1)
        const LensComplex<T> ratio = e2 * (LensComplex<T>{1, 0} / zz);
        LensComplex<T> factor = LensComplex<T>{1, 0} / z;
        T binomial = 1;
        // (a^(2k + 2) at each end)
        T power2 = a2 * a2, power1 = a1 * a1;
        for (int k = 0; k < LENS_SERIES_TERMS; ++k)
        {
            const T p = 2 * k;
            j1 = j1 + (binomial * (power2 - power1) / (p + 2)) * factor;
            j2 = j2 + (binomial * (power2 * a2 - power1 * a1) / (p + 3))
                    * factor;
            j3 = j3 + (binomial * (power2 * a2 * a2 - power1 * a1 * a1)
                       / (p + 4)) * factor;
            // (binom(2k + 2, k + 1) / 4^(k + 1) of that of k)
            binomial *= T(2 * k + 1) / T(2 * k + 2);
            factor = factor * ratio;
            power2 *= a2 * a2;
            power1 *= a1 * a1;
        }
    }
    else
    {
        // S(a) = z sqrt(1 - (e a / z)^2), of the principal root: the branch
        // of z at infinity, cut on the segment between the foci
        const LensComplex<T> inverse_z = LensComplex<T>{1, 0} / z;
        const LensComplex<T> t1 = (e * a1) * inverse_z;
        const LensComplex<T> t2 = (e * a2) * inverse_z;
        const LensComplex<T> one{1, 0};
        const LensComplex<T> s1 = z * lens_sqrt(one - t1 * t1);
        const LensComplex<T> s2 = z * lens_sqrt(one - t2 * t2);
        const LensComplex<T> ea1{0, e * a1}, ea2{0, e * a2};
        // (-i ln((S2 + i e a2) / (S1 + i e a1)))
        const LensComplex<T> angle = LensComplex<T>{0, -1}
                * lens_log((s2 + ea2) / (s1 + ea1));
        j1 = (a2 * a2 - a1 * a1) * (one / (s1 + s2));
        j2 = (T{1} / (2 * e2 * e)) * (zz * angle)
                - (T{1} / (2 * e2)) * (a2 * s2 - a1 * s1);
        j3 = (T{1} / (3 * e2))
                * (T{2} * (zz * j1) - ((a2 * a2) * s2 - (a1 * a1) * s1));
    }
    alpha = alpha + d0 * j1 + d1 * j2 + d2 * j3;
}

// An elliptical layer (see _layers._elliptical): its frame (the sine and
// cosine of the position angle of its major axis, its axis ratio q and
// sigma1), its centre, the normal of its plane, its slices (nslices rows
// of n values of the logarithm of the surface density in its plane, at the
// radii exp(start + k step), and their heights) and the nodes of the
// quadrature (see LensNodes)
template<typename T>
struct LensEllipse
{
    T sin_p, cos_p, q, sigma1;
    T cx, cy, nx, ny;
    int nslices, n;
    T start, step;
    const T* tables;
    const T* heights;
    int nnodes;
    const T* nodes;
};

// The deflection over pi K of an elliptical layer at the point (x, y),
// into alpha_x and alpha_y: of each slice, u J_0 and v J_1 at the point
// from its centre moved by its height along the normal, along x and y
template<typename T> constexpr void
lens_elliptical_point(
        T x, T y, const LensEllipse<T>& e, T& alpha_x, T& alpha_y)
{
    const LensNodes<T> nodes(e.nnodes, e.nodes);
    const T a_end = e.sigma1 * std::exp(e.start + e.step * (e.n - 1));
    const T log_sigma1 = std::log(e.sigma1);
    T ju = 0, jv = 0;
    for (int s = 0; s < e.nslices; ++s)
    {
        const T dx = x - e.cx - e.heights[s] * e.nx;
        const T dy = y - e.cy - e.heights[s] * e.ny;
        const T u = - dx * e.sin_p + dy * e.cos_p;
        const T v = - dx * e.cos_p - dy * e.sin_p;
        lens_elliptical_slice(
                u, v, e.q, log_sigma1, a_end, e.tables + 1LL * s * e.n, e.n,
                e.start, e.step, nodes, ju, jv);
    }
    ju /= e.sigma1 * e.sigma1;
    jv /= e.sigma1 * e.sigma1;
    alpha_x = - ju * e.sin_p - jv * e.cos_p;
    alpha_y = ju * e.cos_p - jv * e.sin_p;
}

// Bands of rings (see _layers._bands): of each of nbands, its centre
// (centres, (nbands, 2)), its frame (frames, (nbands, 5): the sine and
// cosine of the position angle of its major axis, its axis ratio q and
// its inner and outer semi-major axes) and the normal of its plane
// (normals, (nbands, 2)); and of each of their nslices slices, its height
// and the coefficients of q Sigma_sky(a) of each band (coefficients,
// (nslices, nbands, 3))
template<typename T>
struct LensBands
{
    int nbands;
    const T* centres;
    const T* frames;
    const T* normals;
    int nslices;
    const T* coefficients;
    const T* heights;
};

// The deflection over 2 pi K of bands of rings at the point (x, y), into
// alpha_x and alpha_y: the sum of each band in each slice, at the point
// from its centre moved by the height of the slice along its normal
template<typename T> constexpr void
lens_band_point(T x, T y, const LensBands<T>& b, T& alpha_x, T& alpha_y)
{
    T ax = 0, ay = 0;
    for (int s = 0; s < b.nslices; ++s)
    {
    for (int k = 0; k < b.nbands; ++k)
    {
        const T* frame = b.frames + 5 * k;
        const T dx = x - b.centres[2 * k] - b.heights[s] * b.normals[2 * k];
        const T dy = y - b.centres[2 * k + 1]
                - b.heights[s] * b.normals[2 * k + 1];
        const T u = - dx * frame[0] + dy * frame[1];
        const T v = - dx * frame[1] - dy * frame[0];
        const T* c = b.coefficients + 3LL * (1LL * s * b.nbands + k);
        LensComplex<T> alpha;
        lens_band_slice(
                u, v, frame[2], frame[3], frame[4], c[0], c[1], c[2], alpha);
        // (alpha_u - i alpha_v, along x and y)
        const T au = alpha.re, av = -alpha.im;
        ax += - au * frame[0] - av * frame[1];
        ay += au * frame[1] - av * frame[0];
    }
    }
    alpha_x = ax;
    alpha_y = ay;
}

} // namespace gbkfit
