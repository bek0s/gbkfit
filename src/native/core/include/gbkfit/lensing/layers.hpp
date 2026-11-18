#pragma once

#include <algorithm>
#include <cmath>

// The deflections of the layers of the mass of a galaxy laid out on the
// sky (see gbkfit.model.mass.sky and gbkfit.lensing._layers), at one
// point each, without the factors of K = 4 G / c^2 that the callers
// apply: of an elliptical layer (by the integrals of Keeton 2001) and of
// bands of rings (in closed form); thin, or thick, of the integral over
// the height of thin sheets, by a quadrature of each point (see
// lens_height_rule). The functions are constexpr, so that the kernels of
// the host and of cuda (of relaxed constexpr) share them.

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

// The Gauss-Legendre rules of 1 to n nodes over [-1, 1] (see
// _layers.gauss_legendre): those of the rule of m nodes at the offset m
// (m - 1) / 2 of nodes and weights
template<typename T>
struct LensGauss
{
    int n = 0;
    const T* nodes = nullptr;
    const T* weights = nullptr;
};

// The factor of a sheet of no factor (see lens_elliptical_slice)
template<typename T>
struct LensUnit
{
    constexpr T operator()(T) const { return 1; }
};

// Below this d at the end of the integrals of a sheet (see
// lens_elliptical_slice), near the line of a sheet seen close to edge-on,
// they are split at this d: beyond, d^(n + 1/2) changes over decades of
// 1 - w that the tanh-sinh nodes leave unresolved (the deflection across
// a sheet edge-on, 1 pc from its line, was 3e-3 off), and they take a
// Gauss-Legendre quadrature in ln d there, of this many nodes per e-fold,
// in panels of this many nodes at most (to about 2e-7 of the deflection
// of a line, to 1e-8 kpc from it)
constexpr double LENS_LINE_D = 1e-4;
constexpr double LENS_LINE_DENSITY = 4;
constexpr int LENS_LINE_PANEL = 32;

// Add u J_0 and v J_1 of a sheet of an elliptical layer at the point (u,
// v) of its principal frame to ju and jv (its deflection over pi K, see
// _layers._elliptical), times sigma1^2: of the axis ratio q, the table of
// its surface density in its plane (see lens_table), less ln(sigma1) in
// ln R, whose last radius is a_end / sigma1, times factor(a^2) (of the
// semi-major axis a of the sky; called in the order of increasing a): by
// the tanh-sinh quadrature over [0, 1] (nodes) of a point within the
// table, or else mapped onto [0, w_end], where a(w_end) = a_end; and below
// LENS_LINE_D at w_end, split there, with the Gauss-Legendre rules of
// gauss in ln d beyond.
template<typename T, typename Factor = LensUnit<T>> constexpr void
lens_elliptical_slice(
        T u, T v, T q, T log_sigma1, T a_end,
        const T* table, int n, T start, T step,
        const LensNodes<T>& nodes, const LensGauss<T>& gauss,
        T& ju, T& jv, Factor factor = {})
{
    const T e2 = 1 - q * q;
    const T uu = u * u, vv = v * v;
    // (s_end = 1 - w_end: 0 of a point within the table at w = 1, else the
    // root in (0, 1] of e2 u^2 s^2 + b s + c, without the cancellation of
    // 1 - w_end, which near the line of a sheet edge-on is all of it)
    const T aa = a_end * a_end;
    const T c = q * q * (aa - uu) - vv;
    T s_end = 0;
    if (c < 0)
    {
        const T b = vv + (2 * q * q - 1) * uu + e2 * aa;
        s_end = -2 * c
                / (b + std::sqrt(std::max(b * b - 4 * e2 * uu * c, T{0})));
    }
    const T d_end = q * q + e2 * s_end;
    const T start_sigma1 = start + log_sigma1;
    T j0 = 0, j1 = 0;
    if (s_end == 0 && d_end >= T(LENS_LINE_D))
    {
        for (int k = 0; k < nodes.n; ++k)
        {
            const T a2 = nodes.nodes[k] * (uu + vv * nodes.inverse_d[k]);
            if (!(a2 > 0))
                continue;
            const T term = nodes.weights_d[k] * lens_table(
                    table, n, start_sigma1, step, std::log(a2) / 2)
                    * factor(a2);
            j0 += term;
            j1 += term * nodes.inverse_d[k];
        }
        ju += u * j0;
        jv += v * j1;
        return;
    }
    // (the tanh-sinh nodes over [0, 1 - s_top])
    const T s_top = d_end >= T(LENS_LINE_D) ? s_end
            : (T(LENS_LINE_D) - q * q) / e2;
    const T w_top = 1 - s_top;
    for (int k = 0; k < nodes.n; ++k)
    {
        const T w = w_top * nodes.nodes[k];
        const T d = q * q + e2 * (s_top + w_top * nodes.one_minus[k]);
        const T a2 = w * (uu + vv / d);
        if (!(a2 > 0))
            continue;
        const T term = w_top * nodes.weights[k] / std::sqrt(d)
                * lens_table(table, n, start_sigma1, step, std::log(a2) / 2)
                * factor(a2);
        j0 += term;
        j1 += term / d;
    }
    if (d_end < T(LENS_LINE_D))
    {
        // (from d = LENS_LINE_D down to d_end, in ln d: dw = d dln(d) / e2)
        const T top = std::log(T(LENS_LINE_D)), length = top - std::log(d_end);
        const int total = int(std::ceil(T(LENS_LINE_DENSITY) * length));
        const int panels = (total + LENS_LINE_PANEL - 1) / LENS_LINE_PANEL;
        const int m = std::clamp((total + panels - 1) / panels, 3, gauss.n);
        const T width = length / panels;
        const T* x = gauss.nodes + m * (m - 1) / 2;
        const T* weights = gauss.weights + m * (m - 1) / 2;
        for (int p = 0; p < panels; ++p)
        {
            for (int k = 0; k < m; ++k)
            {
                const T d = std::exp(top - width * (p + (x[k] + 1) / 2));
                const T w = 1 - (d - q * q) / e2;
                const T a2 = w * (uu + vv / d);
                if (!(a2 > 0))
                    continue;
                const T term = width / 2 * weights[k] * std::sqrt(d) / e2
                        * lens_table(table, n, start_sigma1, step,
                                     std::log(a2) / 2) * factor(a2);
                j0 += term;
                j1 += term / d;
            }
        }
    }
    ju += u * j0;
    jv += v * j1;
}

// Below this eccentricity, a band takes the series of its integrals (see
// _layers._bands), of this many terms
constexpr double LENS_SERIES_ECCENTRICITY = 0.1;
constexpr int LENS_SERIES_TERMS = 10;

// S + i e a of S = S(a) (see lens_band_slice), as z^2 / (S - i e a) where
// S is near -i e a ((S + i e a)(S - i e a) = S^2 + e^2 a^2 = z^2): the
// larger of the two, without their cancellation (total seen edge-on, near
// the line of a band)
template<typename T> constexpr LensComplex<T>
lens_plus(LensComplex<T> s, LensComplex<T> ea, LensComplex<T> zz)
{
    const LensComplex<T> plus = s + ea, minus = s - ea;
    if (plus.re * plus.re + plus.im * plus.im
            >= minus.re * minus.re + minus.im * minus.im)
        return plus;
    return zz / minus;
}

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
                * lens_log(lens_plus(s2, ea2, zz) / lens_plus(s1, ea1, zz));
        j1 = (a2 * a2 - a1 * a1) * (one / (s1 + s2));
        j2 = (T{1} / (2 * e2 * e)) * (zz * angle)
                - (T{1} / (2 * e2)) * (a2 * s2 - a1 * s1);
        j3 = (T{1} / (3 * e2))
                * (T{2} * (zz * j1) - ((a2 * a2) * s2 - (a1 * a1) * s1));
    }
    alpha = alpha + d0 * j1 + d1 * j2 + d2 * j3;
}

// The most terms of the vertical structure of a thick layer (see
// LensHeights), whose tops cut the rule of its height
constexpr int LENS_HEIGHT_TERMS = 8;

// The vertical structure of a thick layer (see sky.HeightProfile): of
// each of its nterms terms at each of its nknots knots (radii in its
// plane, kpc), the pdf of the height (1/kpc) at ntable heights evenly
// from 0 to the top of the term, cubic between them (as lens_table, of
// the values) and 0 beyond, and the share of the term; of each term, its
// top and its scale height (kpc). Its shape at a knot is the sum over its
// terms of their shares times their pdfs, linear in the radius between
// the knots and held beyond the first and the last. Of a thin layer,
// nterms is 0. The Gauss-Legendre rules of the quadratures of its height
// (see lens_height_rule) and of its sheets near their line (see
// lens_elliptical_slice), of a thin layer too.
template<typename T>
struct LensHeights
{
    int nterms = 0, nknots = 0;
    const T* knots = nullptr;
    const T* shares = nullptr;
    int ntable = 0;
    const T* tables = nullptr;
    const T* tops = nullptr;
    const T* scales = nullptr;
    LensGauss<T> gauss;
};

// The pdf of the height of the term t at the knot k at the height h
template<typename T> constexpr T
lens_height_pdf(const LensHeights<T>& p, int t, int k, T h)
{
    const T top = p.tops[t];
    const T height = std::abs(h);
    if (!(height <= top))
        return 0;
    const int n = p.ntable;
    const T x = height / top * (n - 1);
    const int i = std::clamp(int(x), 1, n - 3);
    const T s = x - i;
    const T* v = p.tables + (1LL * t * p.nknots + k) * n + i - 1;
    // (the Lagrange weights of the nodes -1, 0, 1, 2 at s)
    const T a = s + 1, b = s - 1, c = s - 2;
    return (- s * b * c * v[0] + 3 * a * b * c * v[1]
            - 3 * a * s * c * v[2] + a * s * b * v[3]) / 6;
}

// The shape of a thick layer at the knot k at the height h
template<typename T> constexpr T
lens_height_shape(const LensHeights<T>& p, int k, T h)
{
    T shape = 0;
    for (int t = 0; t < p.nterms; ++t)
        shape += p.shares[1LL * t * p.nknots + k]
                * lens_height_pdf(p, t, k, h);
    return shape;
}

// The quadrature of the height of a thick layer at a point (see
// lens_height_rule): this many nodes per e-fold of the graded length of
// each of its parts, and this many at least
constexpr double LENS_HEIGHT_DENSITY = 5;
constexpr int LENS_HEIGHT_FEWEST = 3;

// The smallest scale of the parts graded from where a point meets the
// line of nodes of a layer, relative to its scale height (see
// lens_height_cuts)
constexpr double LENS_HEIGHT_FLOOR = 1e-2;

// The most cuts of the range of the height of a point: its plane, the
// tops of the terms (on either side) and four of the geometry of a band
constexpr int LENS_HEIGHT_CUTS = 5 + 2 * LENS_HEIGHT_TERMS;

// A height where the quadrature of a point cuts the range of the height,
// where its integrand changes on the scale s (see lens_height_rule)
template<typename T>
struct LensCut
{
    T h, s;
};

// The quadrature over the height h in [-range, range] of a thick layer at
// a point: f(h, w) of each node h and its weight w, but within the hole
// (lo, hi), if lo < hi (where the integrand is 0). The range is cut at the
// cuts within it, and each piece between two is two parts, each graded
// from its end at a cut, where the integrand changes on the scale s of
// the cut (lowered to the distance of any closer cut plus its scale): h =
// end +- s (e^t - 1), of the Gauss-Legendre nodes in t of
// LENS_HEIGHT_DENSITY per e-fold (at least LENS_HEIGHT_FEWEST, at most the
// largest rule), the two parts meeting where their nodes are as dense.
// The parts next to the ends of the range are graded from their cuts.
template<typename T, typename F> constexpr void
lens_height_rule(
        const LensCut<T>* cuts, int ncuts, T range, T lo, T hi,
        const LensGauss<T>& gauss, F&& f)
{
    // (the ends of the range and the cuts within it, in order)
    T hs[LENS_HEIGHT_CUTS + 2] = {}, ss[LENS_HEIGHT_CUTS + 2] = {};
    int np = 0;
    hs[np] = -range;
    ss[np++] = INFINITY;
    for (int i = 0; i < ncuts; ++i)
    {
        if (!(cuts[i].h > -range && cuts[i].h < range))
            continue;
        int j = np++;
        for (; j > 1 && hs[j - 1] > cuts[i].h; --j)
        {
            hs[j] = hs[j - 1];
            ss[j] = ss[j - 1];
        }
        hs[j] = cuts[i].h;
        ss[j] = cuts[i].s;
    }
    hs[np] = range;
    ss[np++] = INFINITY;
    // (each scale lowered to the distance of another cut plus its scale)
    T scales[LENS_HEIGHT_CUTS + 2] = {};
    for (int j = 0; j < np; ++j)
    {
        T s = ss[j];
        for (int k = 1; j > 0 && j < np - 1 && k < np - 1; ++k)
            if (k != j)
                s = std::min(s, std::abs(hs[j] - hs[k]) + ss[k]);
        scales[j] = s;
    }
    const auto part = [&](T end, T other, T scale)
    {
        const T length = std::abs(other - end);
        if (!(length > 0))
            return;
        const T graded = std::log1p(length / scale);
        // (of the constant by value: on the device, a host constant has no
        // address for the reference of clamp)
        const int n = std::clamp(
                int(std::ceil(T(LENS_HEIGHT_DENSITY) * graded)),
                int(LENS_HEIGHT_FEWEST), gauss.n);
        const T* x = gauss.nodes + n * (n - 1) / 2;
        const T* w = gauss.weights + n * (n - 1) / 2;
        const T sign = other > end ? 1 : -1;
        for (int k = 0; k < n; ++k)
        {
            const T e = std::exp(graded / 2 * (x[k] + 1));
            f(end + sign * scale * (e - 1), graded / 2 * w[k] * scale * e);
        }
    };
    const T tiny = 1e-12 * range;
    for (int j = 0; j + 1 < np; ++j)
    {
        const T a = hs[j], b = hs[j + 1];
        const T middle = (a + b) / 2;
        if (b - a <= tiny || (lo < hi && middle > lo && middle < hi))
            continue;
        // (where both parts are as dense: s_a + x = s_b + (b - a - x))
        const T split = std::isinf(scales[j]) ? T(0)
                : std::isinf(scales[j + 1]) ? b - a
                : std::clamp((b - a + scales[j + 1] - scales[j]) / 2,
                             T(0), b - a);
        part(a, a + split, scales[j]);
        part(b, a + split, scales[j + 1]);
    }
}

// The cuts of the quadrature of the height of a point that the vertical
// structure of a thick layer gives (see lens_height_rule): its plane, of
// the scale height of its thinnest term, and the tops of its terms below
// the highest, of their scale heights; their number, and the range (the
// highest top) and that scale height
template<typename T> constexpr int
lens_height_cuts(
        const LensHeights<T>& p, LensCut<T>* cuts, T& range, T& scale)
{
    range = 0;
    scale = INFINITY;
    for (int t = 0; t < p.nterms; ++t)
    {
        range = std::max(range, p.tops[t]);
        scale = std::min(scale, p.scales[t]);
    }
    int n = 0;
    cuts[n++] = {T(0), scale};
    for (int t = 0; t < p.nterms; ++t)
    {
        if (p.tops[t] < range)
        {
            cuts[n++] = {p.tops[t], p.scales[t]};
            cuts[n++] = {-p.tops[t], p.scales[t]};
        }
    }
    return n;
}

// The geometry of a sheet of a thick layer at a point, as the sheet moves
// along the normal of the layer with the height h: of the point at (u0,
// v0) of the principal frame of the sheet in its plane and the normal at
// (nu, nv) there, of the axis ratio q, the semi-major axis a of the
// ellipse of the point is a^2 = a_min^2 + (A^2 / q^2) (h - h0)^2, A^2 =
// q^2 nu^2 + nv^2: complex, singular at h0 +- i c, c = q a_min / A (seen
// edge-on, a jump at h0, where the point meets the line of nodes). None
// (A = 0) of a layer seen face-on, whose sheets are all at the point.
template<typename T>
struct LensSheetPath
{
    bool moves = false;
    T root = 0, h0 = 0, a_min = 0, c = 0;

    constexpr LensSheetPath(T u0, T v0, T nu, T nv, T q)
    {
        const T a2 = q * q * nu * nu + nv * nv;
        if (!(a2 > 0))
            return;
        moves = true;
        root = std::sqrt(a2);
        h0 = (q * q * u0 * nu + v0 * nv) / a2;
        a_min = std::abs(u0 * nv - v0 * nu) / root;
        c = q * a_min / root;
    }
};

// An elliptical layer (see _layers._elliptical): its frame (the sine and
// cosine of the position angle of its major axis, its axis ratio q and
// sigma1), its centre, the normal of its plane, the table of the surface
// density in its plane (n values of the logarithm at the radii exp(start
// + k step)), the nodes of the quadrature of its sheets (see LensNodes)
// and its vertical structure (none, thin)
template<typename T>
struct LensEllipse
{
    T sin_p, cos_p, q, sigma1;
    T cx, cy, nx, ny;
    int n;
    T start, step;
    const T* table;
    int nnodes;
    const T* nodes;
    LensHeights<T> heights;
};

// The shape of a thick elliptical layer of knots at the height h (see
// LensHeights), as the factor of its sheet there (see
// lens_elliptical_slice) at a^2, a / sigma1 the radius: between the two
// knots around the radius, moving on from those before (as the sheet
// calls it, in the order of increasing a)
template<typename T>
struct LensKnotShape
{
    const LensHeights<T>* p;
    T h, inverse_sigma1;
    int k = 0;
    T left, right;

    constexpr LensKnotShape(const LensHeights<T>& heights, T height, T sigma1)
        : p(&heights), h(height), inverse_sigma1(1 / sigma1),
          left(lens_height_shape(heights, 0, height)),
          right(lens_height_shape(heights, 1, height)) {}

    constexpr T operator()(T a2)
    {
        const T radius = std::sqrt(a2) * inverse_sigma1;
        while (k + 2 < p->nknots && radius > p->knots[k + 1])
        {
            ++k;
            left = right;
            right = lens_height_shape(*p, k + 1, h);
        }
        const T t = std::clamp(
                (radius - p->knots[k]) / (p->knots[k + 1] - p->knots[k]),
                T(0), T(1));
        return left + t * (right - left);
    }
};

// The deflection over pi K of an elliptical layer at the point (x, y),
// into alpha_x and alpha_y: thin, u J_0 and v J_1 of its sheet at the
// point, along x and y; thick, the integral over the height of those of
// its sheet at each height (at the point from its centre moved by the
// height along the normal) times its shape there (see lens_height_rule,
// cut where the point meets its line of nodes, at the scale c of
// LensSheetPath, at least LENS_HEIGHT_FLOOR and at most the scale height)
template<typename T> constexpr void
lens_elliptical_point(
        T x, T y, const LensEllipse<T>& e, T& alpha_x, T& alpha_y)
{
    const LensNodes<T> nodes(e.nnodes, e.nodes);
    const T a_end = e.sigma1 * std::exp(e.start + e.step * (e.n - 1));
    const T log_sigma1 = std::log(e.sigma1);
    const T dx = x - e.cx, dy = y - e.cy;
    const T u0 = - dx * e.sin_p + dy * e.cos_p;
    const T v0 = - dx * e.cos_p - dy * e.sin_p;
    T ju = 0, jv = 0;
    const LensHeights<T>& p = e.heights;
    if (p.nterms == 0)
    {
        lens_elliptical_slice(
                u0, v0, e.q, log_sigma1, a_end, e.table, e.n, e.start,
                e.step, nodes, p.gauss, ju, jv);
    }
    else
    {
        // (the normal in the principal frame)
        const T nu = - e.nx * e.sin_p + e.ny * e.cos_p;
        const T nv = - e.nx * e.cos_p - e.ny * e.sin_p;
        LensCut<T> cuts[LENS_HEIGHT_CUTS];
        T range = 0, scale = 0;
        int ncuts = lens_height_cuts(p, cuts, range, scale);
        const LensSheetPath<T> path(u0, v0, nu, nv, e.q);
        if (path.moves)
            cuts[ncuts++] = {path.h0, std::min(std::max(
                    path.c, T(LENS_HEIGHT_FLOOR) * scale), scale)};
        lens_height_rule(cuts, ncuts, range, T(0), T(0), p.gauss,
                         [&](T h, T w)
        {
            const T u = u0 - h * nu, v = v0 - h * nv;
            T jk = 0, jl = 0;
            if (p.nknots == 1)
            {
                const T shape = lens_height_shape(p, 0, h);
                if (shape == 0)
                    return;
                lens_elliptical_slice(
                        u, v, e.q, log_sigma1, a_end, e.table, e.n, e.start,
                        e.step, nodes, p.gauss, jk, jl);
                ju += w * shape * jk;
                jv += w * shape * jl;
                return;
            }
            lens_elliptical_slice(
                    u, v, e.q, log_sigma1, a_end, e.table, e.n, e.start,
                    e.step, nodes, p.gauss, jk, jl,
                    LensKnotShape<T>(p, h, e.sigma1));
            ju += w * jk;
            jv += w * jl;
        });
    }
    ju /= e.sigma1 * e.sigma1;
    jv /= e.sigma1 * e.sigma1;
    alpha_x = - ju * e.sin_p - jv * e.cos_p;
    alpha_y = ju * e.cos_p - jv * e.sin_p;
}

// Bands of rings (see _layers._bands): of each of nbands, its centre
// (centres, (nbands, 2)), its frame (frames, (nbands, 5): the sine and
// cosine of the position angle of its major axis, its axis ratio q and
// its inner and outer semi-major axes), the normal of its plane (normals,
// (nbands, 2)) and the coefficients of q Sigma_sky(a) (coefficients,
// (nbands, 2, 3)); thin, of the first row, and thick (of their vertical
// structure, see LensHeights), of the sum of the rows times the shapes at
// the knots of their indices (knots, (nbands, 2))
template<typename T>
struct LensBands
{
    int nbands;
    const T* centres;
    const T* frames;
    const T* normals;
    const T* coefficients;
    const int* knots;
    LensHeights<T> heights;
};

// Add w times the deflection over 2 pi K of a band of the coefficients
// d0, d1 and d2 at the point (dx, dy) from its centre to alpha_x and
// alpha_y (see lens_band_slice; frame: its row of LensBands)
template<typename T> constexpr void
lens_band_add(
        T dx, T dy, const T* frame, T d0, T d1, T d2, T w,
        T& alpha_x, T& alpha_y)
{
    const T u = - dx * frame[0] + dy * frame[1];
    const T v = - dx * frame[1] - dy * frame[0];
    LensComplex<T> alpha;
    lens_band_slice(u, v, frame[2], frame[3], frame[4], d0, d1, d2, alpha);
    // (alpha_u - i alpha_v, along x and y)
    const T au = alpha.re, av = -alpha.im;
    alpha_x += w * (- au * frame[0] - av * frame[1]);
    alpha_y += w * (au * frame[1] - av * frame[0]);
}

// Add the deflection over 2 pi K of a thick band at the point (dx, dy)
// from its centre to alpha_x and alpha_y (see LensBands, of its rows):
// the integral over the height of its sheet at each height (see
// lens_height_rule), cut where the ellipse of the point crosses its
// edges, and where the point meets its line of nodes, within the band or
// outside (cut at its distance from the band), without its hole (where
// the point is within its inner edge, and it bends nothing)
template<typename T> constexpr void
lens_band_thick(
        T dx, T dy, const T* frame, const T* normal, const T* c,
        const int* knots, const LensHeights<T>& p, T& alpha_x, T& alpha_y)
{
    const T q = frame[2], a1 = frame[3], a2 = frame[4];
    const T u0 = - dx * frame[0] + dy * frame[1];
    const T v0 = - dx * frame[1] - dy * frame[0];
    const T nu = - normal[0] * frame[0] + normal[1] * frame[1];
    const T nv = - normal[0] * frame[1] - normal[1] * frame[0];
    LensCut<T> cuts[LENS_HEIGHT_CUTS];
    T range = 0, scale = 0;
    int ncuts = lens_height_cuts(p, cuts, range, scale);
    T lo = 0, hi = 0;
    const LensSheetPath<T> path(u0, v0, nu, nv, q);
    if (path.moves)
    {
        const T floor = T(LENS_HEIGHT_FLOOR) * scale;
        const T h0 = path.h0, a_min = path.a_min;
        if (a_min < a2)
        {
            // (the heights where the ellipse of the point is each edge)
            const T d2 = q * std::sqrt(a2 * a2 - a_min * a_min) / path.root;
            const T s2 = std::max(std::hypot(d2, path.c), floor);
            cuts[ncuts++] = {h0 - d2, s2};
            cuts[ncuts++] = {h0 + d2, s2};
            if (a_min < a1)
            {
                const T d1 = q * std::sqrt(a1 * a1 - a_min * a_min)
                        / path.root;
                const T s1 = std::max(std::hypot(d1, path.c), floor);
                cuts[ncuts++] = {h0 - d1, s1};
                cuts[ncuts++] = {h0 + d1, s1};
                lo = h0 - d1;
                hi = h0 + d1;
            }
            else
            {
                cuts[ncuts++] = {h0, std::max(path.c, floor)};
            }
        }
        else
        {
            const T e = std::sqrt(std::max(1 - q * q, T(0)));
            cuts[ncuts++] = {h0, std::max(std::max(
                    (a_min - e * a2) / path.root, path.c), floor)};
        }
    }
    const int below = knots[0], above = knots[1];
    lens_height_rule(cuts, ncuts, range, lo, hi, p.gauss, [&](T h, T w)
    {
        T d0, d1, d2;
        if (below == above)
        {
            const T shape = lens_height_shape(p, below, h);
            d0 = shape * (c[0] + c[3]);
            d1 = shape * (c[1] + c[4]);
            d2 = shape * (c[2] + c[5]);
        }
        else
        {
            const T lower = lens_height_shape(p, below, h);
            const T upper = lens_height_shape(p, above, h);
            d0 = lower * c[0] + upper * c[3];
            d1 = lower * c[1] + upper * c[4];
            d2 = lower * c[2] + upper * c[5];
        }
        if (d0 == 0 && d1 == 0 && d2 == 0)
            return;
        lens_band_add(dx - h * normal[0], dy - h * normal[1], frame,
                      d0, d1, d2, w, alpha_x, alpha_y);
    });
}

// The deflection over 2 pi K of bands of rings at the point (x, y), into
// alpha_x and alpha_y: the sum of those of the bands, thin (of their
// first row of coefficients) or thick (see lens_band_thick)
template<typename T> constexpr void
lens_band_point(T x, T y, const LensBands<T>& b, T& alpha_x, T& alpha_y)
{
    T ax = 0, ay = 0;
    for (int k = 0; k < b.nbands; ++k)
    {
        const T* frame = b.frames + 5 * k;
        const T* c = b.coefficients + 6LL * k;
        const T dx = x - b.centres[2 * k];
        const T dy = y - b.centres[2 * k + 1];
        if (b.heights.nterms == 0)
            lens_band_add(dx, dy, frame, c[0], c[1], c[2], T(1), ax, ay);
        else
            lens_band_thick(dx, dy, frame, b.normals + 2 * k, c,
                            b.knots + 2 * k, b.heights, ax, ay);
    }
    alpha_x = ax;
    alpha_y = ay;
}

} // namespace gbkfit
