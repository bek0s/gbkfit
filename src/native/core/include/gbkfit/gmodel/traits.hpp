#pragma once

#include <cassert>

#include "gbkfit/math/math.hpp"

namespace gbkfit {

// Density polar traits
constexpr int RPT_UID_UNIFORM = 1;
constexpr int RPT_UID_EXP = 2;
constexpr int RPT_UID_GAUSS = 3;
constexpr int RPT_UID_GGAUSS = 4;
constexpr int RPT_UID_LORENTZ = 5;
constexpr int RPT_UID_MOFFAT = 6;
constexpr int RPT_UID_SECH2 = 7;
constexpr int RPT_UID_MIXTURE_EXP = 51;
constexpr int RPT_UID_MIXTURE_GAUSS = 52;
constexpr int RPT_UID_MIXTURE_GGAUSS = 53;
constexpr int RPT_UID_MIXTURE_MOFFAT = 54;
constexpr int RPT_UID_NW_UNIFORM = 101;
constexpr int RPT_UID_NW_HARMONIC = 102;
constexpr int RPT_UID_NW_DISTORTION = 103;
constexpr int RPT_UID_SPIRAL = 104;
constexpr int RPT_UID_MAP = 151;

// Density height traits
constexpr int RHT_UID_UNIFORM = 1;
constexpr int RHT_UID_EXP = 2;
constexpr int RHT_UID_GAUSS = 3;
constexpr int RHT_UID_GGAUSS = 4;
constexpr int RHT_UID_LORENTZ = 5;
constexpr int RHT_UID_MOFFAT = 6;
constexpr int RHT_UID_SECH2 = 7;

// Velocity polar traits: the shapes of their speeds
constexpr int VPT_UID_UNIFORM = 1;
constexpr int VPT_UID_ARCTAN = 2;
constexpr int VPT_UID_BOISSIER = 3;
constexpr int VPT_UID_EPINAT = 4;
constexpr int VPT_UID_LRAMP = 5;
constexpr int VPT_UID_TANH = 6;
constexpr int VPT_UID_POLYEX = 7;
constexpr int VPT_UID_RIX = 8;
constexpr int VPT_UID_COURTEAU = 9;
constexpr int VPT_UID_BRANDT = 10;
constexpr int VPT_UID_ISO = 11;
constexpr int VPT_UID_NFW = 12;
constexpr int VPT_UID_WEDGE = 13;
constexpr int VPT_UID_BAR = 14;
constexpr int VPT_UID_NW_UNIFORM = 101;
constexpr int VPT_UID_NW_HARMONIC = 102;
constexpr int VPT_UID_NW_BAR = 103;
constexpr int VPT_UID_SPIRAL = 104;
constexpr int VPT_UID_TABLE = 105;

// Velocity polar traits: the directions of their speeds
constexpr int VPT_DIRECTION_TAN = 1;
constexpr int VPT_DIRECTION_RAD = 2;
constexpr int VPT_DIRECTION_VER = 3;
constexpr int VPT_DIRECTION_LOS = 4;
constexpr int VPT_DIRECTION_OUT = 5;
constexpr int VPT_DIRECTION_PLANE = 6;
constexpr int VPT_DIRECTION_AWAY = 7;

// Velocity height traits
constexpr int VHT_UID_ONE = 1;
constexpr int VHT_UID_LINEAR = 2;
constexpr int VHT_UID_EXP = 3;
constexpr int VHT_UID_GAUSS = 4;
constexpr int VHT_UID_MASS = 5;

// Dispersion polar traits
constexpr int DPT_UID_UNIFORM = 1;
constexpr int DPT_UID_EXP = 2;
constexpr int DPT_UID_GAUSS = 3;
constexpr int DPT_UID_GGAUSS = 4;
constexpr int DPT_UID_LORENTZ = 5;
constexpr int DPT_UID_MOFFAT = 6;
constexpr int DPT_UID_SECH2 = 7;
constexpr int DPT_UID_MIXTURE_EXP = 51;
constexpr int DPT_UID_MIXTURE_GAUSS = 52;
constexpr int DPT_UID_MIXTURE_GGAUSS = 53;
constexpr int DPT_UID_MIXTURE_MOFFAT = 54;
constexpr int DPT_UID_NW_UNIFORM = 101;
constexpr int DPT_UID_NW_HARMONIC = 102;
constexpr int DPT_UID_NW_DISTORTION = 103;

// Dispersion polar traits: the directions of their dispersions (an
// isotropic trait adds to every direction; see gmodel_disk_dispersion)
constexpr int DPT_DIRECTION_ISO = 0;
constexpr int DPT_DIRECTION_RAD = 1;
constexpr int DPT_DIRECTION_TAN = 2;
constexpr int DPT_DIRECTION_VER = 3;

// Dispersion height traits
constexpr int DHT_UID_ONE = 1;
constexpr int DHT_UID_LINEAR = 2;
constexpr int DHT_UID_EXP = 3;
constexpr int DHT_UID_GAUSS = 4;

// Vertical distortion polar traits
constexpr int ZPT_UID_NW_UNIFORM = 101;
constexpr int ZPT_UID_NW_HARMONIC = 102;

// Selection polar traits
constexpr int SPT_UID_AZRANGE = 1;
constexpr int SPT_UID_RRANGE = 2;
constexpr int SPT_UID_NW_AZRANGE = 101;

// The node-wise value (of the values ydata[offset + i * stride] at the
// nodes xdata) at x, in the interval of the nodes that ends at the node
// idx (see ring_index): linear in it, and beyond the last node, that at
// the last node (a component keeps its node-wise values beyond its last
// node; its density traits have tails, see rp_trait_tail)
template<typename T> constexpr T
nodewise(T x, int idx, const T* xdata, const T* ydata, int offset, int stride)
{
    return lerp(std::min(x, xdata[idx]), idx, xdata, ydata, offset, stride);
}

template<typename T> constexpr T
nodewise(T x, int idx, const T* xdata, const T* ydata)
{
    return nodewise(x, idx, xdata, ydata, 0, 1);
}

template<typename T> constexpr void
p_trait_uniform(T& out, const T* params)
{
    out = params[0];
}

template<typename T> constexpr void
p_trait_exponential(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = exponential_1d_fun<T>(r, a, 0, s);
}

template<typename T> constexpr void
p_trait_gauss(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = gauss_1d_fun<T>(r, a, 0, s);
}

template<typename T> constexpr void
p_trait_ggauss(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    T b = params[2];
    out = ggauss_1d_fun<T>(r, a, 0, s, b);
}

template<typename T> constexpr void
p_trait_lorentz(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = lorentz_1d_fun<T>(r, a, 0, s);
}

template<typename T> constexpr void
p_trait_moffat(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    T b = params[2];
    out = moffat_1d_fun<T>(r, a, 0, s, b);
}

template<typename T> constexpr void
p_trait_sech2(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = sech2_1d_fun<T>(r, a, 0, s);
}

template<auto FUN, typename T> constexpr void
p_trait_mixture_1p(T& out, T r, T theta, const T* consts, const T* params)
{
    T res = 0;
    int nblobs = std::rint(consts[0]);
    for(int i = 0; i < nblobs; ++i)
    {
        T blob_r = params[         i];  // polar coord (radius)
        T blob_t = params[1*nblobs+i];  // polar coord (angle)
        T blob_a = params[2*nblobs+i];  // amplitude
        T blob_s = params[3*nblobs+i];  // scale
        T blob_q = params[4*nblobs+i];  // axis ratio
        T blob_p = params[5*nblobs+i];  // blob roll (relative to blob_t)
        blob_t *= DEG_TO_RAD<T>;
        blob_p *= DEG_TO_RAD<T>;
        T xn = r * std::cos(theta) - blob_r * std::cos(blob_t);
        T yn = r * std::sin(theta) - blob_r * std::sin(blob_t);
        transform_lh_rotate_z(xn, yn, xn, yn, blob_t + blob_p);
        T rn = std::sqrt(xn * xn + (yn * yn) / (blob_q * blob_q));
        res += FUN(rn, blob_a, 0, blob_s);
    }
    out = res;
}

template<auto FUN, typename T> constexpr void
p_trait_mixture_2p(T& out, T r, T theta, const T* consts, const T* params)
{
    T res = 0;
    int nblobs = std::rint(consts[0]);
    for(int i = 0; i < nblobs; ++i)
    {
        T blob_r = params[         i];  // polar coord (radius)
        T blob_t = params[1*nblobs+i];  // polar coord (angle)
        T blob_a = params[2*nblobs+i];  // amplitude
        T blob_s = params[3*nblobs+i];  // scale
        T blob_b = params[4*nblobs+i];  // shape
        T blob_q = params[5*nblobs+i];  // axis ratio
        T blob_p = params[6*nblobs+i];  // blob roll (relative to blob_t)
        blob_t *= DEG_TO_RAD<T>;
        blob_p *= DEG_TO_RAD<T>;
        T xn = r * std::cos(theta) - blob_r * std::cos(blob_t);
        T yn = r * std::sin(theta) - blob_r * std::sin(blob_t);
        transform_lh_rotate_z(xn, yn, xn, yn, blob_t + blob_p);
        T rn = std::sqrt(xn * xn + (yn * yn) / (blob_q * blob_q));
        res += FUN(rn, blob_a, 0, blob_s, blob_b);
    }
    out = res;
}

template<typename T> constexpr void
p_trait_mixture_exponential(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_1p<exponential_1d_fun<T>>(out, r, theta, consts, params);
}

template<typename T> constexpr void
p_trait_mixture_gauss(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_1p<gauss_1d_fun<T>>(out, r, theta, consts, params);
}

template<typename T> constexpr void
p_trait_mixture_ggauss(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_2p<ggauss_1d_fun<T>>(out, r, theta, consts, params);
}

template<typename T> constexpr void
p_trait_mixture_moffat(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_2p<moffat_1d_fun<T>>(out, r, theta, consts, params);
}

template <typename T> constexpr void
p_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes,
        T r, const T* params)
{
    out = nodewise(r, rnidx, rnodes, params, 0, 1);
}

template <typename T> constexpr void
p_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    int k = std::rint(consts[0]);
    T a =     nodewise(r, rnidx, rnodes, params,       0, 1);
    T p = k ? nodewise(r, rnidx, rnodes, params, nrnodes, 1) * DEG_TO_RAD<T> :0;
    out = a * std::cos(k * (theta - p));
}

template<typename T> constexpr void
p_trait_nw_distortion(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    T a = nodewise(r, rnidx, rnodes, params,           0, 1);
    T p = nodewise(r, rnidx, rnodes, params,     nrnodes, 1) * DEG_TO_RAD<T>;
    T s = nodewise(r, rnidx, rnodes, params, 2 * nrnodes, 1);
    T t = wrap_angle(theta - p);
    out = a * std::exp(-(t * t * r * r) / (2 * s * s));
}

template<typename T> constexpr void
rp_trait_sample_polar_coords(
        T& out_r, T& out_t, RNG<T>& rng, T rmin, T rmax)
{
    out_r = std::sqrt(rmax * rmax + (rmin * rmin - rmax * rmax) * rng());
    out_t = 2 * PI<T> * rng();
}

template<typename T> constexpr void
rp_trait_sample_polar_coords_nw(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    T half_ring_width = rnodes[1] - rnodes[0];
    T rmin = rnodes[rnidx] - half_ring_width;
    T rmax = rnodes[rnidx] + half_ring_width;
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
}

// The node that ends the interval of nodes of the radius r, in the ring of
// the node rnidx (see lerp)
template<typename T> constexpr int
ring_node(T r, int rnidx, const T* rnodes)
{
    return r < rnodes[rnidx] ? rnidx : rnidx + 1;
}

// A point of the ring of the node rnidx (see
// rp_trait_sample_polar_coords_nw), at a uniform azimuth out_t, and of
// the radius out_r drawn from the density r w(r): w(r) >= 0 is the
// azimuthal mean of the trait at the radius, at most weight_max in the
// ring. By rejection of points uniform in area, so that the clouds of a
// ring follow its trait across it.
template<typename T, typename Weight> constexpr void
rp_trait_sample_polar_coords_ring(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        T weight_max, const Weight& weight)
{
    do {
        rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
    } while (rng() * weight_max > weight(out_r));
}

// A point of the ring of the node rnidx of a radial profile (|profile(r)|,
// monotonic in r: at most at an edge of the ring)
template<typename T, typename Profile> constexpr void
rp_trait_profile_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const Profile& profile)
{
    const T half_ring_width = rnodes[1] - rnodes[0];
    auto weight = [&](T r) { return std::abs(profile(r)); };
    const T weight_max = std::max(
            weight(rnodes[rnidx] - half_ring_width),
            weight(rnodes[rnidx] + half_ring_width));
    rp_trait_sample_polar_coords_ring(
            out_r, out_t, rng, rnidx, rnodes, weight_max, weight);
}

// The node-wise value (of the values params[offset + i] at the nodes) at the
// radius r of the ring of the node rnidx
template<typename T> constexpr T
ring_nodewise(T r, int rnidx, const T* rnodes, const T* params, int offset)
{
    return nodewise(r, ring_node(r, rnidx, rnodes), rnodes, params, offset, 1);
}

// The largest |value| of a node-wise parameter (of the values params[offset
// + i] at the nodes) in the ring of the node rnidx: at one of its edges or
// its middle, between which it is linear
template<typename T> constexpr T
ring_nodewise_max(int rnidx, const T* rnodes, const T* params, int offset)
{
    const T half_ring_width = rnodes[1] - rnodes[0];
    const T inner = ring_nodewise(
            rnodes[rnidx] - half_ring_width, rnidx, rnodes, params, offset);
    const T outer = ring_nodewise(
            rnodes[rnidx] + half_ring_width, rnidx, rnodes, params, offset);
    return std::max(
            std::max(std::abs(inner), std::abs(outer)),
            std::abs(params[offset + rnidx]));
}

template<typename T> constexpr void
rp_trait_uniform(T& out, const T* params)
{
    p_trait_uniform(out, params);
}

template<typename T> constexpr void
rp_trait_uniform_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* rnodes, int nrnodes)
{
    T rmin = rnodes[0];
    T rmax = rnodes[nrnodes - 1];
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
}

template<typename T> constexpr void
rp_trait_exponential(T& out, T r, const T* params)
{
    p_trait_exponential(out, r, params);
}

template<typename T> constexpr void
rp_trait_exponential_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_exponential(out, r, params);
        return out;
    });
}

template<typename T> constexpr void
rp_trait_gauss(T& out, T r, const T* params)
{
    p_trait_gauss(out, r, params);
}

template<typename T> constexpr void
rp_trait_gauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_gauss(out, r, params);
        return out;
    });
}

template<typename T> constexpr void
rp_trait_ggauss(T& out, T r, const T* params)
{
    p_trait_ggauss(out, r, params);
}

template<typename T> constexpr void
rp_trait_ggauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_ggauss(out, r, params);
        return out;
    });
}

template<typename T> constexpr void
rp_trait_lorentz(T& out, T r, const T* params)
{
    p_trait_lorentz(out, r, params);
}

template<typename T> constexpr void
rp_trait_lorentz_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_lorentz(out, r, params);
        return out;
    });
}

template<typename T> constexpr void
rp_trait_moffat(T& out, T r, const T* params)
{
    p_trait_moffat(out, r, params);
}

template<typename T> constexpr void
rp_trait_moffat_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_moffat(out, r, params);
        return out;
    });
}

template<typename T> constexpr void
rp_trait_sech2(T& out, T r, const T* params)
{
    p_trait_sech2(out, r, params);
}

template<typename T> constexpr void
rp_trait_sech2_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes,
        const T* params)
{
    rp_trait_profile_rnd(out_r, out_t, rng, rnidx, rnodes, [&](T r) {
        T out = 0;
        p_trait_sech2(out, r, params);
        return out;
    });
}

// The integral over the plane of the radial profile f of a round blob of
// amplitude 1, size s and shape b, and a random radius of a point of the
// blob (of density f(rho) rho): exponential exp(-rho / s), Gaussian
// exp(-rho^2 / (2 s^2)), generalised Gaussian exp(-(rho / s)^b) and Moffat
// (1 + (rho / s)^2)^(-b) (for b > 1)
template<typename T> constexpr T
mixture_blob_norm_exponential(T s, T b)
{
    (void)b;
    return 2 * PI<T> * s * s;
}

template<typename T> constexpr T
mixture_blob_radius_exponential(RNG<T>& rng, T s, T b)
{
    (void)b;
    return -s * std::log(rng() * rng());
}

template<typename T> constexpr T
mixture_blob_norm_gauss(T s, T b)
{
    (void)b;
    return 2 * PI<T> * s * s;
}

template<typename T> constexpr T
mixture_blob_radius_gauss(RNG<T>& rng, T s, T b)
{
    (void)b;
    return s * std::sqrt(-2 * std::log(rng()));
}

template<typename T> constexpr T
mixture_blob_norm_ggauss(T s, T b)
{
    return 2 * PI<T> * s * s * std::tgamma(2 / b) / b;
}

template<typename T> constexpr T
mixture_blob_radius_ggauss(RNG<T>& rng, T s, T b)
{
    return s * std::pow(gamma_rnd(rng, 2 / b), 1 / b);
}

template<typename T> constexpr T
mixture_blob_norm_moffat(T s, T b)
{
    return PI<T> * s * s / (b - 1);
}

template<typename T> constexpr T
mixture_blob_radius_moffat(RNG<T>& rng, T s, T b)
{
    return s * std::sqrt(std::pow(rng(), 1 / (1 - b)) - 1);
}

// A random point of a mixture of nblobs blobs (see p_trait_mixture_1p and
// p_trait_mixture_2p, of NP parameters each): a blob, drawn by its flux
// (|amplitude| times its integral times its axis ratio), a point of it,
// and the sign of its amplitude
template<auto Norm, auto Radius, int NP, typename T> constexpr void
rp_trait_mixture_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    const int nblobs = std::rint(consts[0]);
    // The parameters of the blobs: r, t, a, s, (b,) q, p
    const T* blob_r = params;
    const T* blob_t = params + nblobs;
    const T* blob_a = params + 2 * nblobs;
    const T* blob_s = params + 3 * nblobs;
    const T* blob_b = NP == 7 ? params + 4 * nblobs : nullptr;
    const T* blob_q = params + (NP - 2) * nblobs;
    const T* blob_p = params + (NP - 1) * nblobs;
    T total = 0;
    for (int i = 0; i < nblobs; ++i)
        total += std::abs(blob_a[i] * blob_q[i])
                * Norm(blob_s[i], blob_b ? blob_b[i] : T{0});
    // Draw a blob by its flux
    const T u = rng() * total;
    int i = 0;
    T sum = 0;
    for (; i < nblobs - 1; ++i)
    {
        sum += std::abs(blob_a[i] * blob_q[i])
                * Norm(blob_s[i], blob_b ? blob_b[i] : T{0});
        if (u < sum)
            break;
    }
    // A point of the blob, in its frame (elongated along its x axis)
    const T rho = Radius(rng, blob_s[i], blob_b ? blob_b[i] : T{0});
    const T psi = 2 * PI<T> * rng();
    const T xb = rho * std::cos(psi);
    const T yb = blob_q[i] * rho * std::sin(psi);
    // Back to the disk plane: the inverse of transform_lh_rotate_z by the
    // angle of the blob, from its centre
    const T t = blob_t[i] * DEG_TO_RAD<T>;
    const T phi = t + blob_p[i] * DEG_TO_RAD<T>;
    const T x = blob_r[i] * std::cos(t)
            - std::sin(phi) * xb - std::cos(phi) * yb;
    const T y = blob_r[i] * std::sin(t)
            + std::cos(phi) * xb - std::sin(phi) * yb;
    out_r = std::sqrt(x * x + y * y);
    out_t = std::atan2(y, x);
    out_s = blob_a[i] < 0 ? -1 : 1;
}

template<typename T> constexpr void
rp_trait_mixture_exponential(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_exponential(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_exponential_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    rp_trait_mixture_rnd<
            mixture_blob_norm_exponential<T>, mixture_blob_radius_exponential<T>,
            6>(out_s, out_r, out_t, rng, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_gauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_gauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_gauss_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    rp_trait_mixture_rnd<
            mixture_blob_norm_gauss<T>, mixture_blob_radius_gauss<T>,
            6>(out_s, out_r, out_t, rng, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_ggauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_ggauss_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    rp_trait_mixture_rnd<
            mixture_blob_norm_ggauss<T>, mixture_blob_radius_ggauss<T>,
            7>(out_s, out_r, out_t, rng, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_moffat(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_moffat_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    rp_trait_mixture_rnd<
            mixture_blob_norm_moffat<T>, mixture_blob_radius_moffat<T>,
            7>(out_s, out_r, out_t, rng, consts, params);
}

template<typename T> constexpr void
rp_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

// A point of the ring of the node rnidx of the node-wise amplitude a
// (params[i]): of the radius drawn from |a|, and out_s the sign of a there
template<typename T> constexpr void
rp_trait_nw_uniform_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng, int rnidx,
        const T* rnodes, const T* params)
{
    rp_trait_sample_polar_coords_ring(
            out_r, out_t, rng, rnidx, rnodes,
            ring_nodewise_max(rnidx, rnodes, params, 0),
            [&](T r) {
                return std::abs(ring_nodewise(r, rnidx, rnodes, params, 0));
            });
    out_s = ring_nodewise(out_r, rnidx, rnodes, params, 0) < 0 ? -1 : 1;
}

template<typename T> constexpr void
rp_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

// Draw the azimuth out_t of a cloud from |cos(k (t - p))| (k > 0, p in
// radians): one of its 2k lobes, centred on p + j pi / k, then the offset d
// from the centre of the lobe, whose density cos(k d) on (-pi / 2k, pi /
// 2k) has the cdf (1 + sin(k d)) / 2; out_s is the sign of the cosine
// there, positive in the even lobes.
template<typename T> constexpr void
harmonic_azimuth_rnd(T& out_s, T& out_t, RNG<T>& rng, int k, T p)
{
    int j = std::min(int(rng() * 2 * k), 2 * k - 1);
    T d = std::asin(2 * rng() - 1) / k;
    out_t = wrap_angle(p + j * PI<T> / k + d);
    out_s = j % 2 ? -1 : 1;
}

// A point of the ring of the node rnidx of a harmonic a cos(k (t - p)), of
// the node-wise a and p (params): of the radius drawn from |a| (the
// azimuthal mean of the |harmonic| is |a| times a constant), and the
// azimuth from the |cosine| of the p there; out_s is the sign of the
// harmonic at the point.
template<typename T> constexpr void
rp_trait_nw_harmonic_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes,
        const T* consts, const T* params)
{
    rp_trait_sample_polar_coords_ring(
            out_r, out_t, rng, rnidx, rnodes,
            ring_nodewise_max(rnidx, rnodes, params, 0),
            [&](T r) {
                return std::abs(ring_nodewise(r, rnidx, rnodes, params, 0));
            });
    out_s = ring_nodewise(out_r, rnidx, rnodes, params, 0) < 0 ? -1 : 1;
    // (a harmonic of order 0 is uniform, and has no phase parameter)
    int k = std::rint(consts[0]);
    if (k == 0)
        return;
    T p = ring_nodewise(out_r, rnidx, rnodes, params, nrnodes)
            * DEG_TO_RAD<T>;
    T lobe = 1;
    harmonic_azimuth_rnd(lobe, out_t, rng, k, p);
    out_s *= lobe;
}

// The density of a spiral density wave of m = consts[0] arms (see
// traits.RPTraitSpiral): a cos(m (theta - p) - f), of p = params[0] and
// the tables a and f at the subnodes, after the consts[1] parameters that
// are not node-wise
template<typename T> constexpr void
rp_trait_spiral(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    int m = std::rint(consts[0]);
    int offset = std::rint(consts[1]);
    T p = params[0] * DEG_TO_RAD<T>;
    T a = nodewise(r, rnidx, rnodes, params, offset, 1);
    T f = nodewise(r, rnidx, rnodes, params, offset + nrnodes, 1);
    out = a * std::cos(m * (theta - p) - f);
}

// A cloud of a spiral density wave in the ring rnidx: of the radius drawn
// from |a| (see rp_trait_nw_harmonic_rnd), and the azimuth from the
// |harmonic| a cos(m (theta - p) - f) of the f there, whose maxima are at
// the azimuths p + f / m (see harmonic_azimuth_rnd); out_s is its sign
// at the point.
template<typename T> constexpr void
rp_trait_spiral_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes,
        const T* consts, const T* params)
{
    int m = std::rint(consts[0]);
    int offset = std::rint(consts[1]);
    rp_trait_sample_polar_coords_ring(
            out_r, out_t, rng, rnidx, rnodes,
            ring_nodewise_max(rnidx, rnodes, params, offset),
            [&](T r) {
                return std::abs(
                        ring_nodewise(r, rnidx, rnodes, params, offset));
            });
    out_s = ring_nodewise(out_r, rnidx, rnodes, params, offset) < 0 ? -1 : 1;
    T f = ring_nodewise(out_r, rnidx, rnodes, params, offset + nrnodes);
    T p = params[0] * DEG_TO_RAD<T> + f / m;
    T lobe = 1;
    harmonic_azimuth_rnd(lobe, out_t, rng, m, p);
    out_s *= lobe;
}

template<typename T> constexpr void
rp_trait_nw_distortion(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    p_trait_nw_distortion(out, rnidx, rnodes, nrnodes, r, theta, params);
}

// The azimuth out_t of a cloud of a distortion of the phase p (radians)
// and the width w (radians, the arc length s over the radius): p + d, d
// drawn from a Gaussian of the width truncated to a turn, by rejection, of
// Gaussian draws for narrow ones, and of uniform draws for wide ones (both
// accept most draws)
template<typename T> constexpr void
distortion_azimuth_rnd(T& out_t, RNG<T>& rng, T p, T w)
{
    T d = 0;
    if (w < PI<T>) {
        do {
            d = w * normal_rnd(rng);
        } while (std::abs(d) > PI<T>);
    } else {
        do {
            d = PI<T> * (2 * rng() - 1);
        } while (rng() > std::exp(-d * d / (2 * w * w)));
    }
    out_t = wrap_angle(p + d);
}

// A point of the ring of the node rnidx of a distortion a exp(-(t r)^2 / (2
// s^2)) (t, the azimuth from p, within half a turn), of the node-wise a, p
// and s (params): of the radius drawn from its azimuthal mean, |a| (|s| /
// r) erf(pi r / (sqrt(2) |s|)) (times 1 / sqrt(2 pi)), which is at most |a|
// sqrt(2 pi) and |a| |s| / r; at the azimuth from p of the width s / r
// (see distortion_azimuth_rnd); out_s is the sign of a there.
template<typename T> constexpr void
rp_trait_nw_distortion_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes,
        const T* params)
{
    auto at = [&](T r, int k) {
        return ring_nodewise(r, rnidx, rnodes, params, k * nrnodes);
    };
    const T inner = rnodes[rnidx] - (rnodes[1] - rnodes[0]);
    const T wide = std::sqrt(2 * PI<T>);
    const T narrow = inner > 0
            ? ring_nodewise_max(rnidx, rnodes, params, 2 * nrnodes) / inner
            : wide;
    rp_trait_sample_polar_coords_ring(
            out_r, out_t, rng, rnidx, rnodes,
            ring_nodewise_max(rnidx, rnodes, params, 0)
            * std::min(wide, narrow),
            [&](T r) {
                const T s = std::abs(at(r, 2));
                return s > 0 && r > 0
                        ? std::abs(at(r, 0)) * (s / r)
                          * std::erf(PI<T> * r / (std::sqrt(T(2)) * s))
                        : T(0);
            });
    out_s = at(out_r, 0) < 0 ? -1 : 1;
    distortion_azimuth_rnd(
            out_t, rng, at(out_r, 1) * DEG_TO_RAD<T>,
            std::abs(at(out_r, 2) / out_r));
}

// A free map: nx by ny square cells of side step (consts) in the plane of
// the disk, centred on its centre, x along azimuth 0 and y along azimuth
// 90 degrees. The cell (i, j) covers [(i - nx / 2) step, (i + 1 - nx / 2)
// step) along x, and likewise along y, and has the density params[j nx +
// i]; 0 outside the grid.
template<typename T> constexpr void
rp_trait_map(T& out, T x, T y, const T* consts, const T* params)
{
    const int nx = std::rint(consts[0]);
    const int ny = std::rint(consts[1]);
    const T step = consts[2];
    const T u = std::floor(x / step + T(nx) / 2);
    const T v = std::floor(y / step + T(ny) / 2);
    out = u >= 0 && u < nx && v >= 0 && v < ny
            ? params[int(v) * nx + int(u)] : 0;
}

// A point of a free map: a cell drawn by its |density|, a uniform point in
// it, and the sign of its density
template<typename T> constexpr void
rp_trait_map_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        const T* consts, const T* params)
{
    const int nx = std::rint(consts[0]);
    const int ny = std::rint(consts[1]);
    const T step = consts[2];
    const int ncells = nx * ny;
    T total = 0;
    for (int k = 0; k < ncells; ++k)
        total += std::abs(params[k]);
    const T u = rng() * total;
    int k = 0;
    T sum = 0;
    for (; k < ncells - 1; ++k)
    {
        sum += std::abs(params[k]);
        if (u < sum)
            break;
    }
    const T x = (k % nx + rng() - T(nx) / 2) * step;
    const T y = (k / nx + rng() - T(ny) / 2) * step;
    out_r = std::sqrt(x * x + y * y);
    out_t = std::atan2(y, x);
    out_s = params[k] < 0 ? -1 : 1;
}

// The density height traits: a vertical profile of the density around the
// midplane (which vertical distortion traits displace), a pdf (it
// integrates to 1), at the height z above the midplane. The density lies
// within zmax (consts[0]) of the midplane: the profile is truncated to the
// heights [-zmax, zmax], and renormalised there, so that each column keeps
// its density (see _trunc_1d_pdf).
template<auto FunTrunc, typename T> constexpr void
rh_trait_pdf_1p(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    out = FunTrunc(z, 0, p1, -zmax, zmax);
}

template<auto FunTrunc, typename T> constexpr void
rh_trait_pdf_2p(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    T p2 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, nrnodes, 1)
            : params[1];
    out = FunTrunc(z, 0, p1, p2, -zmax, zmax);
}

// The half width of the heights over which rh_trait_mean_1p takes the
// mean of a profile of the scale p: 0 (its pdf at the height) for one
// under 1% of the scale, where the pdf is the mean (to about 1e-5), and
// the difference of the CDF at its ends would lose the precision of
// floats (e.g. edge-on, where the cosine of 90 degrees is 6e-17)
template<typename T> constexpr T
rh_trait_mean_width(T w, T p)
{
    return w < T(1e-2) * std::abs(p) ? T(0) : w;
}

// The mean of a density height trait (see rh_trait_pdf_1p) over the
// heights [z - w, z + w]: of its truncated profile, from its CDF (see
// _trunc_1d_mean); its pdf at z for w = 0
template<auto PDF, auto CDF, typename T> constexpr void
rh_trait_mean_1p(
        T& out,
        int rnidx, const T* rnodes, T r, T z, T w,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    out = _trunc_1d_mean<PDF, CDF>(
            -zmax, zmax, z, rh_trait_mean_width(w, p1), T(0), p1);
}

template<auto PDF, auto CDF, typename T> constexpr void
rh_trait_mean_2p(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z, T w,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    T p2 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, nrnodes, 1)
            : params[1];
    out = _trunc_1d_mean<PDF, CDF>(
            -zmax, zmax, z, rh_trait_mean_width(w, p1), T(0), p1, p2);
}

// A random height above the midplane of the density height traits (see
// rh_trait_pdf_1p): of the truncated profile, NaN if it is none
template<auto FunTrunc, typename T> constexpr void
rh_trait_rnd_1p(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    out = FunTrunc(rng, 0, p1, -zmax, zmax);
}

template<auto FunTrunc, typename T> constexpr void
rh_trait_rnd_2p(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    T zmax = consts[0];
    bool use_rnodes = consts[1];
    T p1 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, 0, 1)
            : params[0];
    T p2 = use_rnodes
            ? nodewise(r, rnidx, rnodes, params, nrnodes, 1)
            : params[1];
    out = FunTrunc(rng, 0, p1, p2, -zmax, zmax);
}

template<typename T> constexpr void
rh_trait_uniform_pdf(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_1p<uniform_wm_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_uniform_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_1p<uniform_wm_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_exponential_pdf(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_1p<exponential_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_exponential_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_1p<exponential_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_gauss_pdf(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_1p<gauss_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_gauss_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_1p<gauss_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_ggauss_pdf(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_2p<ggauss_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, nrnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_ggauss_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_2p<ggauss_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_lorentz_pdf(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_1p<lorentz_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_lorentz_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_1p<lorentz_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_moffat_pdf(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_2p<moffat_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, nrnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_moffat_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_2p<moffat_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

template<typename T> constexpr void
rh_trait_sech2_pdf(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    rh_trait_pdf_1p<sech2_1d_pdf_trunc<T>>(
            out, rnidx, rnodes, r, z, consts, params);
}

template<typename T> constexpr void
rh_trait_sech2_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    rh_trait_rnd_1p<sech2_1d_rnd_trunc<T>>(
            out, rng, rnidx, rnodes, nrnodes, r, consts, params);
}

// The speeds of the velocity polar traits along their directions (see
// vp_disk_project): v, the speed parameter of each, is its asymptotic or
// maximum speed

template<typename T> constexpr void
vp_trait_uniform(T& out, const T* params)
{
    p_trait_uniform(out, params);
}

template<typename T> constexpr void
vp_trait_arctan(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    // Expect rt > 0
    out = v * (2/PI<T>) * std::atan(r/rt);
}

template<typename T> constexpr void
vp_trait_boissier(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    // Expect rt > 0
    out =  v * (1 - std::exp(-r/rt));
}

template<typename T> constexpr void
vp_trait_epinat(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    T a = params[2];
    T g = params[3];
    // Expect rt > 0
    out = v * std::pow(r/rt, g) / (1 + std::pow(r/rt, a));
}

template<typename T> constexpr void
vp_trait_lramp(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    out = r < rt ? v * r/rt : v;
}

template<typename T> constexpr void
vp_trait_tanh(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    // Expect rt > 0
    out = v * std::tanh(r/rt);
}

template<typename T> constexpr void
vp_trait_polyex(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    T a = params[2];
    // Expect rt > 0
    out = v * (1 - std::exp(-r/rt)) * (1 + a * r/rt);
}

template<typename T> constexpr void
vp_trait_rix(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    T b = params[2];
    T g = params[3];
    // Expect rt > 0, g > 0
    out = v * std::pow(1 + r/rt, b) * std::pow(1 + std::pow(r/rt, -g), -1/g);
}

// Courteau (1997): v (1 + x)^b / (1 + x^g)^(1 / g), with x = rt / r (0
// at the centre)
template<typename T> constexpr void
vp_trait_courteau(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    T b = params[2];
    T g = params[3];
    const T x = rt / r;
    out = r > 0
            ? v * std::pow(1 + x, b) / std::pow(1 + std::pow(x, g), 1 / g)
            : 0;
}

// Brandt (1960): v (r / rt) / (1/3 + 2/3 (r / rt)^n)^(3 / (2 n)), with
// its maximum v at rt
template<typename T> constexpr void
vp_trait_brandt(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    T n = params[2];
    const T x = r / rt;
    out = v * x
            / std::pow(T{1} / 3 + T{2} / 3 * std::pow(x, n), 3 / (2 * n));
}

// A pseudo-isothermal sphere: v sqrt(1 - atan(u) / u), with u = r / rt,
// the core radius rt and the asymptotic speed v (0 at the centre). Near
// the centre, where 1 - atan(u) / u loses its precision (and can round
// below 0), by its series u^2 / 3 - u^4 / 5 + u^6 / 7.
template<typename T> constexpr void
vp_trait_iso(T& out, T r, const T* params)
{
    T rt = params[0];
    T v = params[1];
    const T u = r / rt;
    const T u2 = u * u;
    const T g = std::abs(u) < T{0.1}
            ? u2 * (T{1} / 3 - u2 * (T{1} / 5 - u2 / 7))
            : 1 - std::atan(u) / u;
    out = v * std::sqrt(g);
}

// The shape of the rotation curve of an NFW halo of scale radius rt, with
// its maximum v (at 2.1626 rt): v sqrt(f(r / rt) / f_max), with f(u) =
// (ln(1 + u) - u / (1 + u)) / u (0 at the centre). Near the centre, where
// the difference loses its precision, by its series u / 2 - 2 u^2 / 3 +
// 3 u^3 / 4 - 4 u^4 / 5.
template<typename T> constexpr void
vp_trait_nfw(T& out, T r, const T* params)
{
    constexpr T F_MAX = T{0.21621659550187317};
    T rt = params[0];
    T v = params[1];
    const T u = r / rt;
    const T f = u < T{0.01}
            ? u * (T{1} / 2 - u * (T{2} / 3 - u * (T{3} / 4 - u * T{4} / 5)))
            : (std::log1p(u) - u / (1 + u)) / u;
    out = u > 0 ? v * std::sqrt(f / F_MAX) : 0;
}

template<typename T> constexpr void
vp_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

// A speed given at the subnodes and beyond the last, at the radii
// consts[1 ... consts[0]] of the tail of the component (params: its
// values at the subnodes, then at those radii; e.g. the rotation of the
// mass of its galaxy, see traits.VPTraitMass): linear between them, and
// beyond the last radius, its value there
template<typename T> constexpr void
vp_trait_table(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    const int n = nrnodes - 1;
    if (!(r > rnodes[n])) {
        out = nodewise(r, rnidx, rnodes, params, 0, 1);
        return;
    }
    const int ntail = std::rint(consts[0]);
    const T* radii = consts + 1;
    const T* values = params + nrnodes;
    T r0 = rnodes[n], v0 = params[n];
    for (int i = 0; i < ntail; ++i) {
        if (r < radii[i]) {
            out = v0 + (r - r0) * (values[i] - v0) / (radii[i] - r0);
            return;
        }
        r0 = radii[i];
        v0 = values[i];
    }
    out = v0;
}

template<typename T> constexpr void
vp_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

// The speed v within the two opposite wedges of the full width s centred
// at the azimuths p and p + 180 degrees, 0 outside
template<typename T> constexpr void
vp_trait_wedge(T& out, T theta, const T* params)
{
    T v = params[0];
    T p = params[1] * DEG_TO_RAD<T>;
    T s = params[2] * DEG_TO_RAD<T>;
    out = std::abs(std::cos(theta - p)) >= std::cos(s * T{0.5}) ? v : 0;
}

// The flow of the speed v along the axis of a bar at the azimuth p, away
// from its centre, within the strip of the full width s along the axis:
// its parts v_x and v_y along the azimuths 0 and 90 degrees, v sign(x)
// (cos(p), sin(p)) where |y| < s / 2, x along the axis and y across it,
// 0 outside
template<typename T> constexpr void
bar_flow(T& vx, T& vy, T v, T p, T s, T r, T theta)
{
    T x = r * std::cos(theta - p);
    T y = r * std::sin(theta - p);
    T flow = std::abs(y) < s * T{0.5} ? v * sign(x) : 0;
    vx = flow * std::cos(p);
    vy = flow * std::sin(p);
}

// The flow along a bar (see bar_flow) of the speed v = params[0] at the
// azimuth p = params[1] in the strip of the width s = params[2]: its parts
// v_x and v_y (see vp_trait_plane_x; vp_trait_bar gives v_y)
template<typename T> constexpr void
vp_trait_bar_plane(T& vx, T& vy, T r, T theta, const T* params)
{
    T v = params[0];
    T p = params[1] * DEG_TO_RAD<T>;
    T s = params[2];
    bar_flow(vx, vy, v, p, s, r, theta);
}

template<typename T> constexpr void
vp_trait_bar(T& out, T r, T theta, const T* params)
{
    T vx = 0;
    vp_trait_bar_plane(vx, out, r, theta, params);
}

// A flow along a bar (see bar_flow) of the speed v(|x|) at the distance
// |x| along its axis, of the values params[2 + i] at the distances
// rnodes[i] (interpolated linearly), and beyond the last, that there (see
// nodewise): its parts v_x and v_y
template<typename T> constexpr void
vp_trait_nw_bar_plane(
        T& vx, T& vy,
        const T* rnodes, int nrnodes, T r, T theta, const T* params)
{
    T p = params[0] * DEG_TO_RAD<T>;
    T s = params[1];
    T d = std::abs(r * std::cos(theta - p));
    vx = vy = 0;
    int index = 1;
    while (index < nrnodes - 1 && d >= rnodes[index])
        ++index;
    T v = nodewise(d, index, rnodes, params, 2, 1);
    bar_flow(vx, vy, v, p, s, r, theta);
}

template<typename T> constexpr void
vp_trait_nw_bar(
        T& out, const T* rnodes, int nrnodes, T r, T theta, const T* params)
{
    T vx = 0;
    vp_trait_nw_bar_plane(vx, out, rnodes, nrnodes, r, theta, params);
}

// The velocity of a spiral density wave of m = consts[0] arms (see
// traits.VPTraitSpiral), v_R = a_R cos(phi) and v_t = a_t sin(phi), phi =
// m (theta - p) - f, of p = params[0] and the tables a_R, a_t and f at the
// subnodes, after the consts[1] parameters that are not node-wise: its
// parts along the azimuths 0 and 90 degrees, v_x = v_R cos(theta) - v_t
// sin(theta) and v_y = v_R sin(theta) + v_t cos(theta)
template<typename T> constexpr void
vp_trait_spiral_plane(
        T& vx, T& vy,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    int m = std::rint(consts[0]);
    int offset = std::rint(consts[1]);
    T p = params[0] * DEG_TO_RAD<T>;
    T ar = nodewise(r, rnidx, rnodes, params, offset, 1);
    T at = nodewise(r, rnidx, rnodes, params, offset + nrnodes, 1);
    T f = nodewise(r, rnidx, rnodes, params, offset + 2 * nrnodes, 1);
    T phi = m * (theta - p) - f;
    T vr = ar * std::cos(phi);
    T vt = at * std::sin(phi);
    vx = vr * std::cos(theta) - vt * std::sin(theta);
    vy = vr * std::sin(theta) + vt * std::cos(theta);
}

template<typename T> constexpr void
vp_trait_spiral(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    T vx = 0;
    vp_trait_spiral_plane(
            vx, out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_uniform(T& out, const T* params)
{
    p_trait_uniform(out, params);
}

template<typename T> constexpr void
dp_trait_exponential(T& out, T r, const T* params)
{
    p_trait_exponential(out, r, params);
}

template<typename T> constexpr void
dp_trait_gauss(T& out, T r, const T* params)
{
    p_trait_gauss(out, r, params);
}

template<typename T> constexpr void
dp_trait_ggauss(T& out, T r, const T* params)
{
    p_trait_ggauss(out, r, params);
}

template<typename T> constexpr void
dp_trait_lorentz(T& out, T r, const T* params)
{
    p_trait_lorentz(out, r, params);
}

template<typename T> constexpr void
dp_trait_moffat(T& out, T r, const T* params)
{
    p_trait_moffat(out, r, params);
}

template<typename T> constexpr void
dp_trait_sech2(T& out, T r, const T* params)
{
    p_trait_sech2(out, r, params);
}

template<typename T> constexpr void
dp_trait_mixture_exponential(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_exponential(out, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_mixture_gauss(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_gauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_mixture_ggauss(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_ggauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_mixture_moffat(
        T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_moffat(out, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

template<typename T> constexpr void
dp_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_nw_distortion(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    p_trait_nw_distortion(out, rnidx, rnodes, nrnodes, r, theta, params);
}

template<typename T> constexpr void
zp_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

template<typename T> constexpr void
zp_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

// 1 for the angles theta within s / 2 of p, 0 for the others. Any angle
// range works (e.g. the clouds of mcdisk are in [0, 2 pi), the smooth
// disk in [-pi, pi]).
template<typename T> constexpr void
azrange(T& out, T theta, T p, T s)
{
    out = std::abs(wrap_angle(theta - p)) < s * T{0.5} ? 1 : 0;
}

template<typename T> constexpr void
sp_trait_azrange(T& out, T theta, const T* params)
{
    T p = params[0] * DEG_TO_RAD<T>;
    T s = params[1] * DEG_TO_RAD<T>;
    azrange(out, theta, p, s);
}

// 1 for the radii from rmin (inclusive) to rmax (exclusive), 0 for the
// others (e.g. an inner hole or an outer edge)
template<typename T> constexpr void
sp_trait_rrange(T& out, T r, const T* params)
{
    out = r >= params[0] && r < params[1] ? 1 : 0;
}

template<typename T> constexpr void
sp_trait_nw_azrange(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    T p = nodewise(r, rnidx, rnodes, params,       0, 1) * DEG_TO_RAD<T>;
    T s = nodewise(r, rnidx, rnodes, params, nrnodes, 1) * DEG_TO_RAD<T>;
    azrange(out, theta, p, s);
}

// The tails of the density polar traits. A component does not end at
// its last node rl (the nodes say where its node-wise parameters are
// given): beyond it, its density polar traits (but the mixtures of blobs
// and the maps, whole as they are) go on as their value there times the
// exponential exp(-(r - rl) / h). Its scale h follows the log-slope g =
// d ln|a| / dr of their amplitude a at rl: -1 / g where they fall (exact
// for an exponential profile), at most rl, and rl where they do not (flat
// or rising, which does not say where they end). The radial profiles are
// taken at rl, and the node-wise traits at the radius, of their node-wise
// values at rl (see nodewise; e.g. the phase of a harmonic). See
// traits.RPTrait.tail_scale, which gives the same scales.

// Whether a density polar trait has a tail
constexpr bool
rp_trait_has_tail(int uid)
{
    return uid != RPT_UID_MIXTURE_EXP && uid != RPT_UID_MIXTURE_GAUSS
            && uid != RPT_UID_MIXTURE_GGAUSS && uid != RPT_UID_MIXTURE_MOFFAT
            && uid != RPT_UID_MAP;
}

// The scale of a tail of the log-slope g at the last node rl
template<typename T> constexpr T
tail_scale(T g, T rl)
{
    return g < 0 ? std::min(-1 / g, rl) : rl;
}

// The log-slope at the last node of a node-wise amplitude (of the values
// params[offset + i] at the nodes): that of its last interval
template<typename T> constexpr T
nodewise_log_slope(const T* rnodes, int nrnodes, const T* params, int offset)
{
    const int n = nrnodes - 1;
    const T a = params[offset + n];
    return (a - params[offset + n - 1]) / ((rnodes[n] - rnodes[n - 1]) * a);
}

// The log-slope of a density polar trait with a tail at the last node:
// of its radial profile, or of its node-wise amplitude
template<typename T> constexpr T
rp_trait_log_slope(
        int uid, const T* consts, const T* params,
        const T* rnodes, int nrnodes)
{
    const T rl = rnodes[nrnodes - 1];
    const T s = params[1];
    switch (uid)
    {
    case RPT_UID_EXP:
        return -1 / s;
    case RPT_UID_GAUSS:
        return -rl / (s * s);
    case RPT_UID_GGAUSS:
        return -params[2] / s * std::pow(rl / s, params[2] - 1);
    case RPT_UID_LORENTZ:
        return -2 * rl / (rl * rl + s * s);
    case RPT_UID_MOFFAT:
        return -2 * params[2] * rl / (rl * rl + s * s);
    case RPT_UID_SECH2:
        return -2 * std::tanh(rl / s) / s;
    case RPT_UID_NW_UNIFORM:
    case RPT_UID_NW_HARMONIC:
    case RPT_UID_NW_DISTORTION:
        return nodewise_log_slope(rnodes, nrnodes, params, 0);
    case RPT_UID_SPIRAL:
        return nodewise_log_slope(
                rnodes, nrnodes, params, int(std::rint(consts[1])));
    default:
        // (the uniform one: flat)
        return 0;
    }
}

// The tail of a density polar trait at the radius r beyond the last node
// rl: exp(-(r - rl) / h)
template<typename T> constexpr T
rp_trait_tail(
        int uid, const T* consts, const T* params,
        const T* rnodes, int nrnodes, T r)
{
    const T rl = rnodes[nrnodes - 1];
    const T g = rp_trait_log_slope(uid, consts, params, rnodes, nrnodes);
    return std::exp(-(r - rl) / tail_scale(g, rl));
}

// A cloud of the tail of a density polar trait (see rp_trait_tail): of
// the radius r = rl + x drawn from r exp(-x / h) (x from an exponential of
// scale h, or with the probability h / (rl + h), from a gamma of shape 2
// and scale h), at which it is drawn as in its rings (the azimuth and
// out_s, the sign of the trait there, for the node-wise traits); that of
// a distortion, whose azimuthal mean falls faster still, by rejection.
template<typename T> constexpr void
rp_trait_tail_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int uid, const T* consts, const T* params,
        const T* rnodes, int nrnodes)
{
    const int n = nrnodes - 1;
    const T rl = rnodes[n];
    const T h = tail_scale(
            rp_trait_log_slope(uid, consts, params, rnodes, nrnodes), rl);
    auto radius = [&]() {
        // (in this order on every driver)
        const T u = rng();
        T x = -h * std::log(rng());
        if (u * (rl + h) < h)
            x -= h * std::log(rng());
        return rl + x;
    };
    out_r = radius();
    out_t = 2 * PI<T> * rng();
    switch (uid)
    {
    case RPT_UID_NW_UNIFORM:
        out_s = params[n] < 0 ? -1 : 1;
        break;
    case RPT_UID_NW_HARMONIC: {
        out_s = params[n] < 0 ? -1 : 1;
        const int k = std::rint(consts[0]);
        if (k == 0)
            break;
        T lobe = 1;
        harmonic_azimuth_rnd(
                lobe, out_t, rng, k, params[nrnodes + n] * DEG_TO_RAD<T>);
        out_s *= lobe;
        break;
    }
    case RPT_UID_SPIRAL: {
        const int m = std::rint(consts[0]);
        const int offset = std::rint(consts[1]);
        out_s = params[offset + n] < 0 ? -1 : 1;
        const T p = params[0] * DEG_TO_RAD<T>
                + params[offset + nrnodes + n] / m;
        T lobe = 1;
        harmonic_azimuth_rnd(lobe, out_t, rng, m, p);
        out_s *= lobe;
        break;
    }
    case RPT_UID_NW_DISTORTION: {
        // (its azimuthal mean over that of its amplitude, (s / r) erf(pi
        // r / (sqrt(2) s)), falls with r)
        const T s = std::abs(params[2 * nrnodes + n]);
        auto mean = [&](T r) {
            return s * std::erf(PI<T> * r / (std::sqrt(T(2)) * s)) / r;
        };
        const T mean_max = s > 0 ? mean(rl) : T(0);
        while (rng() * mean_max > mean(out_r))
            out_r = radius();
        out_s = params[n] < 0 ? -1 : 1;
        distortion_azimuth_rnd(
                out_t, rng, params[nrnodes + n] * DEG_TO_RAD<T>,
                std::abs(s / out_r));
        break;
    }
    default:
        // (a radial profile: the sign is that of the flux of its pool)
        break;
    }
}

// A polar trait of the shapes of the density polar traits at the point
// (x, y, r, theta) of the plane of a component, in the ring rnidx: of a
// coefficient of line opacity (which keeps its node-wise values beyond
// the last node, see nodewise), or of a density polar trait within the
// last node (see rp_trait)
template<typename T> constexpr void
op_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    switch (uid)
    {
    case RPT_UID_UNIFORM:
        rp_trait_uniform(
                out, params);
        break;
    case RPT_UID_EXP:
        rp_trait_exponential(
                out, r, params);
        break;
    case RPT_UID_GAUSS:
        rp_trait_gauss(
                out, r, params);
        break;
    case RPT_UID_GGAUSS:
        rp_trait_ggauss(
                out, r, params);
        break;
    case RPT_UID_LORENTZ:
        rp_trait_lorentz(
                out, r, params);
        break;
    case RPT_UID_MOFFAT:
        rp_trait_moffat(
                out, r, params);
        break;
    case RPT_UID_SECH2:
        rp_trait_sech2(
                out, r, params);
        break;
    case RPT_UID_MIXTURE_EXP:
        rp_trait_mixture_exponential(
                out, r, theta, consts, params);
        break;
    case RPT_UID_MIXTURE_GAUSS:
        rp_trait_mixture_gauss(
                out, r, theta, consts, params);
        break;
    case RPT_UID_MIXTURE_GGAUSS:
        rp_trait_mixture_ggauss(
                out, r, theta, consts, params);
        break;
    case RPT_UID_MIXTURE_MOFFAT:
        rp_trait_mixture_moffat(
                out, r, theta, consts, params);
        break;
    case RPT_UID_NW_UNIFORM:
        rp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case RPT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case RPT_UID_SPIRAL:
        rp_trait_spiral(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case RPT_UID_NW_DISTORTION:
        rp_trait_nw_distortion(
                out, rnidx, rnodes, nrnodes, r, theta, params);
        break;
    case RPT_UID_MAP:
        rp_trait_map(
                out, x, y, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// A density polar trait at the point (x, y, r, theta) of the plane of a
// component, in the ring rnidx; beyond the last node, with its tail (see
// rp_trait_tail): of a radial profile, its value at the last node, and of
// a node-wise trait, its value at the radius (of its node-wise values at
// the last node), times the tail
template<typename T> constexpr void
rp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    const T rl = rnodes[nrnodes - 1];
    if (!(r > rl) || !rp_trait_has_tail(uid)) {
        op_trait(out, uid, consts, params, rnidx, rnodes, nrnodes,
                 x, y, r, theta);
        return;
    }
    const bool profile = uid <= RPT_UID_SECH2;
    op_trait(out, uid, consts, params, rnidx, rnodes, nrnodes,
             x, y, profile ? rl : r, theta);
    out *= rp_trait_tail(uid, consts, params, rnodes, nrnodes, r);
}

template<typename T> constexpr void
rp_trait_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int uid, const T* consts, const T* params,
        int ridx, const T* rnodes, int nrnodes)
{
    // Convert ring index to rnode index
    auto rnidx = ridx + 1;

    switch (uid)
    {
    case RPT_UID_UNIFORM:
        rp_trait_uniform_rnd(
                out_r, out_t,
                rng, rnodes, nrnodes);
        break;
    case RPT_UID_EXP:
        rp_trait_exponential_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_GAUSS:
        rp_trait_gauss_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_GGAUSS:
        rp_trait_ggauss_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_LORENTZ:
        rp_trait_lorentz_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_MOFFAT:
        rp_trait_moffat_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_SECH2:
        rp_trait_sech2_rnd(
                out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_MIXTURE_EXP:
        rp_trait_mixture_exponential_rnd(
                out_s, out_r, out_t,
                rng, consts, params);
        break;
    case RPT_UID_MIXTURE_GAUSS:
        rp_trait_mixture_gauss_rnd(
                out_s, out_r, out_t,
                rng, consts, params);
        break;
    case RPT_UID_MIXTURE_GGAUSS:
        rp_trait_mixture_ggauss_rnd(
                out_s, out_r, out_t,
                rng, consts, params);
        break;
    case RPT_UID_MIXTURE_MOFFAT:
        rp_trait_mixture_moffat_rnd(
                out_s, out_r, out_t,
                rng, consts, params);
        break;
    case RPT_UID_NW_UNIFORM:
        rp_trait_nw_uniform_rnd(
                out_s, out_r, out_t,
                rng, rnidx, rnodes, params);
        break;
    case RPT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic_rnd(
                out_s, out_r, out_t,
                rng, rnidx, rnodes, nrnodes, consts, params);
        break;
    case RPT_UID_SPIRAL:
        rp_trait_spiral_rnd(
                out_s, out_r, out_t,
                rng, rnidx, rnodes, nrnodes, consts, params);
        break;
    case RPT_UID_NW_DISTORTION:
        rp_trait_nw_distortion_rnd(
                out_s, out_r, out_t,
                rng, rnidx, rnodes, nrnodes, params);
        break;
    case RPT_UID_MAP:
        rp_trait_map_rnd(
                out_s, out_r, out_t,
                rng, consts, params);
        break;
    default:
        out_s = NAN;
        out_r = NAN;
        out_t = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
rh_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T r, T z)
{
    switch (uid)
    {
    // The vertical distribution of the density (light or absorbers):
    // probability density functions that integrate to 1.
    case RHT_UID_UNIFORM:
        rh_trait_uniform_pdf(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RHT_UID_EXP:
        rh_trait_exponential_pdf(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RHT_UID_GAUSS:
        rh_trait_gauss_pdf(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RHT_UID_GGAUSS:
        rh_trait_ggauss_pdf(
                out, rnidx, rnodes, nrnodes, r, z, consts, params);
        break;
    case RHT_UID_LORENTZ:
        rh_trait_lorentz_pdf(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RHT_UID_MOFFAT:
        rh_trait_moffat_pdf(
                out, rnidx, rnodes, nrnodes, r, z, consts, params);
        break;
    case RHT_UID_SECH2:
        rh_trait_sech2_pdf(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// The mean of a density height trait (see rh_trait) over the heights
// [z - w, z + w]: what a voxel that spans them takes of a profile, exact
// for a profile thinner than the voxel (see rh_trait_mean_1p); its pdf at
// z for w = 0
template<typename T> constexpr void
rh_trait_mean(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T r, T z, T w)
{
    switch (uid)
    {
    case RHT_UID_UNIFORM:
        rh_trait_mean_1p<uniform_wm_1d_pdf<T>, uniform_wm_1d_cdf<T>>(
                out, rnidx, rnodes, r, z, w, consts, params);
        break;
    case RHT_UID_EXP:
        rh_trait_mean_1p<exponential_1d_pdf<T>, exponential_1d_cdf<T>>(
                out, rnidx, rnodes, r, z, w, consts, params);
        break;
    case RHT_UID_GAUSS:
        rh_trait_mean_1p<gauss_1d_pdf<T>, gauss_1d_cdf<T>>(
                out, rnidx, rnodes, r, z, w, consts, params);
        break;
    case RHT_UID_GGAUSS:
        rh_trait_mean_2p<ggauss_1d_pdf<T>, ggauss_1d_cdf<T>>(
                out, rnidx, rnodes, nrnodes, r, z, w, consts, params);
        break;
    case RHT_UID_LORENTZ:
        rh_trait_mean_1p<lorentz_1d_pdf<T>, lorentz_1d_cdf<T>>(
                out, rnidx, rnodes, r, z, w, consts, params);
        break;
    case RHT_UID_MOFFAT:
        rh_trait_mean_2p<moffat_1d_pdf<T>, moffat_1d_cdf<T>>(
                out, rnidx, rnodes, nrnodes, r, z, w, consts, params);
        break;
    case RHT_UID_SECH2:
        rh_trait_mean_1p<sech2_1d_pdf<T>, sech2_1d_cdf<T>>(
                out, rnidx, rnodes, r, z, w, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
rh_trait_rnd(
        T& out, RNG<T>& rng,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T r)
{
    switch (uid)
    {
    case RHT_UID_UNIFORM:
        rh_trait_uniform_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_EXP:
        rh_trait_exponential_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_GAUSS:
        rh_trait_gauss_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_GGAUSS:
        rh_trait_ggauss_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_LORENTZ:
        rh_trait_lorentz_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_MOFFAT:
        rh_trait_moffat_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RHT_UID_SECH2:
        rh_trait_sech2_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// The speed of a velocity polar trait along its direction
template<typename T> constexpr void
vp_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case VPT_UID_UNIFORM:
        vp_trait_uniform(
                out, params);
        break;
    case VPT_UID_ARCTAN:
        vp_trait_arctan(
                out, r, params);
        break;
    case VPT_UID_BOISSIER:
        vp_trait_boissier(
                out, r, params);
        break;
    case VPT_UID_EPINAT:
        vp_trait_epinat(
                out, r, params);
        break;
    case VPT_UID_LRAMP:
        vp_trait_lramp(
                out, r, params);
        break;
    case VPT_UID_TANH:
        vp_trait_tanh(
                out, r, params);
        break;
    case VPT_UID_POLYEX:
        vp_trait_polyex(
                out, r, params);
        break;
    case VPT_UID_RIX:
        vp_trait_rix(
                out, r, params);
        break;
    case VPT_UID_COURTEAU:
        vp_trait_courteau(
                out, r, params);
        break;
    case VPT_UID_BRANDT:
        vp_trait_brandt(
                out, r, params);
        break;
    case VPT_UID_ISO:
        vp_trait_iso(
                out, r, params);
        break;
    case VPT_UID_NFW:
        vp_trait_nfw(
                out, r, params);
        break;
    case VPT_UID_NW_UNIFORM:
        vp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case VPT_UID_TABLE:
        vp_trait_table(
                out, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case VPT_UID_NW_HARMONIC:
        vp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case VPT_UID_WEDGE:
        vp_trait_wedge(
                out, theta, params);
        break;
    case VPT_UID_BAR:
        vp_trait_bar(
                out, r, theta, params);
        break;
    case VPT_UID_NW_BAR:
        vp_trait_nw_bar(
                out, rnodes, nrnodes, r, theta, params);
        break;
    case VPT_UID_SPIRAL:
        vp_trait_spiral(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// The part v_x, along the azimuth 0, of the velocity of a planar velocity
// polar trait (see VPT_DIRECTION_PLANE; vp_trait gives its part v_y), and
// 0 of the others: for the velocities as vectors (see vp_disk_vector)
template<typename T> constexpr void
vp_trait_plane_x(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    (void)x;
    (void)y;
    T vy = 0;
    switch (uid)
    {
    case VPT_UID_BAR:
        vp_trait_bar_plane(out, vy, r, theta, params);
        break;
    case VPT_UID_NW_BAR:
        vp_trait_nw_bar_plane(out, vy, rnodes, nrnodes, r, theta, params);
        break;
    case VPT_UID_SPIRAL:
        vp_trait_spiral_plane(
                out, vy, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    default:
        out = 0;
        break;
    }
}

// The line-of-sight velocity of the speed v of a velocity polar trait of
// a disk of inclination incl, at the radius r, the azimuth theta and the
// height z of the disk, by the direction of the speed: tangential, v
// cos(theta) sin(incl); radial (positive outwards, in the plane), v
// sin(theta) sin(incl); vertical (positive along z), v cos(incl); away
// from the midplane (along z, on both faces), v sign(z) cos(incl), 0 at
// the midplane; outward (from the centre, along the radius of the
// sphere, sqrt(r^2 + z^2): away from the plane on either side), v (r
// sin(theta) sin(incl) + z cos(incl)) / sqrt(r^2 + z^2); the line of
// sight, v; or for a planar trait (a velocity in the plane), whose v is
// its part along the azimuth 90 degrees, v sin(incl). The tangential and
// radial directions are undefined on the axis, and the outward one at
// the centre, where the velocity is 0.
template<typename T> constexpr T
vp_disk_project(T v, int direction, T r, T theta, T z, T incl)
{
    switch (direction)
    {
    case VPT_DIRECTION_TAN:
        return r == 0 ? 0 : v * (std::cos(theta) * std::sin(incl));
    case VPT_DIRECTION_RAD:
        return r == 0 ? 0 : v * (std::sin(theta) * std::sin(incl));
    case VPT_DIRECTION_VER:
        return v * std::cos(incl);
    case VPT_DIRECTION_AWAY:
        return v * sign(z) * std::cos(incl);
    case VPT_DIRECTION_OUT: {
        const T radius = std::sqrt(r * r + z * z);
        return radius == 0 ? 0 : v * (
                (r * std::sin(theta) * std::sin(incl) + z * std::cos(incl))
                / radius);
    }
    case VPT_DIRECTION_LOS:
        return v;
    case VPT_DIRECTION_PLANE:
        return v * std::sin(incl);
    default:
        assert(false);
        return NAN;
    }
}

// The velocity of the speed v of a velocity polar trait of a disk (see
// vp_disk_project), at the radius r, the azimuth theta and the height z of
// the disk, as a vector of its frame (x and y in its plane, theta from x,
// z along its spin axis), added to out, by the direction of the speed:
// tangential, v (-sin(theta), cos(theta), 0); radial, v (cos(theta),
// sin(theta), 0); vertical, v (0, 0, 1); away from the midplane, v (0, 0,
// sign(z)); outward, v (r cos(theta), r sin(theta), z) / sqrt(r^2 + z^2);
// the line of sight, v (0, sin(incl), cos(incl)), along which
// vp_disk_project projects them all. A planar trait gives its parts in
// the plane instead (see vp_trait_plane_x).
template<typename T> constexpr void
vp_disk_vector(T* out, T v, int direction, T r, T theta, T z, T incl)
{
    switch (direction)
    {
    case VPT_DIRECTION_TAN:
        if (r != 0) {
            out[0] -= v * std::sin(theta);
            out[1] += v * std::cos(theta);
        }
        break;
    case VPT_DIRECTION_RAD:
        if (r != 0) {
            out[0] += v * std::cos(theta);
            out[1] += v * std::sin(theta);
        }
        break;
    case VPT_DIRECTION_VER:
        out[2] += v;
        break;
    case VPT_DIRECTION_AWAY:
        out[2] += v * sign(z);
        break;
    case VPT_DIRECTION_OUT: {
        const T radius = std::sqrt(r * r + z * z);
        if (radius != 0) {
            out[0] += v * r * std::cos(theta) / radius;
            out[1] += v * r * std::sin(theta) / radius;
            out[2] += v * z / radius;
        }
        break;
    }
    case VPT_DIRECTION_LOS:
        out[1] += v * std::sin(incl);
        out[2] += v * std::cos(incl);
        break;
    default:
        assert(false);
        out[0] = out[1] = out[2] = NAN;
        break;
    }
}

// The line-of-sight velocity of the speed v of a velocity polar trait of
// a cone of inclination incl, at the point (x, y, z) of its frame (the
// frame of a thick disk whose spin axis is the axis of the cone) at the
// distance r from its apex, by the direction of the speed: tangential
// (around the axis), v x sin(incl) / R, R the distance from the axis
// (v cos(theta) sin(incl), as on a disk); or outward (from the apex),
// v (y sin(incl) + z cos(incl)) / r. The tangential direction is
// undefined on the axis, and the outward one at the apex, where the
// velocity is 0.
template<typename T> constexpr T
vp_cone_project(T v, int direction, T x, T y, T z, T r, T incl)
{
    switch (direction)
    {
    case VPT_DIRECTION_TAN: {
        const T radius = std::sqrt(x * x + y * y);
        return radius == 0 ? 0 : v * (x * std::sin(incl) / radius);
    }
    case VPT_DIRECTION_OUT:
        return r == 0
                ? 0 : v * ((y * std::sin(incl) + z * std::cos(incl)) / r);
    default:
        assert(false);
        return NAN;
    }
}

// The velocity of the speed v of a velocity polar trait of a cone (see
// vp_cone_project) at the point (x, y, z) of its frame, at the distance r
// from its apex, as a vector of its frame, added to out: tangential, v
// (-y, x, 0) / R; outward, v (x, y, z) / r
template<typename T> constexpr void
vp_cone_vector(T* out, T v, int direction, T x, T y, T z, T r)
{
    switch (direction)
    {
    case VPT_DIRECTION_TAN: {
        const T radius = std::sqrt(x * x + y * y);
        if (radius != 0) {
            out[0] -= v * y / radius;
            out[1] += v * x / radius;
        }
        break;
    }
    case VPT_DIRECTION_OUT:
        if (r != 0) {
            out[0] += v * x / r;
            out[1] += v * y / r;
            out[2] += v * z / r;
        }
        break;
    default:
        assert(false);
        out[0] = out[1] = out[2] = NAN;
        break;
    }
}

// The factors of the velocity and dispersion height traits at the height
// z, with the parameter p of the trait (node-wise if consts[0] is true):
// max(0, |z| - p), which grows linearly above the height p (e.g. a lag,
// with a polar trait of the lag per unit height), and exp(-|z| / p) and
// exp(-z^2 / (2 p^2)), which fall with the scale height p.
template<typename T> constexpr T
h_factor_param(int rnidx, const T* rnodes, T r, const T* consts,
               const T* params)
{
    const bool use_rnodes = consts[0];
    return use_rnodes ? nodewise(r, rnidx, rnodes, params, 0, 1) : params[0];
}

template<typename T> constexpr T
h_factor_linear(T z, T p)
{
    return std::max(T{0}, std::abs(z) - p);
}

template<typename T> constexpr T
h_factor_exponential(T z, T p)
{
    return std::exp(-std::abs(z) / p);
}

template<typename T> constexpr T
h_factor_gauss(T z, T p)
{
    return std::exp(-z * z / (2 * p * p));
}

// The factor of the rotation in the gravity of the galaxy at the height z
// (see traits.VHTraitMass): from its table at the n = consts[1] heights
// from 0 to zmax = consts[0], n node-wise parameters one after the other,
// linear between the heights and between the nodes; above zmax, the
// factor at zmax
template<typename T> constexpr T
h_factor_table(int rnidx, const T* rnodes, int nrnodes, T r, T z,
               const T* consts, const T* params)
{
    const T zmax = consts[0];
    const int n = static_cast<int>(consts[1]);
    // (a height that is not a number gives NaN, so that it is seen: its
    // conversion to int would be undefined)
    if (std::isnan(z))
        return z;
    const T u = std::min(std::abs(z) / zmax, T{1}) * (n - 1);
    const int k = std::min(static_cast<int>(u), n - 2);
    const T t = u - k;
    const T lo = nodewise(r, rnidx, rnodes, params, k * nrnodes, 1);
    const T hi = nodewise(r, rnidx, rnodes, params, (k + 1) * nrnodes, 1);
    return lo + t * (hi - lo);
}

template<typename T> constexpr void
vh_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T r, T z)
{
    switch (uid)
    {
    case VHT_UID_ONE:
        out = 1;
        break;
    case VHT_UID_LINEAR:
        out = h_factor_linear(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    case VHT_UID_EXP:
        out = h_factor_exponential(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    case VHT_UID_GAUSS:
        out = h_factor_gauss(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    case VHT_UID_MASS:
        out = h_factor_table(rnidx, rnodes, nrnodes, r, z, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
dp_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case DPT_UID_UNIFORM:
        dp_trait_uniform(
                out, params);
        break;
    case DPT_UID_EXP:
        dp_trait_exponential(
                out, r, params);
        break;
    case DPT_UID_GAUSS:
        dp_trait_gauss(
                out, r, params);
        break;
    case DPT_UID_GGAUSS:
        dp_trait_ggauss(
                out, r, params);
        break;
    case DPT_UID_LORENTZ:
        dp_trait_lorentz(
                out, r, params);
        break;
    case DPT_UID_MOFFAT:
        dp_trait_moffat(
                out, r, params);
        break;
    case DPT_UID_SECH2:
        dp_trait_sech2(
                out, r, params);
        break;
    case DPT_UID_MIXTURE_EXP:
        dp_trait_mixture_exponential(
                out, r, theta, consts, params);
        break;
    case DPT_UID_MIXTURE_GAUSS:
        dp_trait_mixture_gauss(
                out, r, theta, consts, params);
        break;
    case DPT_UID_MIXTURE_GGAUSS:
        dp_trait_mixture_ggauss(
                out, r, theta, consts, params);
        break;
    case DPT_UID_MIXTURE_MOFFAT:
        dp_trait_mixture_moffat(
                out, r, theta, consts, params);
        break;
    case DPT_UID_NW_UNIFORM:
        dp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case DPT_UID_NW_HARMONIC:
        dp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case DPT_UID_NW_DISTORTION:
        dp_trait_nw_distortion(
                out, rnidx, rnodes, nrnodes, r, theta, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
dh_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T r, T z)
{
    (void)nrnodes;

    switch (uid)
    {
    case DHT_UID_ONE:
        out = 1;
        break;
    case DHT_UID_LINEAR:
        out = h_factor_linear(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    case DHT_UID_EXP:
        out = h_factor_exponential(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    case DHT_UID_GAUSS:
        out = h_factor_gauss(
                z, h_factor_param(rnidx, rnodes, r, consts, params));
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// The vertical distortion traits: a displacement of the midplane, within
// zmax (consts[0]) of the plane of the galaxy: a larger one is limited to
// zmax (the other constants are those of the trait)
template<typename T> constexpr void
zp_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case ZPT_UID_NW_UNIFORM:
        zp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case ZPT_UID_NW_HARMONIC:
        zp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts + 1, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
    const T zmax = consts[0];
    out = std::min(std::max(out, -zmax), zmax);
}

template<typename T> constexpr void
sp_trait(
        T& out,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    (void)consts;
    (void)x;
    (void)y;

    switch (uid)
    {
    case SPT_UID_AZRANGE:
        sp_trait_azrange(
                out, theta, params);
        break;
    case SPT_UID_RRANGE:
        sp_trait_rrange(
                out, r, params);
        break;
    case SPT_UID_NW_AZRANGE:
        sp_trait_nw_azrange(
                out, rnidx, rnodes, nrnodes, r, theta, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

// Evaluate the ntraits traits of a set with Fun (a polar or a height
// trait function, see rp_trait and rh_trait) one at a time, and pass the
// index and the value of each to visit(i, value): no array holds their
// values, so that the kernels keep none on their stack, whatever
// TRAIT_NUM_MAX (on a GPU, it would take registers or local memory)
template<auto Fun, typename T, typename VisitT, typename ...Ts>
constexpr void
traits_each(
        VisitT visit,
        int ntraits, const int* uids,
        const T* cvalues, const int* ccounts,
        const T* pvalues, const int* pcounts,
        int rnidx, const T* rnodes, int nrnodes,
        Ts ...args)
{
    for (int i = 0; i < ntraits; ++i)
    {
        T value = 0;
        Fun(value, uids[i], cvalues, pvalues, rnidx, rnodes, nrnodes, args...);
        visit(i, value);
        cvalues += ccounts[i];
        pvalues += pcounts[i];
    }
}

} // namespace gbkfit
