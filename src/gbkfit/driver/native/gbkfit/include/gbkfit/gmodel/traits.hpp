#pragma once

#include "gbkfit/math/math.hpp"

namespace gbkfit {

// Density polar traits
constexpr int RP_TRAIT_UID_UNIFORM = 1;
constexpr int RP_TRAIT_UID_EXPONENTIAL = 2;
constexpr int RP_TRAIT_UID_GAUSS = 3;
constexpr int RP_TRAIT_UID_GGAUSS = 4;
constexpr int RP_TRAIT_UID_LORENTZ = 5;
constexpr int RP_TRAIT_UID_MOFFAT = 6;
constexpr int RP_TRAIT_UID_SECH2 = 7;
constexpr int RP_TRAIT_UID_MIXTURE_GGAUSS = 8;
constexpr int RP_TRAIT_UID_MIXTURE_MOFFAT = 9;
constexpr int RP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int RP_TRAIT_UID_NW_HARMONIC = 102;
constexpr int RP_TRAIT_UID_NW_DISTORTION = 103;

// Density height traits
constexpr int RH_TRAIT_UID_UNIFORM = 1;
constexpr int RH_TRAIT_UID_EXPONENTIAL = 2;
constexpr int RH_TRAIT_UID_GAUSS = 3;
constexpr int RH_TRAIT_UID_GGAUSS = 4;
constexpr int RH_TRAIT_UID_LORENTZ = 5;
constexpr int RH_TRAIT_UID_MOFFAT = 6;
constexpr int RH_TRAIT_UID_SECH2 = 7;

// Velocity polar traits
constexpr int VP_TRAIT_UID_TAN_UNIFORM = 1;
constexpr int VP_TRAIT_UID_TAN_ARCTAN = 2;
constexpr int VP_TRAIT_UID_TAN_BOISSIER = 3;
constexpr int VP_TRAIT_UID_TAN_EPINAT = 4;
constexpr int VP_TRAIT_UID_TAN_LRAMP = 5;
constexpr int VP_TRAIT_UID_TAN_TANH = 6;
constexpr int VP_TRAIT_UID_TAN_POLYEX = 7;
constexpr int VP_TRAIT_UID_TAN_RIX = 8;
constexpr int VP_TRAIT_UID_NW_TAN_UNIFORM = 101;
constexpr int VP_TRAIT_UID_NW_TAN_HARMONIC = 102;
constexpr int VP_TRAIT_UID_NW_RAD_UNIFORM = 103;
constexpr int VP_TRAIT_UID_NW_RAD_HARMONIC = 104;
constexpr int VP_TRAIT_UID_NW_VER_UNIFORM = 105;
constexpr int VP_TRAIT_UID_NW_VER_HARMONIC = 106;
constexpr int VP_TRAIT_UID_NW_LOS_UNIFORM = 107;
constexpr int VP_TRAIT_UID_NW_LOS_HARMONIC = 108;

// Velocity height traits
constexpr int VH_TRAIT_UID_ONE = 1;

// Dispersion polar traits
constexpr int DP_TRAIT_UID_UNIFORM = 1;
constexpr int DP_TRAIT_UID_EXPONENTIAL = 2;
constexpr int DP_TRAIT_UID_GAUSS = 3;
constexpr int DP_TRAIT_UID_GGAUSS = 4;
constexpr int DP_TRAIT_UID_LORENTZ = 5;
constexpr int DP_TRAIT_UID_MOFFAT = 6;
constexpr int DP_TRAIT_UID_SECH2 = 7;
constexpr int DP_TRAIT_UID_MIXTURE_GGAUSS = 8;
constexpr int DP_TRAIT_UID_MIXTURE_MOFFAT = 9;
constexpr int DP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int DP_TRAIT_UID_NW_HARMONIC = 102;
constexpr int DP_TRAIT_UID_NW_DISTORTION = 103;

// Dispersion height traits
constexpr int DH_TRAIT_UID_ONE = 1;

// Vertical distortion polar traits
constexpr int ZP_TRAIT_UID_NW_UNIFORM = 101;
constexpr int ZP_TRAIT_UID_NW_HARMONIC = 102;

// Selection polar traits
constexpr int SP_TRAIT_UID_AZRANGE = 1;
constexpr int SP_TRAIT_UID_NW_AZRANGE = 101;

// Weight polar traits
constexpr int WP_TRAIT_UID_CRANGE = 1;

template<typename T> constexpr T
nodewise(T x, int idx, const T* xdata, const T* ydata, int offset, int stride)
{
    return lerp(x, idx, xdata, ydata, offset, stride);
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
        T xb = blob_r * std::cos(blob_p);
        T yb = blob_r * std::sin(blob_p);
        T xd = r * std::cos(theta + blob_p - blob_t) - xb;
        T yd = r * std::sin(theta + blob_p - blob_t) - yb;
        T rd = std::sqrt(xd * xd + (yd * yd) / (blob_q * blob_q));
        res += FUN(rd, blob_a, 0, blob_s);
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
p_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_2p<ggauss_1d_fun<T>>(out, r, theta, consts, params);
}

template<typename T> constexpr void
p_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_2p<moffat_1d_fun<T>>(out, r, theta, consts, params);
}

template <typename T> constexpr void
p_trait_nw_uniform(T& out, int rnidx, const T* rnodes, T r, const T* params)
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
    T p = k ? nodewise(r, rnidx, rnodes, params, nrnodes, 1) * DEG_TO_RAD<T> : 0;
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
rp_trait_sample_polar_coords(T& out_r, T& out_t, RNG<T>& rng, T rmin, T rmax)
{
    out_r = std::sqrt(rmax * rmax + (rmin * rmin - rmax * rmax) * rng());
    out_t = 2 * PI<T> * rng();
}

template<typename T> constexpr void
rp_trait_sample_polar_coords_nw(
        T& out_r, T& out_t,
        RNG<T>& rng, int rnidx, const T* rnodes)
{
    T rsep = rnodes[1] - rnodes[0];
    T rmin = rnodes[rnidx] - rsep * T{0.5};
    T rmax = rnodes[rnidx] + rsep * T{0.5};
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
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
    T rsep = rnodes[1] - rnodes[0] * 2; // TODO: why?
    T rmin = rnodes[0] - rsep;
    T rmax = rnodes[nrnodes - 1] + rsep;
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
}

template<typename T> constexpr void
rp_trait_exponential(T& out, T r, const T* params)
{
    p_trait_exponential(out, r, params);
}

template<typename T> constexpr void
rp_trait_exponential_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_gauss(T& out, T r, const T* params)
{
    p_trait_gauss(out, r, params);
}

template<typename T> constexpr void
rp_trait_gauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    //out_r = gauss_1d_rnd<T>(rng, 0, 20);
    //out_t = 2 * PI<T> * rng();
    //out_r = 20 * std::sqrt(-2 * std::log(rng()));
    //out_t = 2 * PI<T> * rng();
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_ggauss(T& out, T r, const T* params)
{
    p_trait_ggauss(out, r, params);
}

template<typename T> constexpr void
rp_trait_ggauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_lorentz(T& out, T r, const T* params)
{
    p_trait_lorentz(out, r, params);
}

template<typename T> constexpr void
rp_trait_lorentz_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_moffat(T& out, T r, const T* params)
{
    p_trait_moffat(out, r, params);
}

template<typename T> constexpr void
rp_trait_moffat_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_sech2(T& out, T r, const T* params)
{
    p_trait_sech2(out, r, params);
}

template<typename T> constexpr void
rp_trait_sech2_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_ggauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_ggauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* consts, const T* params)
{
    (void)out_r;
    (void)out_t;
    (void)rng;
    (void)consts;
    (void)params;
}

template<typename T> constexpr void
rp_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_moffat(out, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_mixture_moffat_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* consts, const T* params)
{
    (void)out_r;
    (void)out_t;
    (void)rng;
    (void)consts;
    (void)params;
}

template<typename T> constexpr void
rp_trait_nw_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

template<typename T> constexpr void
rp_trait_nw_uniform_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int rnidx, const T* rnodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
}

template<typename T> constexpr void
rp_trait_nw_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
}

template<typename T> constexpr void
rp_trait_nw_harmonic_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes,
        const T* consts, const T* params)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, rnidx, rnodes);
    int k = std::rint(consts[0]);
    T a = params[rnidx];
    T p = params[rnidx + nrnodes];
    T value_rnd = a * (rng() * 2 - T{1});
    T value_fun = a * std::cos(k * (out_t - p));
    out_s = value_rnd > value_fun ? -1 : 1;
}

template<typename T> constexpr void
rp_trait_nw_distortion(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    p_trait_nw_distortion(out, rnidx, rnodes, nrnodes, r, theta, params);
}

template<typename T> constexpr void
rp_trait_nw_distortion_rnd(
        T& out_r, T& out_t,
        RNG<T>& rng, int rnidx, const T* rnodes, int nrnodes,
        const T* params)
{
    T p = params[1 * nrnodes + rnidx] * DEG_TO_RAD<T>;
    T s = params[2 * nrnodes + rnidx];
    T rsep = rnodes[1] - rnodes[0];
    T rmin = rnodes[rnidx] - rsep * T{0.5};
    T rmax = rnodes[rnidx] + rsep * T{0.5};
//  out_r = std::sqrt(rmax * rmax + (rmin * rmin - rmax * rmax) * rng());
    out_r = rmin + rng() * rsep;
//  out_t = 2 * PI<T> * rng();
//  out_t = gauss_1d_rnd(rng, p, s / nodes[nidx]);
    out_t = p + gauss_1d_rnd<T>(rng, 0, 1) * s / out_r;
}

template<typename T> constexpr void
rh_trait_uniform(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T p1 = use_n ? nodewise(r, rnidx, rnodes, params, 0, 1) : params[0];
    out = uniform_wm_1d_pdf<T>(z, 0, p1);
}

template<typename T> constexpr void
rh_trait_uniform_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, T r,
        const T* consts, const T* params)
{
    bool use_n = consts[0];
    T p1 = use_n ? nodewise(r, rnidx, rnodes, params, 0, 1) : params[0];
    out = uniform_wm_1d_rnd<T>(rng, 0, p1);
}

template<typename T> constexpr void
rh_trait_exponential(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? exponential_1d_pdf<T>(z, 0, p1)
            : exponential_1d_pdf_trunc<T>(z, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_exponential_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? exponential_1d_rnd<T>(rng, 0, p1)
            : exponential_1d_rnd_trunc<T>(rng, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_gauss(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? gauss_1d_pdf<T>(z, 0, p1)
            : gauss_1d_pdf_trunc<T>(z, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_gauss_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? gauss_1d_rnd<T>(rng, 0, p1)
            : gauss_1d_rnd_trunc<T>(rng, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_ggauss(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    T p2 = use_n == 0 ? params[1] : nodewise(r, rnidx, rnodes, params, nrnodes, 1);
    out = trunc == 0
            ? ggauss_1d_pdf<T>(z, 0, p1, p2)
            : ggauss_1d_pdf_trunc<T>(z, 0, p1, p2, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_ggauss_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    T p2 = use_n == 0 ? params[1] : nodewise(r, rnidx, rnodes, params, nrnodes, 1);
    out = trunc == 0
            ? ggauss_1d_rnd<T>(rng, 0, p1, p2)
            : ggauss_1d_rnd_trunc<T>(rng, 0, p1, p2, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_lorentz(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? lorentz_1d_pdf<T>(z, 0, p1)
            : lorentz_1d_pdf_trunc<T>(z, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_lorentz_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? lorentz_1d_rnd<T>(rng, 0, p1)
            : lorentz_1d_rnd_trunc<T>(rng, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_moffat(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    T p2 = use_n == 0 ? params[1] : nodewise(r, rnidx, rnodes, params, nrnodes, 1);
    out = trunc == 0
            ? moffat_1d_pdf<T>(z, 0, p1, p2)
            : moffat_1d_pdf_trunc<T>(z, 0, p1, p2, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_moffat_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, int nrnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    T p2 = use_n == 0 ? params[1] : nodewise(r, rnidx, rnodes, params, nrnodes, 1);
    out = trunc == 0
            ? moffat_1d_rnd<T>(rng, 0, p1, p2)
            : moffat_1d_rnd_trunc<T>(rng, 0, p1, p2, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_sech2(
        T& out,
        int rnidx, const T* rnodes, T r, T z,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? sech2_1d_pdf<T>(z, 0, p1)
            : sech2_1d_pdf_trunc<T>(z, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
rh_trait_sech2_rnd(
        T& out, RNG<T>& rng,
        int rnidx, const T* rnodes, T r,
        const T* consts, const T* params)
{
    T use_n = consts[0];
    T trunc = consts[1];
    T p1 = use_n == 0 ? params[0] : nodewise(r, rnidx, rnodes, params, 0, 1);
    out = trunc == 0
            ? sech2_1d_rnd<T>(rng, 0, p1)
            : sech2_1d_rnd_trunc<T>(rng, 0, p1, -trunc*p1, trunc*p1);
}

template<typename T> constexpr void
vp_trait_make_tan(T& out, T theta, T incl)
{
    out *= std::cos(theta) * std::sin(incl);
}

template<typename T> constexpr void
vp_trait_make_rad(T& out, T theta, T incl)
{
    out *= std::sin(theta) * std::sin(incl);
}

template<typename T> constexpr void
vp_trait_make_ver(T& out, T incl)
{
    out *= std::cos(incl);
}

template<typename T> constexpr void
vp_trait_tan_uniform(T& out, T theta, T incl, const T* params)
{
    T vt = params[0];
    out = vt;
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_arctan(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = vt * (2/PI<T>) * std::atan(r/rt);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_boissier(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out =  vt * (1 - std::exp(-r/rt));
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_epinat(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    T a = params[2];
    T g = params[3];
    out = vt * std::pow(r/rt, g) / (1 + std::pow(r/rt, a));
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_flat(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = r < rt ? vt * r/rt : vt;
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_tanh(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = vt * std::tanh(r/rt);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_polyex(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    T a = params[2];
    out = vt * (1 - std::exp(-r/rt)) * (1 + a * r/rt);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_tan_rix(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    T b = params[2];
    T g = params[3];
    out = vt * std::pow(1 + r/rt, b) * std::pow(1 + std::pow(r/rt, -g), -1/g);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_nw_tan_uniform(
        T& out,
        int rnidx, const T* rnodes, T r, T theta, T incl,
        const T* params)
{
    if (r == 0) { out = 0; return; }
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_nw_tan_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    if (r == 0) { out = 0; return; }
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_nw_rad_uniform(
        T& out,
        int rnidx, const T* rnodes, T r, T theta, T incl,
        const T* params)
{
    if (r == 0) { out = 0; return; }
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
    vp_trait_make_rad(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_nw_rad_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    if (r == 0) { out = 0; return; }
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
    vp_trait_make_rad(out, theta, incl);
}

template<typename T> constexpr void
vp_trait_nw_ver_uniform(
        T& out,
        int rnidx, const T* rnodes, T r, T incl,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
    vp_trait_make_ver(out, incl);
}

template<typename T> constexpr void
vp_trait_nw_ver_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
    vp_trait_make_ver(out, incl);
}

template<typename T> constexpr void
vp_trait_nw_los_uniform(
        T& out,
        int rnidx, const T* rnodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, rnidx, rnodes, r, params);
}

template<typename T> constexpr void
vp_trait_nw_los_harmonic(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, rnidx, rnodes, nrnodes, r, theta, consts, params);
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
dp_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_ggauss(out, r, theta, consts, params);
}

template<typename T> constexpr void
dp_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
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

template<typename T> constexpr void
_azrange(T& out, T theta, T p, T s)
{
    T pmin = p - s/2;
    T pmax = p + s/2;
    p = wrap_angle(p);
    pmin = wrap_angle(pmin);
    pmax = wrap_angle(pmax);

    if (pmin < pmax) {
        out = (theta >= pmin && theta <= pmax) ? 1 : 0;
    } else {
        out = (theta <= pmin && theta >= pmax) ? 0 : 1;
    }
}

template<typename T> constexpr void
sp_trait_azrange(T& out, T theta, const T* params)
{
    T p = params[0] * DEG_TO_RAD<T>;
    T s = params[1] * DEG_TO_RAD<T>;
    _azrange(out, theta, p, s);
}

template<typename T> constexpr void
sp_trait_nw_azrange(
        T& out,
        int rnidx, const T* rnodes, int nrnodes, T r, T theta,
        const T* params)
{
    T p = nodewise(r, rnidx, rnodes, params,       0, 1) * DEG_TO_RAD<T>;
    T s = nodewise(r, rnidx, rnodes, params, nrnodes, 1) * DEG_TO_RAD<T>;
    _azrange(out, theta, p, s);
}

template<typename T> constexpr void
wp_trait_crange(T& out, T theta, const T* params)
{
    // TODO
    out = 7;
}

template<typename T> constexpr void
rp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case RP_TRAIT_UID_UNIFORM:
        rp_trait_uniform(
                out, params);
        break;
    case RP_TRAIT_UID_EXPONENTIAL:
        rp_trait_exponential(
                out, r, params);
        break;
    case RP_TRAIT_UID_GAUSS:
        rp_trait_gauss(
                out, r, params);
        break;
    case RP_TRAIT_UID_GGAUSS:
        rp_trait_ggauss(
                out, r, params);
        break;
    case RP_TRAIT_UID_LORENTZ:
        rp_trait_lorentz(
                out, r, params);
        break;
    case RP_TRAIT_UID_MOFFAT:
        rp_trait_moffat(
                out, r, params);
        break;
    case RP_TRAIT_UID_SECH2:
        rp_trait_sech2(
                out, r, params);
        break;
    case RP_TRAIT_UID_MIXTURE_GGAUSS:
        rp_trait_mixture_ggauss(
                out, r, theta, consts, params);
        break;
    case RP_TRAIT_UID_MIXTURE_MOFFAT:
        rp_trait_mixture_moffat(
                out, r, theta, consts, params);
        break;
    case RP_TRAIT_UID_NW_UNIFORM:
        rp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case RP_TRAIT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case RP_TRAIT_UID_NW_DISTORTION:
        rp_trait_nw_distortion(
                out, rnidx, rnodes, nrnodes, r, theta, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
rp_trait_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int uid, const T* consts, const T* params,
        int rnidx, const T* rnodes, int nrnodes)
{
    switch (uid)
    {
    case RP_TRAIT_UID_UNIFORM:
        rp_trait_uniform_rnd(
                out_r, out_t, rng, rnodes, nrnodes);
        break;
    case RP_TRAIT_UID_EXPONENTIAL:
        rp_trait_exponential_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_GAUSS:
        rp_trait_gauss_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_GGAUSS:
        rp_trait_ggauss_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_LORENTZ:
        rp_trait_lorentz_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_MOFFAT:
        rp_trait_moffat_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_SECH2:
        rp_trait_sech2_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_MIXTURE_GGAUSS:
        rp_trait_mixture_ggauss_rnd(
                out_r, out_t, rng, consts, params);
        break;
    case RP_TRAIT_UID_MIXTURE_MOFFAT:
        rp_trait_mixture_moffat_rnd(
                out_r, out_t, rng, consts, params);
        break;
    case RP_TRAIT_UID_NW_UNIFORM:
        rp_trait_nw_uniform_rnd(
                out_r, out_t, rng, rnidx, rnodes);
        break;
    case RP_TRAIT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic_rnd(
                out_s, out_r, out_t, rng, rnidx, rnodes, nrnodes, consts, params);
        break;
    case RP_TRAIT_UID_NW_DISTORTION:
        rp_trait_nw_distortion_rnd(
                out_r, out_t, rng, rnidx, rnodes, nrnodes, params);
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
rh_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T r, T z)
{
    switch (uid)
    {
    case RH_TRAIT_UID_UNIFORM:
        rh_trait_uniform(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_EXPONENTIAL:
        rh_trait_exponential(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_GAUSS:
        rh_trait_gauss(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_GGAUSS:
        rh_trait_ggauss(
                out, rnidx, rnodes, nrnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_LORENTZ:
        rh_trait_lorentz(
                out, rnidx, rnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_MOFFAT:
        rh_trait_moffat(
                out, rnidx, rnodes, nrnodes, r, z, consts, params);
        break;
    case RH_TRAIT_UID_SECH2:
        rh_trait_sech2(
                out, rnidx, rnodes, r, z, consts, params);
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
    case RH_TRAIT_UID_UNIFORM:
        rh_trait_uniform_rnd(
                out, rng, rnidx, rnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_EXPONENTIAL:
        rh_trait_exponential_rnd(
                out, rng, rnidx, rnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_GAUSS:
        rh_trait_gauss_rnd(
                out, rng, rnidx, rnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_GGAUSS:
        rh_trait_ggauss_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_LORENTZ:
        rh_trait_lorentz_rnd(
                out, rng, rnidx, rnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_MOFFAT:
        rh_trait_moffat_rnd(
                out, rng, rnidx, rnodes, nrnodes, r, consts, params);
        break;
    case RH_TRAIT_UID_SECH2:
        rh_trait_sech2_rnd(
                out, rng, rnidx, rnodes, r, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
vp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta, T incl)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case VP_TRAIT_UID_TAN_UNIFORM:
        vp_trait_tan_uniform(
                out, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_ARCTAN:
        vp_trait_tan_arctan(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_BOISSIER:
        vp_trait_tan_boissier(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_EPINAT:
        vp_trait_tan_epinat(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_LRAMP:
        vp_trait_tan_flat(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_TANH:
        vp_trait_tan_tanh(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_POLYEX:
        vp_trait_tan_polyex(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_TAN_RIX:
        vp_trait_tan_rix(
                out, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_NW_TAN_UNIFORM:
        vp_trait_nw_tan_uniform(
                out, rnidx, rnodes, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_NW_TAN_HARMONIC:
        vp_trait_nw_tan_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_RAD_UNIFORM:
        vp_trait_nw_rad_uniform(
                out, rnidx, rnodes, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_NW_RAD_HARMONIC:
        vp_trait_nw_rad_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_VER_UNIFORM:
        vp_trait_nw_ver_uniform(
                out, rnidx, rnodes, r, incl, params);
        break;
    case VP_TRAIT_UID_NW_VER_HARMONIC:
        vp_trait_nw_ver_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_LOS_UNIFORM:
        vp_trait_nw_los_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case VP_TRAIT_UID_NW_LOS_HARMONIC:
        vp_trait_nw_los_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
vh_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T r, T z)
{
    (void)consts;
    (void)params;
    (void)rnidx;
    (void)rnodes;
    (void)nrnodes;
    (void)r;
    (void)z;

    switch (uid)
    {
    case VH_TRAIT_UID_ONE:
        out = 1;
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
dp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case DP_TRAIT_UID_UNIFORM:
        dp_trait_uniform(
                out, params);
        break;
    case DP_TRAIT_UID_EXPONENTIAL:
        dp_trait_exponential(
                out, r, params);
        break;
    case DP_TRAIT_UID_GAUSS:
        dp_trait_gauss(
                out, r, params);
        break;
    case DP_TRAIT_UID_GGAUSS:
        dp_trait_ggauss(
                out, r, params);
        break;
    case DP_TRAIT_UID_LORENTZ:
        dp_trait_lorentz(
                out, r, params);
        break;
    case DP_TRAIT_UID_MOFFAT:
        dp_trait_moffat(
                out, r, params);
        break;
    case DP_TRAIT_UID_SECH2:
        dp_trait_sech2(
                out, r, params);
        break;
    case DP_TRAIT_UID_MIXTURE_GGAUSS:
        dp_trait_mixture_ggauss(
                out, r, theta, consts, params);
        break;
    case DP_TRAIT_UID_MIXTURE_MOFFAT:
        dp_trait_mixture_moffat(
                out, r, theta, consts, params);
        break;
    case DP_TRAIT_UID_NW_UNIFORM:
        dp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case DP_TRAIT_UID_NW_HARMONIC:
        dp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    case DP_TRAIT_UID_NW_DISTORTION:
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
dh_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T r, T z)
{
    (void)consts;
    (void)params;
    (void)rnidx;
    (void)rnodes;
    (void)nrnodes;
    (void)r;
    (void)z;

    switch (uid)
    {
    case DH_TRAIT_UID_ONE:
        out = 1;
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
zp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    switch (uid)
    {
    case ZP_TRAIT_UID_NW_UNIFORM:
        zp_trait_nw_uniform(
                out, rnidx, rnodes, r, params);
        break;
    case ZP_TRAIT_UID_NW_HARMONIC:
        zp_trait_nw_harmonic(
                out, rnidx, rnodes, nrnodes, r, theta, consts, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
sp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;
    (void)consts;

    switch (uid)
    {
    case SP_TRAIT_UID_AZRANGE:
        sp_trait_azrange(
                out, theta, params);
        break;
    case SP_TRAIT_UID_NW_AZRANGE:
        sp_trait_nw_azrange(
                out, rnidx, rnodes, nrnodes, r, theta, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<typename T> constexpr void
wp_trait(T& out,
         int uid, const T* consts, const T* params,
         int rnidx, const T* rnodes, int nrnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;
    (void)consts;

    switch (uid)
    {
    case WP_TRAIT_UID_CRANGE:
        wp_trait_crange(
                out, theta, params);
        break;
    default:
        out = NAN;
        assert(false);
        break;
    }
}

template<auto FUN, typename T, typename ...Ts> constexpr void
p_traits(
        T* out,
        int ntraits, const int* uids,
        const T* cvalues, const int* ccounts,
        const T* pvalues, const int* pcounts,
        int rnidx, const T* rnodes, int nrnodes,
        Ts ...args)
{
    for(int i = 0; i < ntraits; ++i)
    {
        FUN(out[i], uids[i], cvalues, pvalues, rnidx, rnodes, nrnodes, args...);
        cvalues += ccounts[i];
        pvalues += pcounts[i];
    }
}

template<auto FUN, typename T, typename ...Ts> constexpr void
h_traits(
        T* out,
        int ntraits, const int* uids,
        const T* cvalues, const int* ccounts,
        const T* pvalues, const int* pcounts,
        int rnidx, const T* rnodes, int nrnodes,
        Ts ...args)
{
    for(int i = 0; i < ntraits; ++i)
    {
        FUN(out[i], uids[i], cvalues, pvalues, rnidx, rnodes, nrnodes, args...);
        cvalues += ccounts[i];
        pvalues += pcounts[i];
    }
}

} // namespace gbkfit
