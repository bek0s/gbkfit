﻿#pragma once

#include <gbkfit/gmodel/traits.hpp>
#include "kernels_math.hpp"

namespace gbkfit { namespace host { namespace kernels {

template <typename T> T
nodewise(T x, int idx, const T* xdata, const T* ydata, int offset, int stride)
{
    return lerp(x, idx, xdata, ydata, offset, stride);
}

template<typename T> void
p_trait_uniform(T& out, const T* params)
{
    out = params[0];
}

template<typename T> void
p_trait_exponential(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = exponential_1d_fun(r, a, T{0}, s);
}

template<typename T> void
p_trait_gauss(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = gauss_1d_fun(r, a, T{0}, s);
}

template<typename T> void
p_trait_ggauss(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    T b = params[2];
    out = ggauss_1d_fun(r, a, T{0}, s, b);
}

template<typename T> void
p_trait_lorentz(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = lorentz_1d_fun(r, a, T{0}, s);
}

template<typename T> void
p_trait_moffat(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    T b = params[2];
    out = moffat_1d_fun(r, a, T{0}, s, b);
}

template<typename T> void
p_trait_sech2(T& out, T r, const T* params)
{
    T a = params[0];
    T s = params[1];
    out = sech2_1d_fun(r, a, T{0}, s);
}

template<typename F, typename T> void
p_trait_mixture(F fun, T& out, T r, T theta, const T* consts, const T* params)
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
        T xb = blob_r * std::cos(blob_p);
        T yb = blob_r * std::sin(blob_p);
        T xd = r * std::cos(theta + blob_p - blob_t) - xb;
        T yd = r * std::sin(theta + blob_p - blob_t) - yb;
        T rd = std::sqrt(xd * xd + (yd * yd) / (blob_q * blob_q));
        res += fun(rd, blob_a, T{0}, blob_s, blob_b);
    }
    out = res;
}

template<typename T> void
p_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture(moffat_1d_fun<T>, out, r, theta, consts, params);
}

template<typename T> void
p_trait_mixture_sgauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture(ggauss_1d_fun<T>, out, r, theta, consts, params);
}

template <typename T> void
p_trait_nw_uniform(T& out, int nidx, const T* nodes, T r, const T* params)
{
    out = nodewise(r, nidx, nodes, params, 0, 1);
}

template <typename T> void
p_trait_nw_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* consts, const T* params)
{
    int k = std::rint(consts[0]);
    T a =     nodewise(r, nidx, nodes, params,      0, 1);
    T p = k ? nodewise(r, nidx, nodes, params, nnodes, 1) * DEG_TO_RAD<T> : 0;
    out = a * std::cos(k * (theta - p));
}

template<typename T> void
p_trait_nw_distortion(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* params)
{
    T a = nodewise(r, nidx, nodes, params,          0, 1);
    T p = nodewise(r, nidx, nodes, params,     nnodes, 1) * DEG_TO_RAD<T>;
    T s = nodewise(r, nidx, nodes, params, 2 * nnodes, 1);
    T t = wrap_angle(theta - p);
    out = a * std::exp(-(t * t * r * r) / (2 * s * s));
}

template<typename T> void
rp_trait_sample_polar_coords(T& out_r, T& out_t, RNG<T>& rng, T rmin, T rmax)
{
    out_r = std::sqrt(rmax * rmax + (rmin * rmin - rmax * rmax) * rng());
    out_t = 2 * PI<T> * rng();
}

template<typename T> void
rp_trait_sample_polar_coords_nw(
        T& out_r, T& out_t,
        RNG<T>& rng, int nidx, const T* nodes)
{
    T rsep = nodes[1] - nodes[0];
    T rmin = nodes[nidx] - rsep * T{0.5};
    T rmax = nodes[nidx] + rsep * T{0.5};
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
}

template<typename T> void
rp_trait_uniform(T& out, const T* params)
{
    p_trait_uniform(out, params);
}

template<typename T> void
rp_trait_uniform_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* nodes, int nnodes)
{
    T rsep = nodes[1] - nodes[0];
    T rmin = nodes[0] - rsep;
    T rmax = nodes[nnodes - 1] + rsep;
    rp_trait_sample_polar_coords(out_r, out_t, rng, rmin, rmax);
}

template<typename T> void
rp_trait_exponential(T& out, T r, const T* params)
{
    p_trait_exponential(out, r, params);
}

template<typename T> void
rp_trait_exponential_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_gauss(T& out, T r, const T* params)
{
    p_trait_gauss(out, r, params);
}

template<typename T> void
rp_trait_gauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_ggauss(T& out, T r, const T* params)
{
    p_trait_ggauss(out, r, params);
}

template<typename T> void
rp_trait_ggauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_lorentz(T& out, T r, const T* params)
{
    p_trait_lorentz(out, r, params);
}

template<typename T> void
rp_trait_lorentz_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_moffat(T& out, T r, const T* params)
{
    p_trait_moffat(out, r, params);
}

template<typename T> void
rp_trait_moffat_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_sech2(T& out, T r, const T* params)
{
    p_trait_sech2(out, r, params);
}

template<typename T> void
rp_trait_sech2_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_sgauss(out, r, theta, consts, params);
}

template<typename T> void
rp_trait_mixture_ggauss_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* consts, const T* params)
{
    (void)out_r;
    (void)out_t;
    (void)rng;
    (void)consts;
    (void)params;
}

template<typename T> void
rp_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_moffat(out, r, theta, consts, params);
}

template<typename T> void
rp_trait_mixture_moffat_rnd(
        T& out_r, T& out_t, RNG<T>& rng, const T* consts, const T* params)
{
    (void)out_r;
    (void)out_t;
    (void)rng;
    (void)consts;
    (void)params;
}

template<typename T> void
rp_trait_nw_uniform(
        T& out,
        int nidx, const T* nodes, T r,
        const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
}

template<typename T> void
rp_trait_nw_uniform_rnd(
        T& out_r, T& out_t, RNG<T>& rng, int nidx, const T* nodes)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);
}

template<typename T> void
rp_trait_nw_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
}

template<typename T> void
rp_trait_nw_harmonic_rnd(
        T& out_s, T& out_r, T& out_t,
        RNG<T>& rng, int nidx, const T* nodes, int nnodes,
        const T* consts, const T* params)
{
    rp_trait_sample_polar_coords_nw(out_r, out_t, rng, nidx, nodes);

    T k = consts[0];
    T a = params[         nidx];
    T p = params[nnodes + nidx];

    T value_rnd = a * (rng() * 2 - T{1});
    T value_fun = a * std::cos(k * (out_t - p));
    out_s = value_rnd > value_fun ? -1 : 1;
}

template<typename T> void
rp_trait_nw_distortion(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* params)
{
    p_trait_nw_distortion(out, nidx, nodes, nnodes, r, theta, params);
}

template<typename T> void
rp_trait_nw_distortion_rnd(
        T& out_r, T& out_t,
        RNG<T>& rng, int nidx, const T* nodes, int nnodes,
        const T* params)
{
    T p = params[1 * nnodes + nidx] * DEG_TO_RAD<T>;
    T s = params[2 * nnodes + nidx];
    T rsep = nodes[1] - nodes[0];
    T rmin = nodes[nidx] - rsep * T{0.5};
    T rmax = nodes[nidx] + rsep * T{0.5};
    out_r = std::sqrt(rmax * rmax + (rmin * rmin - rmax * rmax) * rng());
//  out_t = 2 * PI<T> * rng();
    out_t = gauss_1d_rnd(rng, p, s / nodes[nidx]);
}

template<typename T> void
rh_trait_uniform(T& out, T z, const T* params)
{
    T size = params[0];
    out = uniform_1d_pdf(z, -size/2, size/2);
}

template<typename T> void
rh_trait_uniform_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    out = uniform_1d_rnd(rng, -size/2, size/2);
}

template<typename T> void
rh_trait_exponential(T& out, T z, const T* params)
{
    T size = params[0];
    out = exponential_1d_pdf(z, T{0}, size);
}

template<typename T> void
rh_trait_exponential_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    out = exponential_1d_rnd(rng, T{0}, size);
}

template<typename T> void
rh_trait_gauss(T& out, T z, const T* params)
{
    T size = params[0];
    out = gauss_1d_pdf(z, T{0}, size);
}

template<typename T> void
rh_trait_gauss_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    out = gauss_1d_rnd(rng, T{0}, size);
}

template<typename T> void
rh_trait_ggauss(T& out, T z, const T* params)
{
    T size = params[0];
    T shape = params[1];
    out = ggauss_1d_pdf(z, T{0}, size, shape);
}

template<typename T> void
rh_trait_ggauss_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    T shape = params[1];
    out = ggauss_1d_rnd(rng, T{0}, size, shape);
}

template<typename T> void
rh_trait_lorentz(T& out, T z, const T* params)
{
    T size = params[0];
    out = lorentz_1d_pdf(z, T{0}, size);
}

template<typename T> void
rh_trait_lorentz_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    out = lorentz_1d_rnd(rng, T{0}, size);
}

template<typename T> void
rh_trait_sech2(T& out, T z, const T* params)
{
    T size = params[0];
    out = sech2_1d_pdf(z, T{0}, size);
}

template<typename T> void
rh_trait_sech2_rnd(T& out, RNG<T>& rng, const T* params)
{
    T size = params[0];
    out = sech2_1d_rnd(rng, T{0}, size);
}

template<typename T> void
vp_trait_make_tan(T& out, T theta, T incl)
{
    out *= std::cos(theta) * std::sin(incl);
}

template<typename T> void
vp_trait_make_rad(T& out, T theta, T incl)
{
    out *= std::sin(theta) * std::sin(incl);
}

template<typename T> void
vp_trait_make_ver(T& out, T incl)
{
    out *= std::cos(incl);
}

template<typename T> void
vp_trait_tan_uniform(T& out, T theta, T incl, const T* params)
{
    T vt = params[0];
    out = vt;
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_tan_arctan(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = vt * (2/PI<T>) * std::atan(r/rt);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_tan_boissier(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out =  vt * (1 - std::exp(-r/rt));
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_tan_epinat(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    T a = params[2];
    T b = params[3];
    out = vt * std::pow(r/rt, b) / (1 + std::pow(r/rt, a));
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_tan_flat(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = r < rt ? vt * r/rt : vt;
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_tan_tanh(T& out, T r, T theta, T incl, const T* params)
{
    T rt = params[0];
    T vt = params[1];
    out = vt * std::tanh(r/rt);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_nw_tan_uniform(
        T& out,
        int nidx, const T* nodes, T r, T theta, T incl,
        const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_nw_tan_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
    vp_trait_make_tan(out, theta, incl);
}

template<typename T> void
vp_trait_nw_rad_uniform(
        T& out,
        int nidx, const T* nodes, T r, T theta, T incl,
        const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
    vp_trait_make_rad(out, theta, incl);
}

template<typename T> void
vp_trait_nw_rad_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
    vp_trait_make_rad(out, theta, incl);
}

template<typename T> void
vp_trait_nw_ver_uniform(
        T& out,
        int nidx, const T* nodes, T r, T incl,
        const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
    vp_trait_make_ver(out, incl);
}

template<typename T> void
vp_trait_nw_ver_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta, T incl,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
    vp_trait_make_ver(out, incl);
}

template<typename T> void
vp_trait_nw_los_uniform(T& out, int nidx, const T* nodes, T r, const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
}

template<typename T> void
vp_trait_nw_los_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
}

template<typename T> void
dp_trait_uniform(T& out, const T* params)
{
    p_trait_uniform(out, params);
}

template<typename T> void
dp_trait_exponential(T& out, T r, const T* params)
{
    p_trait_exponential(out, r, params);
}

template<typename T> void
dp_trait_gauss(T& out, T r, const T* params)
{
    p_trait_gauss(out, r, params);
}

template<typename T> void
dp_trait_ggauss(T& out, T r, const T* params)
{
    p_trait_ggauss(out, r, params);
}

template<typename T> void
dp_trait_lorentz(T& out, T r, const T* params)
{
    p_trait_lorentz(out, r, params);
}

template<typename T> void
dp_trait_moffat(T& out, T r, const T* params)
{
    p_trait_moffat(out, r, params);
}

template<typename T> void
dp_trait_sech2(T& out, T r, const T* params)
{
    p_trait_sech2(out, r, params);
}

template<typename T> void
dp_trait_mixture_ggauss(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_sgauss(out, r, theta, consts, params);
}

template<typename T> void
dp_trait_mixture_moffat(T& out, T r, T theta, const T* consts, const T* params)
{
    p_trait_mixture_moffat(out, r, theta, consts, params);
}

template<typename T> void
dp_trait_nw_uniform(T& out, int nidx, const T* nodes, T r, const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
}

template<typename T> void
dp_trait_nw_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
}

template<typename T> void
dp_trait_nw_distortion(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* params)
{
    p_trait_nw_distortion(out, nidx, nodes, nnodes, r, theta, params);
}

template<typename T> void
wp_trait_nw_uniform(T& out, int nidx, const T* nodes, T r, const T* params)
{
    p_trait_nw_uniform(out, nidx, nodes, r, params);
}

template<typename T> void
wp_trait_nw_harmonic(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* consts, const T* params)
{
    p_trait_nw_harmonic(out, nidx, nodes, nnodes, r, theta, consts, params);
}

template<typename T> void
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

template<typename T> void
sp_trait_azrange(T& out, T theta, const T* params)
{
    T p = params[0] * DEG_TO_RAD<T>;
    T s = params[1] * DEG_TO_RAD<T>;
    _azrange(out, theta, p, s);
}

template<typename T> void
sp_trait_nw_azrange(
        T& out,
        int nidx, const T* nodes, int nnodes, T r, T theta,
        const T* params)
{
    T p = nodewise(r, nidx, nodes, params,      0, 1) * DEG_TO_RAD<T>;
    T s = nodewise(r, nidx, nodes, params, nnodes, 1) * DEG_TO_RAD<T>;
    _azrange(out, theta, p, s);
}

template<typename T> void
rp_trait(T& out,
         int uid, const T* consts, const T* params,
         int nidx, const T* nodes, int nnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    out = NAN;
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
                out, nidx, nodes, r, params);
        break;
    case RP_TRAIT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic(
                out, nidx, nodes, nnodes, r, theta, consts, params);
        break;
    case RP_TRAIT_UID_NW_DISTORTION:
        rp_trait_nw_distortion(
                out, nidx, nodes, nnodes, r, theta, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
rh_trait(T& out, int uid, const T* consts, const T* params, T z)
{
    (void)consts;

    out = NAN;
    switch (uid)
    {
    case RH_TRAIT_UID_UNIFORM:
        rh_trait_uniform(
                out, z, params);
        break;
    case RH_TRAIT_UID_EXPONENTIAL:
        rh_trait_exponential(
                out, z, params);
        break;
    case RH_TRAIT_UID_GAUSS:
        rh_trait_gauss(
                out, z, params);
        break;
    case RH_TRAIT_UID_GGAUSS:
        rh_trait_ggauss(
                out, z, params);
        break;
    case RH_TRAIT_UID_LORENTZ:
        rh_trait_lorentz(
                out, z, params);
        break;
    case RH_TRAIT_UID_SECH2:
        rh_trait_sech2(
                out, z, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
rp_trait_rnd(
        T& out_s, T& out_r, T& out_t, RNG<T>& rng,
        int uid, const T* consts, const T* params,
        int nidx, const T* nodes, int nnodes)
{
    out_s = 1;
    out_r = NAN;
    out_t = NAN;
    switch (uid)
    {
    case RP_TRAIT_UID_UNIFORM:
        rp_trait_uniform_rnd(
                out_r, out_t, rng, nodes, nnodes);
        break;
    case RP_TRAIT_UID_EXPONENTIAL:
        rp_trait_exponential_rnd(
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_GAUSS:
        rp_trait_gauss_rnd(
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_GGAUSS:
        rp_trait_ggauss_rnd(
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_LORENTZ:
        rp_trait_lorentz_rnd(
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_MOFFAT:
        rp_trait_moffat_rnd(
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_SECH2:
        rp_trait_sech2_rnd(
                out_r, out_t, rng, nidx, nodes);
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
                out_r, out_t, rng, nidx, nodes);
        break;
    case RP_TRAIT_UID_NW_HARMONIC:
        rp_trait_nw_harmonic_rnd(
                out_s, out_r, out_t, rng, nidx, nodes, nnodes, consts, params);
        break;
    case RP_TRAIT_UID_NW_DISTORTION:
        rp_trait_nw_distortion_rnd(
                out_r, out_t, rng, nidx, nodes, nnodes, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
rh_trait_rnd(
        T& out, RNG<T>& rng,
        int uid, const T* consts, const T* params)
{
    (void)consts;

    out = NAN;
    switch (uid)
    {
    case RH_TRAIT_UID_UNIFORM:
        rh_trait_uniform_rnd(
                out, rng, params);
        break;
    case RH_TRAIT_UID_EXPONENTIAL:
        rh_trait_exponential_rnd(
                out, rng, params);
        break;
    case RH_TRAIT_UID_GAUSS:
        rh_trait_gauss_rnd(
                out, rng, params);
        break;
    case RH_TRAIT_UID_GGAUSS:
        rh_trait_ggauss_rnd(
                out, rng, params);
        break;
    case RH_TRAIT_UID_LORENTZ:
        rh_trait_lorentz_rnd(
                out, rng, params);
        break;
    case RH_TRAIT_UID_SECH2:
        rh_trait_sech2_rnd(
                out, rng, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
vp_trait(T& out,
         int uid, const T* consts, const T* params,
         int nidx, const T* nodes, int nnodes,
         T x, T y, T r, T theta, T incl)
{
    (void)x;
    (void)y;

    out = NAN;
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
    case VP_TRAIT_UID_NW_TAN_UNIFORM:
        vp_trait_nw_tan_uniform(
                out, nidx, nodes, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_NW_TAN_HARMONIC:
        vp_trait_nw_tan_harmonic(
                out, nidx, nodes, nnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_RAD_UNIFORM:
        vp_trait_nw_rad_uniform(
                out, nidx, nodes, r, theta, incl, params);
        break;
    case VP_TRAIT_UID_NW_RAD_HARMONIC:
        vp_trait_nw_rad_harmonic(
                out, nidx, nodes, nnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_VER_UNIFORM:
        vp_trait_nw_ver_uniform(
                out, nidx, nodes, r, incl, params);
        break;
    case VP_TRAIT_UID_NW_VER_HARMONIC:
        vp_trait_nw_ver_harmonic(
                out, nidx, nodes, nnodes, r, theta, incl, consts, params);
        break;
    case VP_TRAIT_UID_NW_LOS_UNIFORM:
        vp_trait_nw_los_uniform(
                out, nidx, nodes, r, params);
        break;
    case VP_TRAIT_UID_NW_LOS_HARMONIC:
        vp_trait_nw_los_harmonic(
                out, nidx, nodes, nnodes, r, theta, consts, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
vh_trait(T& out, int uid, const T* consts, const T* params, T z)
{
    (void)z;
    (void)consts;
    (void)params;

    out = NAN;
    switch (uid)
    {
    case VH_TRAIT_UID_ONE:
        out = 1;
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
dp_trait(T& out,
         int uid, const T* consts, const T* params,
         int nidx, const T* nodes, int nnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    out = NAN;
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
                out, nidx, nodes, r, params);
        break;
    case DP_TRAIT_UID_NW_HARMONIC:
        dp_trait_nw_harmonic(
                out, nidx, nodes, nnodes, r, theta, consts, params);
        break;
    case DP_TRAIT_UID_NW_DISTORTION:
        dp_trait_nw_distortion(
                out, nidx, nodes, nnodes, r, theta, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
dh_trait(T& out, int uid, const T* consts, const T* params, T z)
{
    (void)z;
    (void)consts;
    (void)params;

    out = NAN;
    switch (uid)
    {
    case DH_TRAIT_UID_ONE:
        out = 1;
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
wp_trait(T& out,
         int uid, const T* consts, const T* params,
         int nidx, const T* nodes, int nnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;

    out = NAN;
    switch (uid)
    {
    case WP_TRAIT_UID_NW_UNIFORM:
        wp_trait_nw_uniform(
                out, nidx, nodes, r, params);
        break;
    case WP_TRAIT_UID_NW_HARMONIC:
        wp_trait_nw_harmonic(
                out, nidx, nodes, nnodes, r, theta, consts, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename T> void
sp_trait(T& out,
         int uid, const T* consts, const T* params,
         int nidx, const T* nodes, int nnodes,
         T x, T y, T r, T theta)
{
    (void)x;
    (void)y;
    (void)consts;

    out = NAN;
    switch (uid)
    {
    case SP_TRAIT_UID_AZRANGE:
        sp_trait_azrange(
                out, theta, params);
        break;
    case SP_TRAIT_UID_NW_AZRANGE:
        sp_trait_nw_azrange(
                out, nidx, nodes, nnodes, r, theta, params);
        break;
    default:
        assert(false);
        break;
    }
}

template<typename F, typename T, typename ...Ts> void
p_traits(
        F fun,
        T* out,
        int ntraits, const int* uids,
        const T* cvalues, const int* ccounts,
        const T* pvalues, const int* pcounts,
        int nidx, const T* nodes, int nnodes,
        Ts ...args)
{
    for(int i = 0; i < ntraits; ++i)
    {
        fun(out[i], uids[i], cvalues, pvalues, nidx, nodes, nnodes, args...);
        cvalues += ccounts[i];
        pvalues += pcounts[i];
    }
}

template<typename F, typename T> void
h_traits(
        F fun,
        T* out,
        int ntraits, const int* uids,
        const T* cvalues, const int* ccounts,
        const T* pvalues, const int* pcounts,
        T z)
{
    for(int i = 0; i < ntraits; ++i)
    {
        fun(out[i], uids[i], cvalues, pvalues, z);
        cvalues += ccounts[i];
        pvalues += pcounts[i];
    }
}

}}} // namespace gbkfit::host::kernels
