#pragma once

#include "kernels_common.hpp"

namespace gbkfit { namespace host { namespace kernels {

template<typename T>
constexpr T PI = T{3.14159265358979323846};

template<typename T>
constexpr T DEG_TO_RAD = PI<T> / 180;

template<typename T>
constexpr T RAD_TO_DEG = 180 / PI<T>;

template<typename T> constexpr T
deg_to_rad(T deg)
{
    return deg * DEG_TO_RAD<T>;
}

template<typename T> constexpr T
rad_to_deg(T rad)
{
    return rad * RAD_TO_DEG<T>;
}

template<typename T> constexpr T
wrap_angle(T angle)
{
    while(angle < -PI<T>)
        angle += 2 * PI<T>;
    while(angle > +PI<T>)
        angle -= 2 * PI<T>;
    return angle;
}

template <typename T> constexpr int
sign(T x)
{
    return (T{0} < x) - (x < T{0});
}

template<typename T> constexpr void
transform_lh_rotate_x(T& out_y, T& out_z, T y, T z, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_y =   y * costheta + z * sintheta;
    out_z = - y * sintheta + z * costheta;
}

template<typename T> constexpr void
transform_lh_rotate_y(T& out_x, T& out_z, T x, T z, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_x = x * costheta - z * sintheta;
    out_z = x * sintheta + z * costheta;
}

template<typename T> constexpr void
transform_lh_rotate_z(T& out_x, T& out_y, T x, T y, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_x =   x * costheta + y * sintheta;
    out_y = - x * sintheta + y * costheta;
}

template<typename T> constexpr void
transform_rh_rotate_x(T& out_y, T& out_z, T y, T z, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_y = y * costheta - z * sintheta;
    out_z = y * sintheta + z * costheta;
}

template<typename T> constexpr void
transform_rh_rotate_y(T& out_x, T& out_z, T x, T z, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_x =   x * costheta + z * sintheta;
    out_z = - x * sintheta + z * costheta;
}

template<typename T> constexpr void
transform_rh_rotate_z(T& out_x, T& out_y, T x, T y, T theta)
{
    T sintheta = std::sin(theta);
    T costheta = std::cos(theta);
    out_x = x * costheta - y * sintheta;
    out_y = x * sintheta + y * costheta;
}

template<typename PDF, typename CDF, typename T, typename ...Ts> constexpr T
_trunc_1d_pdf(PDF pdf, CDF cdf, T xmin, T xmax, T x, Ts ...args)
{
    return x >= xmin && x <= xmax
            ? pdf(x, args...) / (cdf(xmax, args...) - cdf(xmin, args...))
            : 0;
}

template<typename F, typename T, typename ...Ts> constexpr T
_trunc_1d_rnd(F fun, T xmin, T xmax, RNG<T>& rng, Ts ...args)
{
    T x;
    do {
        x = fun(rng, args...);
    } while (x < xmin && x > xmax);
    return x;
}

template<typename T> constexpr T
uniform_1d_fun(T x, T a, T b, T c)
{
    return x >= b && x <= c ? a : 0;
}

template<typename T> constexpr T
uniform_1d_cdf(T x, T b, T c)
{
    return x < b ? 0 : (x > c ? 1 : ((x - b) / (c - b)));
}

template<typename T> constexpr T
uniform_1d_pdf(T x, T b, T c)
{
    T a = 1 / (c - b);
    return uniform_1d_fun(x, a, b, c);
}

template<typename T> constexpr T
uniform_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf(
            uniform_1d_pdf<T>, uniform_1d_cdf<T>, xmin, xmax, x, b, c);
}

template<typename T> constexpr T
uniform_1d_rnd(RNG<T>& rng, T b, T c)
{    
    return b + rng() * (c - b);
}

template<typename T> constexpr T
uniform_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd(uniform_1d_rnd<T>, xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
exponential_1d_fun(T x, T a, T b, T c)
{
    return a * std::exp(-std::abs(x - b) / c);
}

template<typename T> constexpr T
exponential_1d_cdf(T x, T b, T c)
{
    std::cout << "commented out" << std::endl;
//  return T{0.5} + T{0.5} * sign(x - b) * (1 - std::exp(-std::abs(x - b) / c));
}

template<typename T> constexpr T
exponential_1d_pdf(T x, T b, T c)
{
    T a = 1 / (2 * c);
    return exponential_1d_fun(x, a, b, c);
}

template<typename T> constexpr T
exponential_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf(
            exponential_1d_pdf<T>, exponential_1d_cdf<T>, xmin, xmax, x, b, c);
}

template<typename T> constexpr T
exponential_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - 1.0;
    return b - c * sign(u) * std::log(1 - 2 * std::abs(u));
}

template<typename T> constexpr T
exponential_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd(exponential_1d_rnd<T>, xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
gauss_1d_fun(T x, T a, T b, T c)
{
    return a * std::exp(-(x - b) * (x - b) / (2 * c * c));
}

template<typename T> constexpr T
gauss_1d_cdf(T x, T b, T c)
{
    return T{0.5} * (1 + erf((x - b)/(c * std::sqrt(2))));
}

template<typename T> constexpr T
gauss_1d_pdf(T x, T b, T c)
{
    T a = 1 / (c * std::sqrt(2 * PI<T>));
    return gauss_1d_fun(x, a, b, c);
}

template<typename T> constexpr T
gauss_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf(
            gauss_1d_pdf<T>, gauss_1d_cdf<T>, xmin, xmax, x, b, c);
}

template<typename T> constexpr T
gauss_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u1 = rng();
    T u2 = rng();
    return b + c * std::sqrt(-2 * std::log(u1)) * std::cos(2 * PI<T> * u2);
}

template<typename T> constexpr T
gauss_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd(gauss_1d_rnd<T>, xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
ggauss_1d_fun(T x, T a, T b, T c, T d)
{
    return a * std::exp(-std::pow(std::abs(x - b) / c, d));
}

template<typename T> constexpr T
ggauss_1d_cdf(T x, T b, T c, T d)
{
    (void)x;
    (void)b;
    (void)c;
    (void)d;
    assert(false);
    return 0;
}

template<typename T> constexpr T
ggauss_1d_pdf(T x, T b, T c, T d)
{
    T a = d / (2 * c * std::tgamma(1 / d));
    return ggauss_1d_fun(x, a, b, c, d);
}

template<typename T> constexpr T
ggauss_1d_pdf_trunc(T x, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_pdf(ggauss_1d_pdf<T>, ggauss_1d_cdf<T>,
            xmin, xmax, x, b, c, d);
}

/*
template<auto Target, auto Proposal, typename T, typename ...Ts> constexpr T
_reject_sample_trunc_1d_rnd(T xmin, T xmax, RNG<T>& rng, Ts ...args)
{
    //T a = Proposal()
    T x;
    do {
        x = F(rng, args...);
    } while (x < xmin && x > xmax);
    return x;
}
*/

template<typename T, typename TProposal, typename TTarget> constexpr T
sample_distribution(RNG<T>& rng, T ampl, TProposal proposal, TTarget target)
{
    T x, y, z;
    do {
        x = proposal();
        y = rng() * ampl;
        z = target(x);
    } while (y > z);
    return x;

}


template<typename T> constexpr T
ggauss_1d_rnd(RNG<T>& rng, T b, T c, T d)
{
    T s = 3 * c;
    return sample_distribution(
            rng,
            ggauss_1d_pdf(T{0}, b, c, d),
            [&] () { return uniform_1d_rnd(rng, -s, s); },
            [&] (T x) { return ggauss_1d_pdf(x, b, c, d); });
}

template<typename T> constexpr T
ggauss_1d_rnd_trunc(RNG<T>& rng, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_rnd<ggauss_1d_rnd<T>>(xmin, xmax, rng, b, c, d);
}

template<typename T> constexpr T
lorentz_1d_fun(T x, T a, T b, T c)
{
    return a * c * c / ((x - b) * (x - b) + c * c);
}

template<typename T> constexpr T
lorentz_1d_cdf(T x, T b, T c)
{
    return T{0.5} + std::atan((x - b) / c) / PI<T>;
}

template<typename T> constexpr T
lorentz_1d_pdf(T x, T b, T c)
{
    T a = 1 / (PI<T> * c);
    return lorentz_1d_fun(x, a, b, c);
}

template<typename T> constexpr T
lorentz_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf(
            lorentz_1d_pdf<T>, lorentz_1d_cdf<T>, xmin, xmax, x, b, c);
}

template<typename T> constexpr T
lorentz_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - 1.0;
    return b + c * std::tan(PI<T> * T{0.5} * u);
}

template<typename T> constexpr T
lorentz_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd(lorentz_1d_rnd<T>, xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
moffat_1d_fun(T x, T a, T b, T c, T d)
{
    return a / std::pow(1 + ((x - b) / c) * ((x - b) / c), d);
}

template<typename T> constexpr T
sech2_1d_fun(T x, T a, T b, T c)
{
    return a * std::pow(1 / std::cosh((x - b) / c), 2);
}

template<typename T> constexpr T
sech2_1d_cdf(T x, T b, T c)
{
    return T{0.5} * (1 + std::tanh((x - b) / c));
}

template<typename T> constexpr T
sech2_1d_pdf(T x, T b, T c)
{
    T a = 1 / (2 * c);
    return sech2_1d_fun(x, a, b, c);
}

template<typename T> constexpr T
sech2_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf(
            sech2_1d_pdf<T>, sech2_1d_cdf<T>, xmin, xmax, x, b, c);
}

template<typename T> constexpr T
sech2_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - 1.0;
    return b + c * std::atanh(u);
}

template<typename T> constexpr T
sech2_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd(sech2_1d_rnd<T>, xmin, xmax, rng, b, c);
}

template<typename T> T
interp_linear(T t, T y1, T y2)
{
    return y1 + t * (y2 - y1);
}

template<typename T> T
interp_linear(T x, int index, const T* xdata, const T* ydata)
{
    T x1 = xdata[index-1];
    T x2 = xdata[index];
    T y1 = ydata[index-1];
    T y2 = ydata[index];
    return y1 + (x - x1) / (x2 - x1) * (y2 - y1);
}



template<typename T> size_t
gsl_interp_bsearch(const T x_array[], T x, size_t index_lo, size_t index_hi)
{
    size_t ilo = index_lo;
    size_t ihi = index_hi;
    while(ihi > ilo + 1) {
        size_t i = (ihi + ilo)/2;
        if(x_array[i] > x)
            ihi = i;
        else
            ilo = i;
    }

    return ilo;
}

}}} // namespace gbkfit::host::kernels
