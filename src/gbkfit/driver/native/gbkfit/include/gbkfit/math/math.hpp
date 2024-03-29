#pragma once

namespace gbkfit {

template<typename T> struct RNG;

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
//  out_x = + x * costheta + y * sintheta;
//  out_y = - x * sintheta + y * costheta;
    out_x = - x * sintheta + y * costheta;
    out_y = - x * costheta - y * sintheta;
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

template<auto FUN, typename T, typename ...Ts> constexpr T
_trunc_1d_fun(T xmin, T xmax, T x, Ts ...args)
{
    return x >= xmin && x <= xmax
            ? FUN(x, args...)
            : 0;
}

template<auto PDF, auto CDF, typename T, typename ...Ts> constexpr T
_trunc_1d_pdf(T xmin, T xmax, T x, Ts ...args)
{
    return x >= xmin && x <= xmax
            ? PDF(x, args...) / (CDF(xmax, args...) - CDF(xmin, args...))
            : 0;
}

template<auto FUN, typename T, typename ...Ts> constexpr T
_trunc_1d_rnd(T xmin, T xmax, RNG<T>& rng, Ts ...args)
{
    T x = 0;
    do {
        x = FUN(rng, args...);
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
    return _trunc_1d_pdf<uniform_1d_pdf<T>, uniform_1d_cdf<T>>(
            xmin, xmax, x, b, c);
}

template<typename T> constexpr T
uniform_1d_rnd(RNG<T>& rng, T b, T c)
{
    return b + rng() * (c - b);
}

template<typename T> constexpr T
uniform_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd<uniform_1d_rnd<T>>(xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
uniform_wm_1d_fun(T x, T a, T b, T c)
{
    return uniform_1d_fun(x, a, b - c, b + c);
}

template<typename T> constexpr T
uniform_wm_1d_fun_trunc(T x, T a, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_fun<uniform_wm_1d_fun<T>>(xmin, xmax, x, a, b, c);
}

template<typename T> constexpr T
uniform_wm_1d_cdf(T x, T b, T c)
{
    return uniform_1d_cdf(x, b - c, b + c);
}

template<typename T> constexpr T
uniform_wm_1d_pdf(T x, T b, T c)
{
    return uniform_1d_pdf(x, b - c, b + c);
}

template<typename T> constexpr T
uniform_wm_1d_pdf_trunc(T x, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_pdf<uniform_1d_pdf<T>, uniform_1d_cdf<T>>(
            xmin, xmax, x, b - c, b + c);
}

template<typename T> constexpr T
uniform_wm_1d_rnd(RNG<T>& rng, T b, T c)
{
    return uniform_1d_rnd(rng, b - c, b + c);
}

template<typename T> constexpr T
uniform_wm_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd<uniform_1d_rnd<T>>(xmin, xmax, rng, b - c, b + c);
}
// todo: verify this is correct
template<typename T, typename TTarget, typename ...Ts> constexpr T
rejection_sampling(TTarget target, RNG<T>& rng, T trunc, Ts... args)
{
    T x=0, y=0, z=0;
    do {
        x = uniform_wm_1d_rnd<T>(rng, 0, trunc);
        y = target(0, args...) * rng();
        z = target(x, args...);
    } while (y > z);
    return x;
}

template<typename T> constexpr T
exponential_1d_fun(T x, T a, T b, T c)
{
    return a * std::exp(-std::abs(x - b) / c);
}

template<typename T> constexpr T
exponential_1d_fun_trunc(T x, T a, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_fun<exponential_1d_fun<T>>(xmin, xmax, x, a, b, c);
}
// todo: implement this
template<typename T> constexpr T
exponential_1d_cdf(T x, T b, T c)
{
    return T{0.5} + T{0.5} * sign(x - b) * (1 - std::exp(-std::abs(x - b) / c));
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
    return _trunc_1d_pdf<exponential_1d_pdf<T>, exponential_1d_cdf<T>>(
            xmin, xmax, x, b, c);
}

template<typename T> constexpr T
exponential_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - T{1};
    return b - c * sign(u) * std::log(1 - 2 * std::abs(u));
}

template<typename T> constexpr T
exponential_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd<exponential_1d_rnd<T>>(xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
gauss_1d_fun(T x, T a, T b, T c)
{
    return a * std::exp(-(x - b) * (x - b) / (2 * c * c));
}

template<typename T> constexpr T
gauss_1d_fun_trunc(T x, T a, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_fun<gauss_1d_fun<T>>(xmin, xmax, x, a, b, c);
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
    return _trunc_1d_pdf<gauss_1d_pdf<T>, gauss_1d_cdf<T>>(
            xmin, xmax, x, b, c);
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
    return _trunc_1d_rnd<gauss_1d_rnd<T>>(xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
ggauss_1d_fun(T x, T a, T b, T c, T d)
{
    return a * std::exp(-std::pow(std::abs(x - b) / c, d));
}

template<typename T> constexpr T
ggauss_1d_fun_trunc(T x, T a, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_fun<ggauss_1d_fun<T>>(xmin, xmax, x, a, b, c, d);
}
// todo: implement this
template<typename T> constexpr T
ggauss_1d_cdf(T x, T b, T c, T d)
{
    (void)x;
    (void)b;
    (void)c;
    (void)d;
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
    return _trunc_1d_pdf<ggauss_1d_pdf<T>, ggauss_1d_cdf<T>>(
            xmin, xmax, x, b, c, d);
}

template<typename T> constexpr T
ggauss_1d_rnd(RNG<T>& rng, T b, T c, T d)
{
    return rejection_sampling(ggauss_1d_pdf<T>, rng, 5 * c, b, c, d);
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
lorentz_1d_fun_trunc(T x, T a, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_fun<lorentz_1d_fun<T>>(xmin, xmax, x, a, b, c);
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
    return _trunc_1d_pdf<lorentz_1d_pdf<T>, lorentz_1d_cdf<T>>(
            xmin, xmax, x, b, c);
}

template<typename T> constexpr T
lorentz_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - T{1};
    return b + c * std::tan(PI<T> * T{0.5} * u);
}

template<typename T> constexpr T
lorentz_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd<lorentz_1d_rnd<T>>(xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
moffat_1d_fun(T x, T a, T b, T c, T d)
{
    return a / std::pow(1 + ((x - b) / c) * ((x - b) / c), d);
}

template<typename T> constexpr T
moffat_1d_fun_trunc(T x, T a, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_fun<moffat_1d_fun<T>>(xmin, xmax, x, a, b, c, d);
}

// todo: implement this
template<typename T> constexpr T
moffat_1d_cdf(T x, T b, T c, T d)
{
    (void)x;
    (void)b;
    (void)c;
    (void)d;
    return 0;
}
// todo: implement this
template<typename T> constexpr T
moffat_1d_pdf(T x, T b, T c, T d)
{
    (void)x;
    (void)b;
    (void)c;
    (void)d;
    return 0;
}

template<typename T> constexpr T
moffat_1d_pdf_trunc(T x, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_pdf<moffat_1d_pdf<T>, moffat_1d_cdf<T>>(
            xmin, xmax, x, b, c, d);
}

template<typename T> constexpr T
moffat_1d_rnd(RNG<T>& rng, T b, T c, T d)
{
    return rejection_sampling(moffat_1d_pdf<T>, rng, 5 * c, b, c, d);
}

template<typename T> constexpr T
moffat_1d_rnd_trunc(RNG<T>& rng, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_rnd<moffat_1d_rnd<T>>(xmin, xmax, rng, b, c, d);
}

template<typename T> constexpr T
sech2_1d_fun(T x, T a, T b, T c)
{
    return a * std::pow(1 / std::cosh((x - b) / c), 2);
}

template<typename T> constexpr T
sech2_1d_fun_trunc(T x, T a, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_fun<sech2_1d_fun<T>>(xmin, xmax, x, a, b, c);
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
    return _trunc_1d_pdf<sech2_1d_pdf<T>, sech2_1d_cdf<T>>(
            xmin, xmax, x, b, c);
}

template<typename T> constexpr T
sech2_1d_rnd(RNG<T>& rng, T b, T c)
{
    T u = rng() * 2 - T{1};
    return b + c * std::atanh(u);
}

template<typename T> constexpr T
sech2_1d_rnd_trunc(RNG<T>& rng, T b, T c, T xmin, T xmax)
{
    return _trunc_1d_rnd<sech2_1d_rnd<T>>(xmin, xmax, rng, b, c);
}

template<typename T> constexpr T
lerp(T x, int idx, const T* xdata, const T* ydata, int offset, int stride)
{
    int idx1 = idx - 1;
    int idx2 = idx;
    T x1 = xdata[idx1];
    T x2 = xdata[idx2];
    T y1 = ydata[offset + idx1 * stride];
    T y2 = ydata[offset + idx2 * stride];
    T t = (x - x1) / (x2 - x1);
    return y1 + t * (y2 - y1);
}

template<typename T> constexpr T
lerp(T x, int index, const T* xdata, const T* ydata)
{
    return lerp(x, index, xdata, ydata, 0, 1);
}

} // namespace gbkfit
