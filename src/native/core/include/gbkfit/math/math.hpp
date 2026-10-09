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

// x to the power n >= 0, by multiplication: std::pow computes an
// integer power of a float in double precision, which is slow on gpus
template<typename T> constexpr T
ipow(T x, int n)
{
    T result = 1;
    for (int i = 0; i < n; ++i)
        result *= x;
    return result;
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
    // Rejection sampling: draw until the value is within the range
    T x = 0;
    do {
        x = FUN(rng, args...);
    } while (x < xmin || x > xmax);
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

// A standard normal random number (Box-Muller)
template<typename T> constexpr T
normal_rnd(RNG<T>& rng)
{
    // In this order on every driver (the operands of an expression are
    // not evaluated in a given order)
    const T u1 = rng();
    const T u2 = rng();
    return std::sqrt(-2 * std::log(u1)) * std::cos(2 * PI<T> * u2);
}

// A random number of the gamma distribution of shape k and scale 1
// (Marsaglia and Tsang 2000; for k < 1, through the shape k + 1)
template<typename T> constexpr T
gamma_rnd(RNG<T>& rng, T k)
{
    const T boost = k < 1 ? std::pow(rng(), 1 / k) : T{1};
    const T d = (k < 1 ? k + 1 : k) - T{1} / 3;
    const T c = 1 / std::sqrt(9 * d);
    while (true)
    {
        const T x = normal_rnd(rng);
        T v = 1 + c * x;
        if (v <= 0)
            continue;
        v = v * v * v;
        const T u = rng();
        if (u < 1 - T{0.0331} * x * x * x * x
                || std::log(u) < T{0.5} * x * x + d * (1 - v + std::log(v)))
            return boost * d * v;
    }
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
    // Inverse transform sampling of the Laplace distribution,
    // with u uniform in (-0.5, 0.5)
    T u = rng() - T{0.5};
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
    // A Gaussian of width 0 is a step at b
    if (c == 0)
        return x < b ? T{0} : T{1};
    return T{0.5} * (1 + std::erf((x - b) / (c * std::sqrt(T{2}))));
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
    return b + c * normal_rnd(rng);
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
// The regularised lower incomplete gamma function P(a, x), for a > 0 and
// x >= 0: by its series for x < a + 1, and otherwise by the continued
// fraction of 1 - P (modified Lentz)
template<typename T> constexpr T
gamma_p(T a, T x)
{
    constexpr int MAX_TERMS = 500;
    constexpr T EPS = sizeof(T) == sizeof(float) ? T{1.2e-7} : T{2.3e-16};
    constexpr T TINY = T{1e-30};
    if (x <= 0)
        return 0;
    const T scale = std::exp(-x + a * std::log(x) - std::lgamma(a));
    if (x < a + 1)
    {
        T ap = a;
        T term = 1 / a;
        T sum = term;
        for (int n = 0; n < MAX_TERMS; ++n)
        {
            ap += 1;
            term *= x / ap;
            sum += term;
            if (std::abs(term) < std::abs(sum) * EPS)
                break;
        }
        return sum * scale;
    }
    T b = x + 1 - a;
    T c = 1 / TINY;
    T d = 1 / b;
    T h = d;
    for (int i = 1; i < MAX_TERMS; ++i)
    {
        const T an = -i * (i - a);
        b += 2;
        d = an * d + b;
        d = std::abs(d) < TINY ? TINY : d;
        c = b + an / c;
        c = std::abs(c) < TINY ? TINY : c;
        d = 1 / d;
        const T delta = d * c;
        h *= delta;
        if (std::abs(delta - 1) < EPS)
            break;
    }
    return 1 - scale * h;
}

// The regularised incomplete beta function I_x(a, b), for a, b > 0 and
// 0 <= x <= 1: by its continued fraction (modified Lentz), for
// x < (a + 1) / (a + b + 2), and otherwise by 1 - I_{1 - x}(b, a)
template<typename T> constexpr T
beta_inc(T a, T b, T x)
{
    constexpr int MAX_TERMS = 500;
    constexpr T EPS = sizeof(T) == sizeof(float) ? T{1.2e-7} : T{2.3e-16};
    constexpr T TINY = T{1e-30};
    if (x <= 0)
        return 0;
    if (x >= 1)
        return 1;
    const bool swap = x > (a + 1) / (a + b + 2);
    if (swap)
    {
        const T a_ = a;
        a = b;
        b = a_;
        x = 1 - x;
    }
    const T scale = std::exp(
            std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
            + a * std::log(x) + b * std::log1p(-x)) / a;
    T c = 1;
    T d = 1 - (a + b) * x / (a + 1);
    d = std::abs(d) < TINY ? TINY : d;
    d = 1 / d;
    T h = d;
    for (int m = 1; m < MAX_TERMS; ++m)
    {
        // The even and the odd step of the fraction
        const T even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
        d = 1 + even * d;
        d = std::abs(d) < TINY ? TINY : d;
        c = 1 + even / c;
        c = std::abs(c) < TINY ? TINY : c;
        d = 1 / d;
        h *= d * c;
        const T odd =
                -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
        d = 1 + odd * d;
        d = std::abs(d) < TINY ? TINY : d;
        c = 1 + odd / c;
        c = std::abs(c) < TINY ? TINY : c;
        d = 1 / d;
        const T delta = d * c;
        h *= delta;
        if (std::abs(delta - 1) < EPS)
            break;
    }
    return swap ? 1 - scale * h : scale * h;
}

// The cdf of the generalised Gaussian exp(-(|x - b| / c)^d) (normalised):
// 1/2 + sign(x - b) P(1 / d, (|x - b| / c)^d) / 2
template<typename T> constexpr T
ggauss_1d_cdf(T x, T b, T c, T d)
{
    const T p = gamma_p(1 / d, std::pow(std::abs(x - b) / c, d));
    return T{0.5} + T{0.5} * (x < b ? -p : p);
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

// A random number of the generalised Gaussian: |x - b| / c is a gamma
// random number of shape 1 / d to the power 1 / d, on either side of b
template<typename T> constexpr T
ggauss_1d_rnd(RNG<T>& rng, T b, T c, T d)
{
    const T distance = c * std::pow(gamma_rnd(rng, 1 / d), 1 / d);
    return rng() < T{0.5} ? b - distance : b + distance;
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

// The cdf of the Moffat profile (1 + ((x - b) / c)^2)^-d (normalised, for
// d > 1/2), a Student's t distribution of 2d - 1 degrees of freedom: with
// u = (x - b) / c, the mass beyond |u| on either side is half of
// I_{1 / (1 + u^2)}(d - 1/2, 1/2) = 1 - I_{u^2 / (1 + u^2)}(1/2, d - 1/2),
// whichever has an argument of at most 1/2 (1 - x loses the precision of
// an x near 1)
template<typename T> constexpr T
moffat_1d_cdf(T x, T b, T c, T d)
{
    const T u2 = ((x - b) / c) * ((x - b) / c);
    const T beyond = u2 > 1
            ? beta_inc(d - T{0.5}, T{0.5}, 1 / (1 + u2))
            : 1 - beta_inc(T{0.5}, d - T{0.5}, u2 / (1 + u2));
    return x < b ? beyond / 2 : 1 - beyond / 2;
}

template<typename T> constexpr T
moffat_1d_pdf(T x, T b, T c, T d)
{
    T a = std::exp(std::lgamma(d) - std::lgamma(d - T{0.5}))
            / (c * std::sqrt(PI<T>));
    return moffat_1d_fun(x, a, b, c, d);
}

template<typename T> constexpr T
moffat_1d_pdf_trunc(T x, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_pdf<moffat_1d_pdf<T>, moffat_1d_cdf<T>>(
            xmin, xmax, x, b, c, d);
}

// A random number of the Moffat profile: (x - b) / c is a normal random
// number over the square root of twice a gamma random number of shape
// d - 1/2 (a Student's t random number over the square root of 2d - 1)
template<typename T> constexpr T
moffat_1d_rnd(RNG<T>& rng, T b, T c, T d)
{
    const T z = normal_rnd(rng);
    return b + c * z / std::sqrt(2 * gamma_rnd(rng, d - T{0.5}));
}

template<typename T> constexpr T
moffat_1d_rnd_trunc(RNG<T>& rng, T b, T c, T d, T xmin, T xmax)
{
    return _trunc_1d_rnd<moffat_1d_rnd<T>>(xmin, xmax, rng, b, c, d);
}

template<typename T> constexpr T
sech2_1d_fun(T x, T a, T b, T c)
{
    const T sech = 1 / std::cosh((x - b) / c);
    return a * sech * sech;
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
