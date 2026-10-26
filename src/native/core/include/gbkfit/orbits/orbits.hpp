#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "gbkfit/gmodel/gmodels.hpp"
#include "gbkfit/math/math.hpp"
#include "gbkfit/orbits/orbit.hpp"

namespace gbkfit {

// The cubic Hermite basis at t in [0, 1] (h: the weights of the value at
// 0, the slope at 0, the value at 1 and the slope at 1) and its
// derivatives (d)
template<typename T> constexpr void
orbits_hermite_basis(T t, T* h, T* d)
{
    const T t2 = t * t;
    const T t3 = t2 * t;
    h[0] = 2 * t3 - 3 * t2 + 1;
    h[1] = t3 - 2 * t2 + t;
    h[2] = -2 * t3 + 3 * t2;
    h[3] = t3 - t2;
    d[0] = 6 * t2 - 6 * t;
    d[1] = 3 * t2 - 4 * t + 1;
    d[2] = -6 * t2 + 6 * t;
    d[3] = 3 * t2 - 2 * t;
}

// The cell j of the nodes of b of a potential, and the position of b in
// it (in [0, 1])
template<typename T> constexpr int
orbits_potential_cell_b(const PotentialArgs<T>& p, T b, T& s)
{
    const T pb = b * (2 * (p.nb - 1)) / PI<T>;
    const int j = std::min(std::max(int(pb), 0), p.nb - 2);
    s = pb - j;
    return j;
}

// The potential Phi along b at the radius of node i of a potential (see
// PotentialArgs), its slope along u (g), and their derivatives along b
template<typename T> constexpr void
orbits_potential_row(
        const PotentialArgs<T>& p, int i, T b,
        T& phi, T& phi_b, T& g, T& g_b)
{
    T s, h[4], d[4];
    const int j = orbits_potential_cell_b(p, b, s);
    orbits_hermite_basis(s, h, d);
    const T db = PI<T> / (2 * (p.nb - 1));
    phi = phi_b = g = g_b = 0;
    for (int c = 0; c < 2; ++c)
    {
        const T* v = p.values + 4 * (i * p.nb + j + c);
        phi += h[2 * c] * v[0] + h[2 * c + 1] * v[2] * db;
        phi_b += d[2 * c] * v[0] + d[2 * c + 1] * v[2] * db;
        g += h[2 * c] * v[1] + h[2 * c + 1] * v[3] * db;
        g_b += d[2 * c] * v[1] + d[2 * c + 1] * v[3] * db;
    }
    phi_b /= db;
    g_b /= db;
}

// The potential Phi at (u, b) (u = ln r) and its derivatives along u and
// b (see PotentialArgs)
template<typename T> constexpr void
orbits_potential(
        const PotentialArgs<T>& p, T u, T b,
        T& phi, T& phi_u, T& phi_b)
{
    const T u0 = p.range[0];
    const T du = p.range[1];
    const T un = u0 + (p.nr - 1) * du;
    if (u < u0 || u > un)
    {
        // (beyond the table: e is r^2 / r0^2 below it and rn / r above)
        const bool below = u < u0;
        T g, g_b;
        orbits_potential_row(p, below ? 0 : p.nr - 1, b, phi, phi_b, g, g_b);
        const T e = below ? std::exp(2 * (u - u0)) : std::exp(un - u);
        const T f = below ? (e - 1) / 2 : 1 - e;
        phi += g * f;
        phi_b += g_b * f;
        phi_u = g * e;
        return;
    }
    const T pu = (u - u0) / du;
    const int i = std::min(int(pu), p.nr - 2);
    const T t = pu - i;
    T s, ht[4], dt[4], hs[4], ds[4];
    const int j = orbits_potential_cell_b(p, b, s);
    orbits_hermite_basis(t, ht, dt);
    orbits_hermite_basis(s, hs, ds);
    const T db = PI<T> / (2 * (p.nb - 1));
    phi = phi_u = phi_b = 0;
    for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 2; ++c)
    {
        // (the value, the slopes and the cross slope at the node, per
        // cell)
        const T* v = p.values + 4 * ((i + a) * p.nb + j + c);
        const T f = v[0];
        const T fu = v[1] * du;
        const T fb = v[2] * db;
        const T fub = v[3] * du * db;
        const T vt = ht[2 * a], st = ht[2 * a + 1];
        const T vs = hs[2 * c], ss = hs[2 * c + 1];
        const T dvt = dt[2 * a], dst = dt[2 * a + 1];
        const T dvs = ds[2 * c], dss = ds[2 * c + 1];
        phi += vt * vs * f + vt * ss * fb + st * vs * fu + st * ss * fub;
        phi_u += dvt * vs * f + dvt * ss * fb + dst * vs * fu
                + dst * ss * fub;
        phi_b += vt * dvs * f + vt * dss * fb + st * dvs * fu
                + st * dss * fub;
    }
    phi_u /= du;
    phi_b /= db;
}

// The potential at the point x of the frame of a spheroid (arcsec), the
// force there (minus its gradient: per unit mass, (km/s)^2 / arcsec), and
// |Phi_u| / r^2 (the square of the frequency of the circular orbit of its
// radius, for a spherical potential)
template<typename T> constexpr T
orbits_gravity(
        const PotentialArgs<T>& p, const T* x, T* force,
        T& omega2)
{
    const T R2 = x[0] * x[0] + x[1] * x[1];
    const T r2 = R2 + x[2] * x[2];
    if (r2 == 0)
    {
        // (the centre: the limit of the harmonic core below the table)
        T phi, phi_b, g, g_b;
        orbits_potential_row(p, 0, T{0}, phi, phi_b, g, g_b);
        force[0] = force[1] = force[2] = 0;
        omega2 = std::abs(g) * std::exp(-2 * p.range[0]);
        return phi - g / 2;
    }
    const T R = std::sqrt(R2);
    const T u = std::log(r2) / 2;
    const T b = std::atan2(std::abs(x[2]), R);
    T phi, phi_u, phi_b;
    orbits_potential(p, u, b, phi, phi_u, phi_b);
    // F = -Phi_u grad(u) - Phi_b grad(b), grad(u) = x / r^2 and grad(b) =
    // (-|z| x / R, -|z| y / R, sign(z) R) / r^2 (on the axis, Phi_b is 0)
    const T radial = -phi_u / r2;
    const T polar = R > 0 ? phi_b * std::abs(x[2]) / (R * r2) : 0;
    force[0] = (radial + polar) * x[0];
    force[1] = (radial + polar) * x[1];
    force[2] = radial * x[2] - phi_b * std::copysign(R, x[2]) / r2;
    omega2 = std::abs(phi_u) / r2;
    return phi;
}

// The potential and the force at point i of points (x, y, z, arcsec, in
// the frame of a spheroid): potential[i] and forces[3i] to forces[3i + 2]
template<typename T> constexpr void
orbits_potential_point(
        int i, const PotentialArgs<T>& p, const T* points,
        T* potential, T* forces)
{
    const T x[3] = {points[3 * i], points[3 * i + 1], points[3 * i + 2]};
    T force[3], omega2;
    potential[i] = orbits_gravity(p, x, force, omega2);
    for (int k = 0; k < 3; ++k)
        forces[3 * i + k] = force[k];
}

// An orbit at a time: its position (arcsec), velocity (km/s), the force
// there, its potential, and the rate of its steps (see OrbitArgs)
template<typename T>
struct OrbitPoint
{
    T x[3];
    T v[3];
    T force[3];
    T phi;
    T rate;
};

// The rate of the steps of an orbit point (see OrbitArgs), of the square
// of the circular frequency of its radius (see orbits_gravity)
template<typename T> constexpr T
orbits_rate(const OrbitPoint<T>& point, T omega2)
{
    const T r2 = point.x[0] * point.x[0] + point.x[1] * point.x[1]
            + point.x[2] * point.x[2];
    const T v2 = point.v[0] * point.v[0] + point.v[1] * point.v[1]
            + point.v[2] * point.v[2];
    return r2 > 0 ? std::sqrt(omega2 + v2 / r2) : std::sqrt(omega2);
}

// The force, the potential and the rate of the steps at the position and
// velocity of an orbit point
template<typename T> constexpr void
orbits_point_gravity(const PotentialArgs<T>& p, OrbitPoint<T>& point)
{
    T omega2;
    point.phi = orbits_gravity(p, point.x, point.force, omega2);
    point.rate = orbits_rate(point, omega2);
}

// A kick-drift-kick leapfrog step of an orbit of dt, from start to end
template<typename T> constexpr void
orbits_step(
        const PotentialArgs<T>& p, const OrbitPoint<T>& start, T dt,
        OrbitPoint<T>& end)
{
    T half[3];
    for (int k = 0; k < 3; ++k)
    {
        half[k] = start.v[k] + start.force[k] * dt / 2;
        end.x[k] = start.x[k] + half[k] * dt;
    }
    T omega2;
    end.phi = orbits_gravity(p, end.x, end.force, omega2);
    for (int k = 0; k < 3; ++k)
        end.v[k] = half[k] + end.force[k] * dt / 2;
    end.rate = orbits_rate(end, omega2);
}

// The position and the velocity of an orbit at the fraction tau of a step
// of dt from start to end: cubic Hermite in time, of the positions and
// the velocities at its ends
template<typename T> constexpr void
orbits_interpolate(
        const OrbitPoint<T>& start, const OrbitPoint<T>& end, T dt,
        T tau, T* x, T* v)
{
    T h[4], d[4];
    orbits_hermite_basis(tau, h, d);
    for (int k = 0; k < 3; ++k)
    {
        x[k] = h[0] * start.x[k] + h[1] * dt * start.v[k]
                + h[2] * end.x[k] + h[3] * dt * end.v[k];
        v[k] = (d[0] * start.x[k] + d[2] * end.x[k]) / dt
                + d[1] * start.v[k] + d[3] * end.v[k];
    }
}

// The index of the interval of edges (n + 1, increasing) that holds a
// value, or -1 beyond them
template<typename T> constexpr int
orbits_interval(const T* edges, int n, T value)
{
    if (!(value >= edges[0] && value < edges[n]))
        return -1;
    int lo = 0, hi = n;
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        (value < edges[mid] ? hi : lo) = mid;
    }
    return lo;
}

// Add the terms gamma h_n of a velocity of the given weight in a
// Gauss-Hermite series (of van der Marel & Franx 1993) to out[n], n in
// [0, order]: 2 sqrt(pi) weight alpha(w) H_n(w), w the velocity in the
// series ((v - mean) / dispersion) and alpha(w) = exp(-w^2 / 2) /
// sqrt(2 pi) (the sums over the velocities of a distribution are its
// gamma h_n: of a distribution gamma alpha(w) / dispersion (1 + sum h_n
// H_n(w)), by the orthogonality of the H_n under alpha^2), with AddFunT
template<auto AddFunT, typename T, typename S> constexpr void
orbits_gauss_hermite(T w, T weight, int order, S* out)
{
    const T a = weight * std::sqrt(T{2}) * std::exp(-w * w / 2);
    // (H_0 = 1, H_1 = sqrt(2) w, and H_{n+1} = (sqrt(2) w H_n - sqrt(n)
    // H_{n-1}) / sqrt(n + 1))
    T previous = 0;
    T current = 1;
    for (int n = 0; n <= order; ++n)
    {
        AddFunT(out + n, a * current);
        const T next = (std::sqrt(T{2}) * w * current
                - std::sqrt(T(n)) * previous) / std::sqrt(T(n + 1));
        previous = current;
        current = next;
    }
}

// The period (arcsec / (km/s)) of the circular orbit in the plane of an
// energy ((km/s)^2) in a potential (see PotentialArgs): 2 pi R / v_c of
// the radius R of the energy Phi(R) + v_c^2 / 2 (v_c^2 = Phi_u), its ln
// linear in the energy between the radii of the table; that of the first
// or the last radius beyond them
template<typename T> constexpr T
orbits_circular_period(const PotentialArgs<T>& p, T energy)
{
    // (the energy and the ln of the period of the radius of node i)
    const auto node = [&p](int i, T& e, T& ln_period) {
        const T* v = p.values + 4 * i * p.nb;
        const T r = std::exp(p.range[0] + i * p.range[1]);
        e = v[0] + v[1] / 2;
        ln_period = std::log(2 * PI<T> * r / std::sqrt(v[1]));
    };
    T e0, ln0, e1, ln1;
    node(0, e0, ln0);
    node(p.nr - 1, e1, ln1);
    if (!(energy > e0))
        return std::exp(ln0);
    if (!(energy < e1))
        return std::exp(ln1);
    int lo = 0, hi = p.nr - 1;
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        T e, ln_period;
        node(mid, e, ln_period);
        (energy < e ? hi : lo) = mid;
    }
    node(lo, e0, ln0);
    node(hi, e1, ln1);
    return std::exp(ln0 + (ln1 - ln0) * (energy - e0) / (e1 - e0));
}

// Add to a value (see OrbitRows: the records of an orbit that one thread
// writes)
template<typename T> constexpr void
orbits_add(T* addr, T val)
{
    *addr += val;
}

// Where the records of an orbit go, and those of its twin (see
// OrbitArgs), as values of type S: its moments in the targets [first,
// last), those of target t at moments + (t - first) (order + 2), and its
// time in the cells, that of cell (i, j) at cells + i ncell_b + j (none
// if null); those of its twin twin_moments and twin_cells after them
template<typename S>
struct OrbitRows
{
    S* moments = nullptr;
    int first = 0;
    int last = 0;
    long long twin_moments = 0;
    S* cells = nullptr;
    long long twin_cells = 0;
};

// Add the light of a copy of a sample of an orbit (see OrbitArgs) at the
// velocity v on the line of sight to its moments in target t (at row; see
// OrbitRows), of the share of the copy in the target, with AddFunT
template<auto AddFunT, typename T, typename S> constexpr void
orbits_record_moments(S* row, int t, T v, T share, const OrbitArgs<T>& a)
{
    AddFunT(row, share);
    orbits_gauss_hermite<AddFunT>(
            (v - a.target_v[t]) / a.target_s[t], share, a.order, row + 1);
}

// Add light at the velocity v on the line of sight to the spectrum of
// target t (see OrbitArgs), in its channel, if any, with AddFunT (atomic
// where the orbits are recorded in parallel)
template<auto AddFunT, typename T> constexpr void
orbits_record_spectrum(int t, T v, T light, const OrbitArgs<T>& a)
{
    const T channel = a.channel_scale * v + a.channel_offset;
    if (!(channel >= 0 && channel < a.nchannels))
        return;
    AddFunT(a.spectra + (long long)(channel) * a.ntargets + t, light);
}

// Record a copy of a sample of orbit k on the sky: at the point (x, y)
// of the sky with the velocity v on the line of sight, of the given
// weight, in the targets of rows (see OrbitRows) of the cell of each
// grid that holds it, with its twin (see OrbitArgs); AddFunT adds to the
// spectra, and AddRowFunT to the rows
template<auto AddFunT, auto AddRowFunT, typename T, typename S>
constexpr void
orbits_record_sky(
        int k, T x, T y, T v, T weight,
        const OrbitArgs<T>& a, const OrbitRows<S>& rows)
{
    const long long orbit = a.twins ? 2 * k : k;
    const int width = a.order + 2;
    for (int s = 0; s < a.nsky; ++s)
    {
        const T* grid = a.grids + 6 * s;
        const T gx = grid[0] * x + grid[1] * y + grid[2];
        const T gy = grid[3] * x + grid[4] * y + grid[5];
        if (!(gx >= 0 && gx < a.grid_sizes[2 * s]
                && gy >= 0 && gy < a.grid_sizes[2 * s + 1]))
            continue;
        const long long cell = a.grid_cells[s]
                + (long long)(gy) * a.grid_sizes[2 * s] + (long long)(gx);
        for (long long e = a.offsets[cell]; e < a.offsets[cell + 1]; ++e)
        {
            const int t = a.targets[e];
            if (t < rows.first || t >= rows.last)
                continue;
            const T share = weight * a.shares[e];
            if (rows.moments)
            {
                S* row = rows.moments + (long long)(t - rows.first) * width;
                orbits_record_moments<AddRowFunT>(row, t, v, share, a);
                if (a.twins)
                    orbits_record_moments<AddRowFunT>(
                            row + rows.twin_moments, t, -v, share, a);
            }
            if (a.spectra)
            {
                orbits_record_spectrum<AddFunT>(
                        t, v, share * a.light[orbit], a);
                if (a.twins)
                    orbits_record_spectrum<AddFunT>(
                            t, -v, share * a.light[orbit + 1], a);
            }
        }
    }
}

// The projection of a point of the frame of a spheroid on the sky, from
// its centre (see OrbitArgs): the rows of the matrix of
// transform_cpos_posa_incl_inverse (the sky x and y, and the line of
// sight, positive away from us)
template<typename T> constexpr void
orbits_sky(const OrbitArgs<T>& a, T (&sky)[3][3])
{
    for (int c = 0; c < 3; ++c)
    {
        T e[3] = {0, 0, 0};
        e[c] = 1;
        transform_cpos_posa_incl_inverse(
                e[0], e[1], e[2], T{0}, T{0},
                a.posa * DEG_TO_RAD<T>, a.incl * DEG_TO_RAD<T>);
        for (int r = 0; r < 3; ++r)
            sky[r][c] = e[r];
    }
}

// Add a sample of an orbit at the position x (in the frame of its
// spheroid) to the time of the orbit in its cell, and of its twin (see
// OrbitRows), with AddRowFunT
template<auto AddRowFunT, typename T, typename S> constexpr void
orbits_record_cell(
        const T* x, const OrbitArgs<T>& a, const OrbitRows<S>& rows)
{
    const T R = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const int i = orbits_interval(
            a.cell_r, a.ncell_r, std::sqrt(R * R + x[2] * x[2]));
    const int j = orbits_interval(
            a.cell_b, a.ncell_b, std::atan2(std::abs(x[2]), R));
    if (i < 0 || j < 0)
        return;
    S* cell = rows.cells + i * a.ncell_b + j;
    const T weight = T{1} / a.nsamples;
    AddRowFunT(cell, weight);
    if (a.twins)
        AddRowFunT(cell + rows.twin_cells, weight);
}

// Record copy c (of 2 nphi) of sample n of orbit k, at the position x and
// the velocity v (in the frame of its spheroid; see OrbitArgs), on the
// sky, projected by the matrix sky (see orbits_sky), into rows (see
// orbits_record_sky): turned about the axis by the angle of copy c / 2
// (nphi of them evenly spaced, from a fraction of a turn that changes
// from sample to sample by the golden ratio), and mirrored about the
// plane if c is odd
template<auto AddFunT, auto AddRowFunT, typename T, typename S>
constexpr void
orbits_record_copy(
        int k, int n, int c, const T* x, const T* v,
        const T (&sky)[3][3], const OrbitArgs<T>& a,
        const OrbitRows<S>& rows)
{
    constexpr T GOLDEN = 0.6180339887498949;
    const T offset = n * GOLDEN - std::floor(n * GOLDEN);
    const T phi = 2 * PI<T> * (c / 2 + offset) / a.nphi;
    const T cos = std::cos(phi);
    const T sin = std::sin(phi);
    const T px = cos * x[0] - sin * x[1];
    const T py = sin * x[0] + cos * x[1];
    const T pvx = cos * v[0] - sin * v[1];
    const T pvy = sin * v[0] + cos * v[1];
    const T mirror = c % 2 ? 1 : -1;
    const T pz = mirror * x[2];
    const T pvz = mirror * v[2];
    const T sx = sky[0][0] * px + sky[0][1] * py + sky[0][2] * pz;
    const T sy = sky[1][0] * px + sky[1][1] * py + sky[1][2] * pz;
    const T sv = sky[2][0] * pvx + sky[2][1] * pvy + sky[2][2] * pvz;
    orbits_record_sky<AddFunT, AddRowFunT>(
            k, sx, sy, sv, T{1} / (2 * a.nphi * a.nsamples), a, rows);
}

// Integrate orbit k (see OrbitArgs) and keep its samples: sample n at
// samples + 6 n (x, y, z and vx, vy, vz; NaN for those past max_steps),
// with its steps and energies, if wanted
template<typename T> constexpr void
orbits_trajectory(int k, const OrbitArgs<T>& a, T* samples)
{
    OrbitPoint<T> start, end;
    for (int c = 0; c < 3; ++c)
    {
        start.x[c] = a.starts[6 * k + c];
        start.v[c] = a.starts[6 * k + 3 + c];
    }
    orbits_point_gravity(a.potential, start);
    const auto energy = [](const OrbitPoint<T>& point) {
        return point.phi + (point.v[0] * point.v[0]
                + point.v[1] * point.v[1] + point.v[2] * point.v[2]) / 2;
    };
    const T start_energy = energy(start);

    const T duration = a.turns * orbits_circular_period(
            a.potential, start_energy);
    const T accuracy = a.accuracy;
    T t = 0;
    int n = 0;
    int steps = 0;
    while (n < a.nsamples && steps < a.max_steps)
    {
        // (a trial step for the rate at its end, then the step)
        const T left = duration - t;
        orbits_step(a.potential, start,
                    std::min(accuracy / start.rate, left), end);
        const T dt = std::min(
                accuracy * (1 / start.rate + 1 / end.rate) / 2, left);
        orbits_step(a.potential, start, dt, end);
        for (; n < a.nsamples; ++n)
        {
            const T time = (n + T{0.5}) * duration / a.nsamples;
            if (time >= t + dt)
                break;
            T* sample = samples + 6LL * n;
            orbits_interpolate(
                    start, end, dt, (time - t) / dt, sample, sample + 3);
        }
        t += dt;
        start = end;
        ++steps;
    }
    for (; n < a.nsamples; ++n)
        samples[6LL * n] = std::numeric_limits<T>::quiet_NaN();

    if (a.steps)
        a.steps[k] = steps;
    if (a.energies)
    {
        a.energies[2 * k] = start_energy;
        a.energies[2 * k + 1] = energy(start);
    }
}

// Record orbit k from its samples (see orbits_trajectory), one after the
// other, into its outputs (see OrbitArgs), which only it writes to, from
// 0, and the spectra, which AddFunT adds to (from 0 before)
template<auto AddFunT, typename T> constexpr void
orbits_record_orbit(int k, const OrbitArgs<T>& a, const T* samples)
{
    const int twins = a.twins ? 2 : 1;
    const long long orbit = (long long)k * twins;
    OrbitRows<T> rows;
    rows.last = a.ntargets;
    if (a.cells)
    {
        const long long size = (long long)a.ncell_r * a.ncell_b;
        rows.cells = a.cells + orbit * size;
        rows.twin_cells = size;
        std::fill(rows.cells, rows.cells + twins * size, T{0});
    }
    if (a.moments)
    {
        const long long size = (long long)a.ntargets * (a.order + 2);
        rows.moments = a.moments + orbit * size;
        rows.twin_moments = size;
        std::fill(rows.moments, rows.moments + twins * size, T{0});
    }
    T sky[3][3];
    orbits_sky(a, sky);
    for (int n = 0; n < a.nsamples; ++n)
    {
        const T* sample = samples + 6LL * n;
        if (std::isnan(sample[0]))
            continue;
        if (rows.cells)
            orbits_record_cell<orbits_add<T>>(sample, a, rows);
        for (int c = 0; a.nsky && c < 2 * a.nphi; ++c)
            orbits_record_copy<AddFunT, orbits_add<T>>(
                    k, n, c, sample, sample + 3, sky, a, rows);
    }
}

} // namespace gbkfit
