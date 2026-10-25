#pragma once

#include "gbkfit/gmodel/component.hpp"

namespace gbkfit {

// A spheroid, as the smooth spheroid kernel takes it (with the grid it is
// evaluated on and the outputs: see GridArgs). Its frame is that of a
// thick disk whose spin axis is the axis of the spheroid: the centre at
// its origin, R the distance from the axis and z the height along it.
// Plain data, so that it can be passed by value to a cuda kernel.
template<typename T>
struct SpheroidArgs
{
    // The geometry: one value each (vsys with kinematics only)
    const T* xpos = nullptr;
    const T* ypos = nullptr;
    const T* posa = nullptr;
    const T* incl = nullptr;
    const T* vsys = nullptr;

    // The radius (arcsec) of the sphere around the centre that holds the
    // density (see terms)
    T rmax = 0;

    // The density: the sum of nterms terms, term k of the amplitude a,
    // the scale s and the axis ratio q in terms[3k], terms[3k + 1] and
    // terms[3k + 2] (its height over its width): a f(m / s), m^2 = R^2 +
    // z^2 / q^2, up to m = rmax / max(1, q), the largest spheroid of its
    // shape in the sphere of radius rmax. f(u) = exp(T(ln u) - c u^p),
    // with c and p in function[0] and function[1], and T the table (see
    // table), or 0 without one: e.g. a Gaussian, exp(-u^2 / 2), of c =
    // 1/2 and p = 2.
    int nterms = 0;
    const T* terms = nullptr;
    const T* function = nullptr;

    // The table T (optional): at ln u = table_range[0] + i *
    // table_range[1] for i in [0, ntable), linear in between; below its
    // first point, along the slope table_range[2] (a power law of u), and
    // beyond its last one, -infinity (f is 0)
    int ntable = 0;
    const T* table = nullptr;
    const T* table_range = nullptr;

    // Its kinematics (optional): the tables of its Jeans equations, at the
    // radii R = exp(jeans_range[0] + i jeans_range[1]) and the heights |z|
    // = exp(jeans_range[2] + j jeans_range[3]) (arcsec) for i in [0,
    // jeans_nr) and j in [0, jeans_nz): <v_phi>, sigma_R, sigma_phi and
    // sigma_z (km/s), one plane after the other (jeans[(p jeans_nr + i)
    // jeans_nz + j] of plane p), bilinear in ln R and ln |z|, and at the
    // nearest point of the tables beyond them. Each point moves with a
    // Gaussian distribution of velocities, of the mean vsys + <v_phi>
    // cos(theta) sin(incl) and the dispersion of the velocity ellipsoid
    // on the line of sight (see gmodel_ellipsoid_dispersion).
    int jeans_nr = 0;
    int jeans_nz = 0;
    const T* jeans = nullptr;
    const T* jeans_range = nullptr;

    // Whether it is evaluated as its projection on the sky, on a grid
    // without a depth axis, of the surface density of the sum of its
    // terms, each a f(m / s), m^2 = x^2 + y^2 / q^2 (x along the major axis
    // of the projection, y along its minor axis; q is that of the
    // projection), without a table and to any radius; or else on the
    // voxels of a grid with a depth axis
    bool projected = false;
};

} // namespace gbkfit
