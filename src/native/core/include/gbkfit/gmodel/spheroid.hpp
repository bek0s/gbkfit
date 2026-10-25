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
    // The geometry: one value each
    const T* xpos = nullptr;
    const T* ypos = nullptr;
    const T* posa = nullptr;
    const T* incl = nullptr;

    // The radius (arcsec) of the sphere around the centre that holds the
    // density (see terms)
    T rmax = 0;

    // The density: the sum of nterms terms, term k of the amplitude a,
    // the scale s and the axis ratio q in terms[3k], terms[3k + 1] and
    // terms[3k + 2] (its height over its width): a f(m / s), m^2 = R^2 +
    // z^2 / q^2, up to m = rmax / max(1, q), the largest spheroid of its
    // shape in the sphere of radius rmax. f is the table if there is one
    // (see SpheroidArgs::table), and a Gaussian, exp(-u^2 / 2), if not.
    int nterms = 0;
    const T* terms = nullptr;

    // The table of f (optional): ln f at ln u = table_range[0] + i *
    // table_range[1] for i in [0, ntable), linear in between; below its
    // first point, ln f falls along the slope table_range[2] (a power law
    // of u), and beyond its last one, f is 0
    int ntable = 0;
    const T* table = nullptr;
    const T* table_range = nullptr;
};

} // namespace gbkfit
