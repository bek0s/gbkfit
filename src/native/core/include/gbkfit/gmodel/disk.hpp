#pragma once

#include "gbkfit/gmodel/component.hpp"

namespace gbkfit {

// A disk, as the smooth and Monte Carlo disk kernels take it (with the
// grid it is evaluated on and the outputs: see GridArgs). Plain data, so
// that it can be passed by value to a cuda kernel.
template<typename T>
struct DiskArgs
{
    // The radial nodes, and the geometry at each node, or a single value
    // if the disk is not loose (vsys, xpos, ypos) or tilted (posa, incl).
    // There is no vsys without velocity traits.
    bool loose = false;
    bool tilted = false;
    int nrnodes = 0;
    const T* rnodes = nullptr;
    const T* vsys = nullptr;
    const T* xpos = nullptr;
    const T* ypos = nullptr;
    const T* posa = nullptr;
    const T* incl = nullptr;
    // sin(posa), cos(posa), sin(incl) and cos(incl) of each value of posa
    // and incl, for the search of the ring of a point of a loose or
    // tilted disk (see ring_info), made once per evaluation
    const T* posa_incl_sincos = nullptr;

    // Density (r), velocity (v) and dispersion (d) polar and height
    // traits, and vertical distortion (z) and selection (s) polar traits.
    // A height trait set has as many traits as its polar trait set, or
    // none (a thin disk). The velocity polar traits are speeds, each along
    // its direction (VPT_DIRECTION_*, see vp_disk_project).
    TraitSet<T> rpt, rht, vpt, vht, dpt, dht, zpt, spt;
    const int* vpt_directions = nullptr;

    // The polar traits of the coefficients of its line opacity, and the
    // line of each: its row in GridArgs::lines, or -1 if the line is not
    // evaluated. The coefficient of a line is the sum of its traits.
    TraitSet<T> opt;
    const int* opt_rows = nullptr;

    // The dispersion of the vertical equilibrium of the disk at each node,
    // if it has one: it adds to that of the dispersion traits in
    // quadrature
    const T* deq = nullptr;
};

// The kinds of traits of a disk: the trait sets of DiskArgs
enum class TraitKind
{
    RPT, RHT, VPT, VHT, DPT, DHT, ZPT, SPT, OPT
};

// The evaluation of the traits of one kind of a disk at points of its
// plane, each trait apart (see gmodel_disk_traits_point): the (sub)nodes
// of the disk and its traits of that kind; the radius r of each point,
// and its azimuth theta (degrees) for polar traits or its height z above
// the midplane for height traits; and if project, the inclination
// (degrees) at which the velocity polar traits are projected on the line
// of sight, each by its direction (directions; otherwise their speeds).
// out gets the value of trait i at point j at i * npoints + j.
template<typename T>
struct DiskTraitsArgs
{
    TraitKind kind = TraitKind::RPT;
    int nrnodes = 0;
    const T* rnodes = nullptr;
    TraitSet<T> traits;
    const int* directions = nullptr;
    int npoints = 0;
    const T* r = nullptr;
    const T* theta = nullptr;
    const T* z = nullptr;
    bool project = false;
    T incl = 0;
    T* out = nullptr;
};

// The extra arguments of the Monte Carlo disk kernel. The clouds are made
// in pools: one for each density trait with an analytical integral, and
// one for each ring of the others. The arguments are the seed of the
// random numbers, the number of clouds, and for each pool the cumulative
// number of clouds and the (signed) flux of each of its clouds; and
// whether each density trait has an analytical integral.
template<typename T>
struct MCDiskArgs
{
    const T* cloud_flux = nullptr;
    unsigned int seed = 0;
    int nclouds = 0;
    const int* ncloudscsum = nullptr;
    int ncloudscsum_len = 0;
    const bool* has_analytical_integral = nullptr;
};

} // namespace gbkfit
