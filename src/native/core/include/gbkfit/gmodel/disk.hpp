#pragma once

namespace gbkfit {

// The traits of one kind (e.g., the velocity polar traits) of a disk:
// n traits, their uids, and their constant and parameter values, the
// values of each trait after those of the previous one.
template<typename T>
struct TraitSet
{
    int n = 0;
    const int* uids = nullptr;
    const T* cvalues = nullptr;
    const int* ccounts = nullptr;
    const T* pvalues = nullptr;
    const int* pcounts = nullptr;
};

// Everything the smooth and Monte Carlo disk kernels need. Plain data,
// so that it can be passed by value to a cuda kernel.
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

    // Density (r), velocity (v) and dispersion (d) polar and height
    // traits, and vertical distortion (z), selection (s) and weight (w)
    // polar traits. A height trait set has as many traits as its polar
    // trait set, or none (a thin disk).
    TraitSet<T> rpt, rht, vpt, vht, dpt, dht, zpt, spt, wpt;

    // The opacity of the 3d spatial grid (optional)
    const T* opacity = nullptr;

    // The 3d spatial grid, in (x, y, z) order, and the spectral axis.
    // The x and y axes of the grid are rotated on the sky by spat_rota
    // degrees (counterclockwise, like a position angle).
    int spat_size[3] = {0, 0, 0};
    T spat_step[3] = {0, 0, 0};
    T spat_zero[3] = {0, 0, 0};
    T spat_rota = 0;
    int spec_size = 0;
    T spec_step = 0;
    T spec_zero = 0;

    // The outputs (all optional)
    T* image = nullptr;
    T* scube = nullptr;
    T* wdata = nullptr;
    T* wdata_cmp = nullptr;
    T* rdata = nullptr;
    T* rdata_cmp = nullptr;
    T* ordata = nullptr;
    T* ordata_cmp = nullptr;
    T* vdata_cmp = nullptr;
    T* ddata_cmp = nullptr;
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
