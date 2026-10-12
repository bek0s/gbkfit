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
    TraitSet<T> rpt, rht, vpt, vht, dpt, dht, zpt, spt;

    // The opacity of the 3d spatial grid (optional): nopacity cubes, one
    // after the other, each with the optical depth from the viewer to the
    // far side of each voxel (see gmodel_opacity_accumulate) of the dust
    // of one extinction law. The light of line l of the spectral cube (or
    // of the image: l = 0) crosses the sum of the optical depths of the
    // cubes k times opacity_ratios[l * nopacity + k]: the ratio of each at
    // the wavelength of the line.
    int nopacity = 0;
    const T* opacity = nullptr;
    const T* opacity_ratios = nullptr;

    // The 3d spatial grid, in (x, y, z) order, and the spectral axis.
    // The x and y axes of the grid are rotated on the sky by spat_rota
    // degrees (counterclockwise, like a position angle). Without a depth
    // axis (the plane of the sky), z has size 1 and step 0.
    int spat_size[3] = {0, 0, 0};
    T spat_step[3] = {0, 0, 0};
    T spat_zero[3] = {0, 0, 0};
    T spat_rota = 0;
    int spec_size = 0;
    T spec_step = 0;
    T spec_zero = 0;

    // The emission lines that the spectral cube gets: line l is at the
    // spectral coordinate lines[3l] + lines[3l + 1] * velocity, with the
    // dispersion lines[3l + 1] * dispersion and the flux lines[3l + 2] *
    // flux (its offset, scale and flux). Required with a spectral cube.
    int nlines = 0;
    const T* lines = nullptr;

    // The outputs (all optional)
    T* image = nullptr;
    T* scube = nullptr;
    T* rdata = nullptr;
    T* rdata_cmp = nullptr;
    T* ordata = nullptr;
    T* ordata_cmp = nullptr;
    // The velocity and dispersion of each voxel are means weighted by the
    // absolute density: vdata_cmp and ddata_cmp get the weighted sums, and
    // vdweight_cmp the sum of the weights
    T* vdata_cmp = nullptr;
    T* ddata_cmp = nullptr;
    T* vdweight_cmp = nullptr;

    // The factor of the density in rdata (e.g. the ratio of the optical
    // depth of dust at the wavelength of an observation to that at V), and
    // that of the light in the image (e.g. the flux ratio of the band of
    // the image to the first band of a component)
    T rdata_scale = 1;
    T image_scale = 1;
};

// The kinds of traits of a disk: the trait sets of DiskArgs
enum class TraitKind
{
    RPT, RHT, VPT, VHT, DPT, DHT, ZPT, SPT
};

// The evaluation of the traits of one kind of a disk at points of its
// plane, each trait apart (see gmodel_disk_traits_point): the (sub)nodes
// of the disk and its traits of that kind; the radius r of each point,
// and its azimuth theta (degrees) for polar traits or its height z above
// the midplane for height traits; and the inclination (degrees) at which
// the velocity polar traits are projected on the line of sight. out gets
// the value of trait i at point j at i * npoints + j.
template<typename T>
struct DiskTraitsArgs
{
    TraitKind kind = TraitKind::RPT;
    int nrnodes = 0;
    const T* rnodes = nullptr;
    TraitSet<T> traits;
    int npoints = 0;
    const T* r = nullptr;
    const T* theta = nullptr;
    const T* z = nullptr;
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
