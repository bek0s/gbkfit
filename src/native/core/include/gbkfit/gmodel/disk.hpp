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
    // axis (the plane of the sky), z has size 1 and step 0. The grid can
    // be a slab of planes of a depth axis: its spat_size[2] planes from
    // the plane slab_zero, of an axis whose first plane is at spat_zero[2]
    // (so that every slab places a point in the same plane; a thin disk is
    // evaluated on the whole axis).
    int spat_size[3] = {0, 0, 0};
    T spat_step[3] = {0, 0, 0};
    T spat_zero[3] = {0, 0, 0};
    T spat_rota = 0;
    int slab_zero = 0;
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

// The fields of a disk that the march of radiative transfer reads, which
// it writes instead of its light (with neither an image nor a spectral
// cube; see DiskArgs), and the coefficients of its line opacity: apart
// from DiskArgs, so that the kernels of light do not carry them.
template<typename T>
struct FieldsArgs
{
    // The fields: planes of the size of the 3d grid, one after the other.
    // A thick smooth disk writes the light of each voxel (its density
    // times the depth of the voxel: j), its velocity v and dispersion s.
    // A Monte Carlo disk adds the moments of its clouds in each voxel (see
    // gmodel_mcdisk_fields_moments), which then become j, v and s of one
    // line, in four planes (the fourth is scratch). A thin disk writes j,
    // v, s and the position along the depth axis (in voxels) where its
    // plane crosses the line of sight of each spaxel, in four images
    // (planes of the size of the x and y axes). With line opacity (opt), a
    // thick disk writes after s a plane of each line: j times the
    // coefficient of the line (before the scratch of a Monte Carlo disk).
    T* fields = nullptr;

    // The polar traits of the coefficients of its line opacity, and the
    // line of each: its row in DiskArgs::lines, or -1 if the line is not
    // evaluated. The coefficient of a line is the sum of its traits.
    TraitSet<T> opt;
    const int* opt_rows = nullptr;
};

// The kinds of traits of a disk: the trait sets of DiskArgs (and the
// coefficients of line opacity of FieldsArgs)
enum class TraitKind
{
    RPT, RHT, VPT, VHT, DPT, DHT, ZPT, SPT, OPT
};

// The evaluation of the traits of one kind of a disk at points of its
// plane, each trait apart (see gmodel_disk_traits_point): the (sub)nodes
// of the disk and its traits of that kind; the radius r of each point,
// and its azimuth theta (degrees) for polar traits or its height z above
// the midplane for height traits; and the inclination (degrees) at which
// the velocity polar traits are projected on the line of sight, each by
// its direction (directions). out gets the value of trait i at point j at
// i * npoints + j.
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
