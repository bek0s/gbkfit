#pragma once

namespace gbkfit {

// The traits of one kind (e.g., the velocity polar traits) of a component:
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

// Where a component is evaluated, and what it writes there: the grid, its
// emission lines, the opacity that dims its light, and the outputs. Plain
// data, so that it can be passed by value to a cuda kernel.
template<typename T>
struct GridArgs
{
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

// The fields of a component that the march of radiative transfer reads,
// which it writes instead of its light (with neither an image nor a
// spectral cube; see GridArgs): planes of the size of the 3d grid, one
// after the other. A thick smooth component writes the light of each
// voxel (its density times the depth of the voxel: j), its velocity v and
// dispersion s. A Monte Carlo disk adds the moments of its clouds in each
// voxel (see gmodel_mcdisk_fields_moments), which then become j, v and s
// of one line, in four planes (the fourth is scratch). A thin disk writes
// j, v, s and the position along the depth axis (in voxels) where its
// plane crosses the line of sight of each spaxel, in four images (planes
// of the size of the x and y axes). With line opacity (see DiskArgs::opt),
// a thick disk writes after s a plane of each line: j times the
// coefficient of the line (before the scratch of a Monte Carlo disk).
template<typename T>
struct FieldsArgs
{
    T* fields = nullptr;
};

} // namespace gbkfit
