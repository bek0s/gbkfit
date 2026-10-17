#pragma once

#include "gbkfit/constants.hpp"
#include "gbkfit/gmodel/cone.hpp"
#include "gbkfit/gmodel/disk.hpp"
#include "gbkfit/gmodel/traits.hpp"
#include "gbkfit/random.hpp"
#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

template<auto AddFunT, typename T> void constexpr
gmodel_image_evaluate(T* image, int x, int y, T rvalue, int spat_size_x)
{
    const int idx = index_2d_to_1d(x, y, spat_size_x);
    AddFunT(&image[idx], rvalue);
}

// The channels [cmin, cmax] of [c0, c1) where a line of the given centre
// and dispersion (in the units of the spectral axis) has most of its flux
// (within LINE_WIDTH_MULTIPLIER dispersions); false if there are none. A
// line of a centre or dispersion that is not a number (e.g. of parameters
// out of their range) is in every channel: it makes them NaN, as a flux
// that is not a number does, so that it is seen.
template<typename T> constexpr bool
gmodel_line_range(
        T centre, T sigma, int c0, int c1, T spec_step, T spec_zero,
        int& cmin, int& cmax)
{
    if (std::isnan(centre) || std::isnan(sigma))
    {
        cmin = c0;
        cmax = c1 - 1;
        return c0 < c1;
    }
    // The channels are clamped before they become ints: converting a
    // float out of the range of int is undefined.
    const T lo = std::max<T>(std::rint(
            (centre - sigma * LINE_WIDTH_MULTIPLIER<T> - spec_zero)
            / spec_step), c0);
    const T hi = std::min<T>(std::rint(
            (centre + sigma * LINE_WIDTH_MULTIPLIER<T> - spec_zero)
            / spec_step), c1 - 1);
    if (lo > hi)
        return false;
    cmin = lo;
    cmax = hi;
    return true;
}

// Give add(c, value) the light of a line of the given flux, centre and
// dispersion in each channel c of [cmin, cmax] (see gmodel_line_range):
// the mean of the line over the channel (the cube holds flux per unit
// velocity), which conserves the flux of lines of any width, including
// 0. The channels share their edges, so each channel needs one tail of
// the line (see gauss_1d_mass: precise far in its wings too).
template<typename T, typename AddT> constexpr void
gmodel_line_channels(
        T flux, T centre, T sigma, int cmin, int cmax,
        T spec_step, T spec_zero, AddT add)
{
    if (std::isnan(centre) || std::isnan(sigma))
    {
        for (int c = cmin; c <= cmax; ++c)
            add(c, T(NAN));
        return;
    }
    auto edge_lo = gauss_1d_tail(
            spec_zero + (cmin - T{0.5}) * spec_step, centre, sigma);
    for (int c = cmin; c <= cmax; ++c)
    {
        const auto edge_hi = gauss_1d_tail(
                spec_zero + (c + T{0.5}) * spec_step, centre, sigma);
        add(c, flux * gauss_1d_mass(edge_lo, edge_hi) / spec_step);
        edge_lo = edge_hi;
    }
}

// Add a line of the given flux, centre and dispersion (in the units of
// the spectral axis) to the spectrum (x, y) of a spectral cube (see
// gmodel_line_range and gmodel_line_channels)
template<auto AddFunT, typename T> void constexpr
gmodel_scube_line(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size_z,
        T spec_step,
        T spec_zero)
{
    int zmin, zmax;
    if (!gmodel_line_range(
            vvalue, dvalue, 0, spec_size_z, spec_step, spec_zero,
            zmin, zmax))
        return;
    gmodel_line_channels(
            rvalue, vvalue, dvalue, zmin, zmax, spec_step, spec_zero,
            [&](int z, T value) {
                AddFunT(&scube[index_3d_to_1d(
                        x, y, z, spat_size_x, spat_size_y)], value);
            });
}

// Turn the optical depth of each voxel of the spaxel (x, y) of an opacity
// cube, of the given size, into the optical depth from the viewer to the
// far side of the voxel: a running sum along z, from the voxel nearest
// the viewer (see gmodel_attenuation)
template<typename T> constexpr void
gmodel_opacity_accumulate(
        int x, int y, int size_x, int size_y, int size_z, T* opacity)
{
    T tau = 0;
    for (int z = 0; z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x, y, z, size_x, size_y);
        tau += opacity[idx];
        opacity[idx] = tau;
    }
}

// The optical depth from the viewer to the position pz along the line of
// sight of the spaxel (x, y), in voxels (the voxel z spans pz from z - 0.5
// to z + 0.5), with the absorbers of each voxel spread uniformly along it.
// The opacity cube holds the optical depth from the viewer to the far side
// of each voxel (see gmodel_opacity_accumulate). It is 0 in front of the
// grid, and that of the whole line of sight behind it.
template<typename T> constexpr T
gmodel_optical_depth(int x, int y, T pz, const T* opacity, const int* size)
{
    if (std::isnan(pz))
        return pz;
    // The position from the near side of the grid, and its voxel
    const T p = std::min(std::max(pz + T{0.5}, T{0}), T(size[2]));
    const int z = std::min(int(p), size[2] - 1);
    const T tau_front = z > 0
            ? opacity[index_3d_to_1d(x, y, z - 1, size[0], size[1])]
            : T{0};
    const T tau_own =
            opacity[index_3d_to_1d(x, y, z, size[0], size[1])] - tau_front;
    return tau_front + (p - z) * tau_own;
}

// The opacity that the light of a voxel crosses, in each cube of opacity
// (see GridArgs::opacity): the optical depth of the voxels in front of it
// (front), and that of the absorbers of the voxel itself (own), which are
// mixed uniformly with its emitters. Only the values of the cubes are set
// (not zeroed first: the kernels without opacity would pay for it).
template<typename T>
struct VoxelOpacity
{
    T front[OPACITY_NUM_MAX];
    T own[OPACITY_NUM_MAX];
};

// The opacity of the voxel (x, y, z) (see VoxelOpacity)
template<typename T> constexpr VoxelOpacity<T>
gmodel_voxel_opacity(int x, int y, int z, const GridArgs<T>& g)
{
    const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    VoxelOpacity<T> o;
    for (int k = 0; k < g.nopacity; ++k)
    {
        const T* opacity = g.opacity + k * nspat;
        o.front[k] = z > 0
                ? opacity[index_3d_to_1d(
                        x, y, z - 1, g.spat_size[0], g.spat_size[1])]
                : T{0};
        o.own[k] = opacity[index_3d_to_1d(
                x, y, z, g.spat_size[0], g.spat_size[1])] - o.front[k];
    }
    return o;
}

// The opacity of a thin layer at the position pz along the line of sight
// of the spaxel (x, y) (see gmodel_optical_depth): all of it in front
template<typename T> constexpr VoxelOpacity<T>
gmodel_layer_opacity(int x, int y, T pz, const GridArgs<T>& g)
{
    const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    VoxelOpacity<T> o;
    for (int k = 0; k < g.nopacity; ++k)
    {
        o.front[k] = gmodel_optical_depth(
                x, y, pz, g.opacity + k * nspat, g.spat_size);
        o.own[k] = 0;
    }
    return o;
}

// The optical depth of the opacity o of n cubes in front of the light and
// of its own voxel, at the wavelength of the light: ratios has the ratio
// of each cube there (a row of GridArgs::opacity_ratios)
template<typename T> constexpr void
gmodel_optical_depths(
        const VoxelOpacity<T>& o, int n, const T* ratios,
        T& tau_front, T& tau_own)
{
    tau_front = 0;
    tau_own = 0;
    for (int k = 0; k < n; ++k)
    {
        tau_front += ratios[k] * o.front[k];
        tau_own += ratios[k] * o.own[k];
    }
}

// The fraction of the light of a voxel that reaches the viewer through
// the optical depth in front of the voxel, and on average (1 - exp(-tau))
// / tau of it through the absorbers of the voxel of optical depth tau,
// which are mixed uniformly with its emitters
template<typename T> constexpr T
gmodel_transmission(T tau_front, T tau_own)
{
    const T own = tau_own != 0 ? -std::expm1(-tau_own) / tau_own : T{1};
    return std::exp(-tau_front) * own;
}

// The fraction of the light that reaches the viewer, who is before the
// first voxel along z (the velocities along +z recede from the viewer),
// through the opacity o of n cubes, with ratios their ratios at the
// wavelength of the light (see gmodel_optical_depths and
// gmodel_transmission)
template<typename T> constexpr T
gmodel_attenuation(const VoxelOpacity<T>& o, int n, const T* ratios)
{
    T tau_front, tau_own;
    gmodel_optical_depths(o, n, ratios, tau_front, tau_own);
    return gmodel_transmission(tau_front, tau_own);
}

// Add the emission lines (see GridArgs::lines) of light of the given flux,
// velocity and dispersion to the spectrum (x, y) of a spectral cube. The
// opacity o of nopacity cubes (if not null) dims each line with its row of
// ratios (see GridArgs::opacity_ratios).
template<auto AddFunT, typename T> void constexpr
gmodel_scube_evaluate(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int nlines, const T* lines,
        const VoxelOpacity<T>* o, int nopacity, const T* opacity_ratios,
        int spat_size_x, int spat_size_y,
        int spec_size_z,
        T spec_step,
        T spec_zero)
{
    for (int l = 0; l < nlines; ++l)
    {
        const T offset = lines[3 * l];
        const T scale = lines[3 * l + 1];
        const T flux = lines[3 * l + 2];
        const T lvalue = o
                ? rvalue * gmodel_attenuation(
                        *o, nopacity, opacity_ratios + l * nopacity)
                : rvalue;
        gmodel_scube_line<AddFunT>(
                scube, x, y,
                lvalue * flux, offset + scale * vvalue, scale * dvalue,
                spat_size_x, spat_size_y,
                spec_size_z, spec_step, spec_zero);
    }
}

// Add the light of a point of the given density (rvalue), velocity and
// dispersion to the pixel (x, y) of the image and to the spectrum (x, y) of
// the spectral cube of g (those it has): its light after the opacity in
// front of it (orvalue) to the image, and its lines dimmed by the opacity
// o (if not null; see gmodel_scube_evaluate) to the spectral cube
template<auto AddFunT, typename T> constexpr void
gmodel_light_write(
        int x, int y, T rvalue, T orvalue, T vvalue, T dvalue,
        const VoxelOpacity<T>* o, const GridArgs<T>& g)
{
    if (g.image) {
        gbkfit::gmodel_image_evaluate<AddFunT>(
                g.image, x, y, orvalue * g.image_scale,
                g.spat_size[0]);
    }

    if (g.scube) {
        gbkfit::gmodel_scube_evaluate<AddFunT>(
                g.scube, x, y, rvalue, vvalue, dvalue,
                g.nlines, g.lines,
                o, g.nopacity, g.opacity_ratios,
                g.spat_size[0], g.spat_size[1],
                g.spec_size,
                g.spec_step,
                g.spec_zero);
    }
}

// Write the maps of the voxel idx of a smooth component, which no other
// point of the component writes to: its density (rvalue) to rdata (times
// rdata_scale, added to those of the other components) and rdata_cmp, its
// density after the opacity in front of it (orvalue) to ordata and
// ordata_cmp, and the means of its velocity and dispersion (of one value)
// to vdata_cmp, ddata_cmp and vdweight_cmp (see GridArgs)
template<typename T> constexpr void
gmodel_maps_write(int idx, T rvalue, T orvalue, T vvalue, T dvalue,
                  const GridArgs<T>& g)
{
    if (g.rdata) {
        g.rdata[idx] += rvalue * g.rdata_scale;
    }
    if (g.rdata_cmp) {
        g.rdata_cmp[idx] = rvalue;
    }
    if (g.ordata) {
        g.ordata[idx] += orvalue;
    }
    if (g.ordata_cmp) {
        g.ordata_cmp[idx] = orvalue;
    }
    if (g.vdata_cmp) {
        g.vdata_cmp[idx] = std::abs(rvalue) * vvalue;
    }
    if (g.ddata_cmp) {
        g.ddata_cmp[idx] = std::abs(rvalue) * dvalue;
    }
    if (g.vdweight_cmp) {
        g.vdweight_cmp[idx] = std::abs(rvalue);
    }
}

// Write the fields of the voxel idx of a smooth volume (see
// FieldsArgs::fields): its light j (rvalue), velocity v and dispersion s
template<typename T> constexpr void
gmodel_volume_fields_write(
        int idx, T rvalue, T vvalue, T dvalue,
        const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    f.fields[idx] = rvalue;
    f.fields[nspat + idx] = vvalue;
    f.fields[2 * nspat + idx] = dvalue;
}

// Whether every selection trait of the set spt keeps the point (x, y, r,
// theta) of the plane of a component, in the ring rnidx (a point is kept
// only if every trait selects it)
template<typename T> constexpr bool
gmodel_selected(
        const TraitSet<T>& spt,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta)
{
    T values[TRAIT_NUM_MAX] = {0};
    p_traits<sp_trait<T>>(
            values,
            spt.n, spt.uids,
            spt.cvalues, spt.ccounts,
            spt.pvalues, spt.pcounts,
            rnidx, rnodes, nrnodes,
            x, y, r, theta);
    for (int i = 0; i < spt.n; ++i)
        if (!values[i])
            return false;
    return true;
}

// The sum of the polar traits of the set pt at the point (x, y, r, theta)
// of the plane of a component, in the ring rnidx, each times its height
// trait of the set ht at the height |h| (or alone without height traits:
// a thin component). project(values) transforms the values of the polar
// traits first (e.g. the speeds of the velocity traits onto the line of
// sight; see vp_disk_project_traits).
template<auto PTraitFunT, auto HTraitFunT, typename T, typename ProjectT>
constexpr T
gmodel_traits_sum(
        const TraitSet<T>& pt, const TraitSet<T>& ht,
        int rnidx, const T* rnodes, int nrnodes,
        T x, T y, T r, T theta, T h, ProjectT project)
{
    T pvalues[TRAIT_NUM_MAX] = {0};
    T hvalues[TRAIT_NUM_MAX] = {0};
    p_traits<PTraitFunT>(
            pvalues,
            pt.n, pt.uids,
            pt.cvalues, pt.ccounts,
            pt.pvalues, pt.pcounts,
            rnidx, rnodes, nrnodes,
            x, y, r, theta);
    project(pvalues);
    if (ht.uids)
    {
        h_traits<HTraitFunT>(
                hvalues,
                pt.n, ht.uids,
                ht.cvalues, ht.ccounts,
                ht.pvalues, ht.pcounts,
                rnidx, rnodes, nrnodes,
                r, std::abs(h));
    }
    T sum = 0;
    for (int i = 0; i < pt.n; ++i)
        sum += pvalues[i] * (ht.uids ? hvalues[i] : 1);
    return sum;
}

template<typename T> constexpr void
transform_cpos(T& x, T& y, T xpos, T ypos)
{
    x -= xpos;
    y -= ypos;
}

// Rotates by posa + 90 degrees (see transform_lh_rotate_z()), so that
// the position angle is measured from north (the y axis)
template<typename T> constexpr void
transform_posa(T& x, T& y, T posa)
{
    transform_lh_rotate_z(x, y, x, y, posa);
}

// Inverse of transform_posa(). Note that it is not transform_posa()
// with -posa, because transform_posa() is not a rotation by posa.
template<typename T> constexpr void
transform_posa_inverse(T& x, T& y, T posa)
{
    transform_lh_rotate_z(x, y, x, y, -posa - PI<T>);
}

template<typename T> constexpr void
transform_incl(T& y, T incl)
{
    y /= std::cos(incl);
}

template<typename T> constexpr void
transform_incl(T& y, T& z, T incl)
{
    transform_lh_rotate_x(y, z, y, z, incl);
}

template<typename T> constexpr void
transform_cpos_posa_incl(T& x, T& y, T xposi, T yposi, T posai, T incli)
{
    transform_cpos(x, y, xposi, yposi);
    transform_posa(x, y, posai);
    transform_incl(y, incli);
}

template<typename T> constexpr void
transform_cpos_posa_incl(T& x, T& y, T& z, T xposi, T yposi, T posai, T incli)
{
    transform_cpos(x, y, xposi, yposi);
    transform_posa(x, y, posai);
    transform_incl(y, z, incli);
}

// Inverse of transform_cpos_posa_incl(): from disk to sky coordinates
template<typename T> constexpr void
transform_cpos_posa_incl_inverse(
        T& x, T& y, T& z, T xposi, T yposi, T posai, T incli)
{
    transform_incl(y, z, -incli);
    transform_posa_inverse(x, y, posai);
    transform_cpos(x, y, -xposi, -yposi);
}

// The radius of a point (x, y) of the sky in the plane of the node of the
// given index of a disk (transform_cpos_posa_incl), with the sines and
// cosines of its posa and incl (sin posa, cos posa, sin incl, cos incl of
// each node, see DiskArgs::posa_incl_sincos)
template<typename T> constexpr T
rnode_radius(
        T x, T y, bool loose, bool tilted, int index,
        const T* xpos, const T* ypos, const T* sincos)
{
    const T* sc = sincos + 4 * (tilted ? index : 0);
    x -= loose ? xpos[index] : xpos[0];
    y -= loose ? ypos[index] : ypos[0];
    // (transform_posa: transform_lh_rotate_z by posa; transform_incl)
    const T xr = - x * sc[0] + y * sc[1];
    const T yr = (- x * sc[1] - y * sc[0]) / sc[3];
    return std::sqrt(xr * xr + yr * yr);
}

// The radius of a point (x, y, z) in the frame of the node of the given
// index of a disk: its cylindrical radius (see the 2d rnode_radius)
template<typename T> constexpr T
rnode_radius(
        T x, T y, T z, bool loose, bool tilted, int index,
        const T* xpos, const T* ypos, const T* sincos)
{
    const T* sc = sincos + 4 * (tilted ? index : 0);
    x -= loose ? xpos[index] : xpos[0];
    y -= loose ? ypos[index] : ypos[0];
    // (transform_posa, then transform_incl: transform_lh_rotate_x by incl)
    const T xr = - x * sc[0] + y * sc[1];
    const T yr = (- x * sc[1] - y * sc[0]) * sc[3] + z * sc[2];
    return std::sqrt(xr * xr + yr * yr);
}

template<typename T> constexpr bool
disk_info(int& index, T radius, int nnodes, const T* nodes)
{
    // Ignore anything smaller than the first node
    if (radius < nodes[0])
        return false;

    // Ignore anything larger than the last node
    if (radius >= nodes[nnodes-1])
        return false;

    // Linear search for the first node beyond the radius: the outer node
    // of its ring (the radius is not below nodes[0], checked above)
    for(index = 1; index < nnodes; ++index)
        if (radius < nodes[index])
            break;

    return true;
}

template<typename T> constexpr bool
ring_info(
        int& index, T& radius,
        T x, T y,
        bool loose, bool tilted, int nnodes, const T* nodes,
        const T* xpos, const T* ypos, const T* sincos)
{
    T radius_min = rnode_radius(
            x, y, loose, tilted, 0, xpos, ypos, sincos);

    // Ignore anything smaller than the first node
    if (radius_min < nodes[0])
        return false;

    T radius_max = rnode_radius(
            x, y, loose, tilted, nnodes-1, xpos, ypos, sincos);

    // Ignore anything larger than the last node
    if (radius_max >= nodes[nnodes-1])
        return false;
#if 1
    // Linear search
    for(index = 1; index < nnodes; ++index)
    {
        T radius_cur = rnode_radius(
                x, y, loose, tilted, index, xpos, ypos, sincos);
        if (radius_cur < nodes[index])
            break;
    }
#else
    // Binary search
    int ilo = 1;
    int ihi = nnodes - 1;
    while(ihi > ilo + 1)
    {
        index = (ihi + ilo)/2;
        T radius_cur = rnode_radius(
                x, y, loose, tilted, index, xpos, ypos, sincos);
        if(nodes[index] > radius_cur)
            ihi = index;
        else
            ilo = index;
    }
#endif
    // The radius is calculated using the velfi strategy
    T d1 = nodes[index-1] - rnode_radius(
            x, y, loose, tilted, index-1, xpos, ypos, sincos);
    T d2 = nodes[index] - rnode_radius(
            x, y, loose, tilted, index, xpos, ypos, sincos);
    radius = (nodes[index] * d1 - nodes[index - 1] * d2) / (d1 - d2);

    return true;
}

template<typename T> constexpr bool
ring_info(
        int& index, T& radius,
        T x, T y, T z,
        bool loose, bool tilted, int nnodes, const T* nodes,
        const T* xpos, const T* ypos, const T* sincos)
{
    T radius_min = rnode_radius(
            x, y, z, loose, tilted, 0, xpos, ypos, sincos);

    // Ignore anything smaller than the first node
    if (radius_min < nodes[0])
        return false;

    T radius_max = rnode_radius(
            x, y, z, loose, tilted, nnodes-1, xpos, ypos, sincos);

    // Ignore anything larger than the last node
    if (radius_max >= nodes[nnodes-1])
        return false;
#if 1
    // Linear search
    for(index = 1; index < nnodes; ++index)
    {
        T radius_cur = rnode_radius(
                x, y, z, loose, tilted, index, xpos, ypos, sincos);
        if (radius_cur < nodes[index])
            break;
    }
#else
    // Binary search
    int ilo = 1;
    int ihi = nnodes - 1;
    while(ihi > ilo + 1)
    {
        index = (ihi + ilo)/2;
        T radius_cur = rnode_radius(
                x, y, z, loose, tilted, index, xpos, ypos, sincos);
        if(nodes[index] > radius_cur)
            ihi = index;
        else
            ilo = index;
    }
#endif
    // The radius is calculated using the velfi strategy
    T d1 = nodes[index-1] - rnode_radius(
            x, y, z, loose, tilted, index-1, xpos, ypos, sincos);
    T d2 = nodes[index] - rnode_radius(
            x, y, z, loose, tilted, index, xpos, ypos, sincos);
    radius = (nodes[index] * d1 - nodes[index - 1] * d2) / (d1 - d2);

    return true;
}

// The number of planes of line opacity in the fields of a thick disk (see
// FieldsArgs::fields): one for each line with line opacity, none without
template<typename T> constexpr int
gmodel_fields_lines(const DiskArgs<T>& a, const GridArgs<T>& g)
{
    return a.opt.uids ? g.nlines : 0;
}

// The coefficient of the line opacity of each line (see DiskArgs::opt)
// at a point of the plane of the disk, in the ring rnidx, into
// coefficients (one for each line; there are at most TRAIT_NUM_MAX, one
// for each trait)
template<typename T> constexpr void
gmodel_line_opacity(
        T* coefficients, int rnidx, T x, T y, T r, T theta,
        const DiskArgs<T>& a, int nlines)
{
    T values[TRAIT_NUM_MAX] = {0};
    p_traits<rp_trait<T>>(
            values,
            a.opt.n, a.opt.uids,
            a.opt.cvalues, a.opt.ccounts,
            a.opt.pvalues, a.opt.pcounts,
            rnidx, a.rnodes, a.nrnodes,
            x, y, r, theta);
    for (int l = 0; l < nlines; ++l)
        coefficients[l] = 0;
    for (int i = 0; i < a.opt.n; ++i)
        if (a.opt_rows[i] >= 0)
            coefficients[a.opt_rows[i]] += values[i];
}

// The velocity that the moments of the clouds of a Monte Carlo disk are
// relative to (see gmodel_mcdisk_fields_moments): its systemic velocity
// at its first node (0 without one), so that the squares of the
// velocities are small (their differences lose less precision)
template<typename T> constexpr T
gmodel_fields_vref(const DiskArgs<T>& a)
{
    return a.vsys ? a.vsys[0] : T{0};
}

// Evaluate the cloud ci of a Monte Carlo disk. HasOpacity is whether
// opacity dims the disk (g.opacity is not null), HasFields whether it
// writes the fields of the march instead of its light (f.fields is not
// null), and HasLineOpacity whether they have its line opacity (a.opt has
// traits): template parameters, so that the other kernels have none of
// their code (nor its registers and local memory, on a GPU).
template<
        auto AtomicAddFunT, bool HasOpacity, bool HasFields,
        bool HasLineOpacity, typename T>
constexpr void
gmodel_mcdisk_evaluate_cloud(
        RNG<T>& rng, int ci, const DiskArgs<T>& a, const GridArgs<T>& g,
        const MCDiskArgs<T>& mc, const FieldsArgs<T>& f)
{
    int rnidx=0, tidx=0;
    const T* rpt_cptr = a.rpt.cvalues;
    const T* rpt_pptr = a.rpt.pvalues;
    const T* rht_cptr = a.rht.cvalues;
    const T* rht_pptr = a.rht.pvalues;
    T xd=0, yd=0, zd=0, rd=0, theta=0, sign=1;
    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T ptvalues[TRAIT_NUM_MAX] = {0};
    T rvalue=0, vvalue=0, dvalue=0, zvalue=0;

    // Find the pool of the cloud; its clouds share its flux
    while(ci >= mc.ncloudscsum[rnidx]) {
        rnidx++;
    }
    rvalue = mc.cloud_flux[rnidx];

    // Find which trait and subring the cloud belongs to.
    for(tidx = 0; tidx < a.rpt.n; ++tidx)
    {
        // One pool of clouds for the whole disk, or one for each ring (the
        // nodes are the ring centres between the edges of the disk)
        int size = mc.has_analytical_integral[tidx] ? 1 : a.nrnodes - 2;
        if (rnidx < size)
            break;
        rnidx -= size;
        rpt_cptr += a.rpt.ccounts[tidx];
        rpt_pptr += a.rpt.pcounts[tidx];
        rht_cptr += a.rht.ccounts[tidx];
        rht_pptr += a.rht.pcounts[tidx];
    }

    // Density polar trait
    // The first and last radial nodes must be ignored.
    rp_trait_rnd<T>(
            sign, rd, theta, rng,
            a.rpt.uids[tidx], rpt_cptr, rpt_pptr,
            rnidx, a.rnodes, a.nrnodes);

    // The sign of the cloud within its pool (harmonics)
    rvalue *= sign;

    // Convert to surface brightness
    rvalue /= g.spat_step[0] * g.spat_step[1];

    // Calculate cartesian coordinates on disk plane.
    xd = rd * std::cos(theta);
    yd = rd * std::sin(theta);

    // Recalculate radial node index.
    // This is done in order to account for:
    //  - rptraits with an analytical integral (no subrings).
    //  - pixels in the first and last half subrings.
    if (!disk_info(rnidx, rd, a.nrnodes, a.rnodes)) {
        return;
    }

    // Selection traits
    if (!gmodel_selected(a.spt, rnidx, a.rnodes, a.nrnodes, xd, yd, rd, theta))
        return;

    // Density height trait: the height of the cloud above the midplane (a
    // cloud of a profile that is none, at a NaN height, is discarded below)
    rh_trait_rnd<T>(
                zd, rng, a.rht.uids[tidx], rht_cptr, rht_pptr,
                rnidx, a.rnodes, a.nrnodes, rd);
    const T height = zd;

    // Vertical distortion traits
    if (a.zpt.uids)
    {
        p_traits<zp_trait<T>>(
                ptvalues,
                a.zpt.n, a.zpt.uids,
                a.zpt.cvalues, a.zpt.ccounts,
                a.zpt.pvalues, a.zpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xd, yd, rd, theta);

        for (int i = 0; i < a.zpt.n; ++i)
            zvalue += ptvalues[i];

        // The distortion displaces the midplane
        zd += zvalue;
    }

    vsysi = !a.vsys ? 0
            : a.loose ? lerp(rd, rnidx, a.rnodes, a.vsys) : a.vsys[0];
    xposi = a.loose ? lerp(rd, rnidx, a.rnodes, a.xpos) : a.xpos[0];
    yposi = a.loose ? lerp(rd, rnidx, a.rnodes, a.ypos) : a.ypos[0];
    posai = a.tilted ? lerp(rd, rnidx, a.rnodes, a.posa) : a.posa[0];
    incli = a.tilted ? lerp(rd, rnidx, a.rnodes, a.incl) : a.incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;

    T xn=xd, yn=yd, zn=zd;
    transform_cpos_posa_incl_inverse(xn, yn, zn, xposi, yposi, posai, incli);

    // world-to-image transform
    transform_rh_rotate_z(xn, yn, xn, yn, -g.spat_rota * DEG_TO_RAD<T>);

    // The voxel of the cloud. Discard the clouds outside the cube, while
    // their positions are floats: converting one out of the range of int
    // (e.g. of a heavy-tailed height), or NaN, is undefined.
    const T px = std::rint((xn - g.spat_zero[0]) / g.spat_step[0]);
    const T py = std::rint((yn - g.spat_zero[1]) / g.spat_step[1]);
    const T pz = std::rint((zn - g.spat_zero[2]) / g.spat_step[2])
            - g.slab_zero;
    if (!(px >= 0 && px < g.spat_size[0] &&
          py >= 0 && py < g.spat_size[1] &&
          pz >= 0 && pz < g.spat_size[2])) {
        return;
    }
    const int x = px;
    const int y = py;
    const int z = pz;

    // Velocity traits: their speeds, projected on the line of sight
    vvalue = gmodel_traits_sum<vp_trait<T>, vh_trait<T>>(
            a.vpt, a.vht, rnidx, a.rnodes, a.nrnodes,
            xd, yd, rd, theta, height,
            [&](T* values) {
                vp_disk_project_traits(
                        values, a.vpt.n, a.vpt_directions, rd, theta, incli);
            });

    // Apply systemic velocity
    vvalue += vsysi;

    // Dispersion traits
    dvalue = gmodel_traits_sum<dp_trait<T>, dh_trait<T>>(
            a.dpt, a.dht, rnidx, a.rnodes, a.nrnodes,
            xd, yd, rd, theta, height, [](T*) {});

    // Ensure positive dispersion
    // Ideally, we want the caller to never use parameter values that result in
    // a negative dispersion. Unfortunately, this is hard to achieve.
    // Forcing a positive dispersion is a reasonable solution.
    dvalue = std::abs(dvalue);

    // The opacity in front of the light, and the light that it lets
    // through (that of the image, or of the first line)
    VoxelOpacity<T> opacity;
    if (HasOpacity) {
        opacity = gmodel_voxel_opacity(x, y, z, g);
    }
    const T orvalue = HasOpacity
            ? rvalue * gmodel_attenuation(
                    opacity, g.nopacity, g.opacity_ratios)
            : rvalue;
    gmodel_light_write<AtomicAddFunT>(
            x, y, rvalue, orvalue, vvalue, dvalue,
            HasOpacity ? &opacity : nullptr, g);

    const int idx = index_3d_to_1d(x, y, z, g.spat_size[0], g.spat_size[1]);

    if (g.rdata) {
        AtomicAddFunT(&g.rdata[idx], rvalue * g.rdata_scale);
    }
    if (g.rdata_cmp) {
        AtomicAddFunT(&g.rdata_cmp[idx], rvalue);
    }
    if (g.ordata) {
        AtomicAddFunT(&g.ordata[idx], orvalue);
    }
    if (g.ordata_cmp) {
        AtomicAddFunT(&g.ordata_cmp[idx], orvalue);
    }
    // The clouds of a voxel add to the means of its velocity and dispersion
    if (g.vdata_cmp) {
        AtomicAddFunT(&g.vdata_cmp[idx], std::abs(rvalue) * vvalue);
    }
    if (g.ddata_cmp) {
        AtomicAddFunT(&g.ddata_cmp[idx], std::abs(rvalue) * dvalue);
    }
    if (g.vdweight_cmp) {
        AtomicAddFunT(&g.vdweight_cmp[idx], std::abs(rvalue));
    }
    // The moments of the clouds of the voxel (see
    // gmodel_mcdisk_fields_moments), and their light times the coefficient
    // of each line
    if (HasFields) {
        const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
                * g.spat_size[2];
        const int nk = HasLineOpacity ? g.nlines : 0;
        const T weight = std::abs(rvalue);
        const T dv = vvalue - gmodel_fields_vref(a);
        AtomicAddFunT(&f.fields[idx], rvalue);
        AtomicAddFunT(&f.fields[nspat + idx], weight * dv);
        AtomicAddFunT(
                &f.fields[2 * nspat + idx],
                weight * (dv * dv + dvalue * dvalue));
        if (HasLineOpacity) {
            T coefficients[TRAIT_NUM_MAX];
            gmodel_line_opacity(
                    coefficients, rnidx, xd, yd, rd, theta, a, g.nlines);
            for (int l = 0; l < nk; ++l) {
                AtomicAddFunT(
                        &f.fields[(3 + l) * nspat + idx],
                        rvalue * coefficients[l]);
            }
        }
        AtomicAddFunT(&f.fields[(3 + nk) * nspat + idx], weight);
    }
}

// Turn the moments of the clouds of a Monte Carlo disk in the voxel idx
// (see FieldsArgs::fields): the sum of their light (j), and of their
// absolute light times dv, times dv^2 + s^2 and alone (the last plane),
// where dv is their velocity relative to gmodel_fields_vref, into the
// velocity and the dispersion of one line of the same mean and variance
// as the sum of their lines (weighted by their absolute light). A voxel
// without clouds gets 0.
template<typename T> constexpr void
gmodel_mcdisk_fields_moments(
        long long idx, const DiskArgs<T>& a, const GridArgs<T>& g,
        const FieldsArgs<T>& f)
{
    const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    T* v = &f.fields[nspat + idx];
    T* s = &f.fields[2 * nspat + idx];
    const T weight = f.fields[(3 + gmodel_fields_lines(a, g)) * nspat + idx];
    if (weight == 0) {
        *v = *s = 0;
        return;
    }
    const T dv = *v / weight;
    *s = std::sqrt(std::max(*s / weight - dv * dv, T{0}));
    *v = gmodel_fields_vref(a) + dv;
}

// The number of evaluations of a smooth disk along each spaxel: one for
// each voxel of a thick disk, and one for a thin disk (see
// gmodel_smdisk_evaluate_spaxel)
template<typename T> constexpr int
gmodel_smdisk_size_z(const DiskArgs<T>& a, const GridArgs<T>& g)
{
    return a.rht.uids ? g.spat_size[2] : 1;
}

// Evaluate the voxel (x, y, z) of a thick smooth disk, or the spaxel (x, y)
// of a thin one (z = 0). On a grid with a depth axis, the light of a thin
// disk is in the voxel where its plane crosses the line of sight of the
// spaxel (none if that is outside the grid, though its light still reaches
// the image or spectral cube), dimmed by the opacity in front of the
// crossing. AddFunT adds the light of the voxel to the pixel and the
// spectrum of its spaxel, which the other voxels of the spaxel add to too:
// it must be atomic if they are evaluated in parallel. Only this voxel
// writes to its other outputs. HasOpacity is whether opacity dims the disk,
// HasFields whether it writes the fields of the march, and HasLineOpacity
// whether they have its line opacity (see gmodel_mcdisk_evaluate_cloud).
template<
        auto AddFunT, bool HasOpacity, bool HasFields, bool HasLineOpacity,
        typename T>
constexpr void
gmodel_smdisk_evaluate_spaxel(
        int x, int y, int z, const DiskArgs<T>& a, const GridArgs<T>& g,
        const FieldsArgs<T>& f)
{
    bool is_thin = a.rht.uids == nullptr;
    bool has_depth = g.spat_step[2] != 0;

    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T xn=x, yn=y, zn=z, rn=0, theta=0;
    int rnidx = -1;

    // image-to-world transform
    xn = g.spat_zero[0] + x * g.spat_step[0];
    yn = g.spat_zero[1] + y * g.spat_step[1];
    zn = g.spat_zero[2] + (z + g.slab_zero) * g.spat_step[2];
    transform_rh_rotate_z(xn, yn, xn, yn, g.spat_rota * DEG_TO_RAD<T>);

    // If the disk is loose or tilted, we need to calculate the pixel's
    // radial node index and radius now.
    if (a.loose || a.tilted)
    {
        bool is_on_disk = is_thin
                ? ring_info(
                    rnidx, rn, xn, yn, a.loose, a.tilted,
                    a.nrnodes, a.rnodes, a.xpos, a.ypos, a.posa_incl_sincos)
                : ring_info(
                    rnidx, rn, xn, yn, zn, a.loose, a.tilted,
                    a.nrnodes, a.rnodes, a.xpos, a.ypos, a.posa_incl_sincos);
        if (!is_on_disk)
            return;
    }

    // Calculate systemic velocity and geometrical parameters
    vsysi = !a.vsys ? 0
            : a.loose ? lerp(rn, rnidx, a.rnodes, a.vsys) : a.vsys[0];
    xposi = a.loose ? lerp(rn, rnidx, a.rnodes, a.xpos) : a.xpos[0];
    yposi = a.loose ? lerp(rn, rnidx, a.rnodes, a.ypos) : a.ypos[0];
    posai = a.tilted ? lerp(rn, rnidx, a.rnodes, a.posa) : a.posa[0];
    incli = a.tilted ? lerp(rn, rnidx, a.rnodes, a.incl) : a.incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;

    // world-to-disk transform
    if (is_thin) {
        transform_cpos_posa_incl(xn, yn, xposi, yposi, posai, incli);
    } else {
        transform_cpos_posa_incl(xn, yn, zn, xposi, yposi, posai, incli);
    }

    // The position along the depth axis (in voxels) of the light: that of
    // the voxel, or for a thin disk that of the crossing of its plane, at
    // the depth yn * sin(incl)
    T pz = z;
    if (is_thin && has_depth) {
        pz = (yn * std::sin(incli) - g.spat_zero[2]) / g.spat_step[2];
    }

    // If the disk is not loose or tilted, we need to calculate the pixel's
    // radial node index and radius now.
    if (!(a.loose || a.tilted)) {
        rn = std::sqrt(xn * xn + yn * yn);
        bool is_on_disk = disk_info(rnidx, rn, a.nrnodes, a.rnodes);
        if (!is_on_disk) {
            return;
        }
    }
    theta = std::atan2(yn, xn);

    // These are needed for trait evaluation
    T ptvalues[TRAIT_NUM_MAX] = {0};
    T rvalue=0, vvalue=0, dvalue=0, zvalue=0;

    // Selection traits
    if (!gmodel_selected(a.spt, rnidx, a.rnodes, a.nrnodes, xn, yn, rn, theta))
        return;

    // Vertical distortion traits
    if (a.zpt.uids)
    {
        p_traits<zp_trait<T>>(
                ptvalues,
                a.zpt.n, a.zpt.uids,
                a.zpt.cvalues, a.zpt.ccounts,
                a.zpt.pvalues, a.zpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta);

        for (int i = 0; i < a.zpt.n; ++i)
            zvalue += ptvalues[i];

        // The distortion displaces the midplane: the height of the voxel
        // above it (as the Monte Carlo disk places its clouds)
        zn -= zvalue;
    }

    // Density traits
    rvalue = gmodel_traits_sum<rp_trait<T>, rh_trait<T>>(
            a.rpt, a.rht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, zn, [](T*) {});

    // Discard pixels with zero density
    if (!rvalue) {
        return;
    }

    // Thin disk requires density correction for inclination (seen from
    // either side: above 90 degrees, and at 90 in float, the cosine is
    // negative)
    if (is_thin) {
        rvalue /= std::abs(std::cos(incli));
    }

    // Thick disk requires integration along the spatial z axis
    if (!is_thin) {
        rvalue *= g.spat_step[2];
    }

    // Velocity traits: their speeds, projected on the line of sight
    vvalue = gmodel_traits_sum<vp_trait<T>, vh_trait<T>>(
            a.vpt, a.vht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, zn,
            [&](T* values) {
                vp_disk_project_traits(
                        values, a.vpt.n, a.vpt_directions, rn, theta, incli);
            });

    // Apply systemic velocity
    vvalue += vsysi;

    // Dispersion traits
    dvalue = gmodel_traits_sum<dp_trait<T>, dh_trait<T>>(
            a.dpt, a.dht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, zn, [](T*) {});

    // Ensure positive dispersion
    // Ideally, we want the caller to never use parameter values that result in
    // a negative dispersion. Unfortunately, this is hard to achieve.
    // Forcing a positive dispersion is a reasonable solution.
    dvalue = std::abs(dvalue);

    // The opacity in front of the light (that of a thin disk is a layer,
    // which only the opacity in front of it dims), and the light that it
    // lets through (that of the image, or of the first line)
    VoxelOpacity<T> opacity;
    if (HasOpacity) {
        opacity = is_thin
                ? gmodel_layer_opacity(x, y, pz, g)
                : gmodel_voxel_opacity(x, y, z, g);
    }
    const T orvalue = HasOpacity
            ? rvalue * gmodel_attenuation(
                    opacity, g.nopacity, g.opacity_ratios)
            : rvalue;
    gmodel_light_write<AddFunT>(
            x, y, rvalue, orvalue, vvalue, dvalue,
            HasOpacity ? &opacity : nullptr, g);

    // The fields of the layer of a thin disk, wherever its plane crosses
    // the line of sight (see FieldsArgs::fields)
    if (HasFields && is_thin) {
        const long long nimage = 1LL * g.spat_size[0] * g.spat_size[1];
        const int idx = index_2d_to_1d(x, y, g.spat_size[0]);
        f.fields[idx] = rvalue;
        f.fields[nimage + idx] = vvalue;
        f.fields[2 * nimage + idx] = dvalue;
        f.fields[3 * nimage + idx] = pz;
    }

    // The voxel of the 3d outputs (a thin disk outside the grid has none).
    // The position is checked while it is a float: converting one out of
    // the range of int, or NaN, is undefined.
    const T vz = std::rint(pz);
    if (!(vz >= 0 && vz < g.spat_size[2])) {
        return;
    }
    const int idx = index_3d_to_1d(
            x, y, int(vz), g.spat_size[0], g.spat_size[1]);
    gmodel_maps_write(idx, rvalue, orvalue, vvalue, dvalue, g);

    // The fields of the voxel of a thick disk, and its light times the
    // coefficient of each line
    if (HasFields && !is_thin) {
        gmodel_volume_fields_write(idx, rvalue, vvalue, dvalue, g, f);
        if (HasLineOpacity) {
            const long long nspat = 1LL * g.spat_size[0] * g.spat_size[1]
                    * g.spat_size[2];
            T coefficients[TRAIT_NUM_MAX];
            gmodel_line_opacity(
                    coefficients, rnidx, xn, yn, rn, theta, a, g.nlines);
            for (int l = 0; l < g.nlines; ++l)
                f.fields[(3 + l) * nspat + idx] = rvalue * coefficients[l];
        }
    }
}

// Evaluate the voxel (x, y, z) of a smooth cone. A point of its frame (see
// ConeArgs) is in the shell of its distance r from the apex, at the
// azimuth theta around the axis and on the side of the sign of z, and its
// height traits are at its distance from the mid-surface of the walls, d
// = R cos(wall) - |z| sin(wall) (R the distance from the axis; on a disk,
// wall = 90 degrees, |d| = |z|). AddFunT adds the light of the voxel to
// the pixel and the spectrum of its spaxel, which the other voxels of the
// spaxel add to too: it must be atomic if they are evaluated in parallel.
// Only this voxel writes to its other outputs. HasOpacity is whether
// opacity dims the cone, and HasFields whether it writes the fields of the
// march.
template<auto AddFunT, bool HasOpacity, bool HasFields, typename T>
constexpr void
gmodel_smcone_evaluate_voxel(
        int x, int y, int z, const ConeArgs<T>& a, const GridArgs<T>& g,
        const FieldsArgs<T>& f)
{
    // image-to-world transform
    T xn = g.spat_zero[0] + x * g.spat_step[0];
    T yn = g.spat_zero[1] + y * g.spat_step[1];
    T zn = g.spat_zero[2] + (z + g.slab_zero) * g.spat_step[2];
    transform_rh_rotate_z(xn, yn, xn, yn, g.spat_rota * DEG_TO_RAD<T>);

    // world-to-cone transform
    const T incli = a.incl[0] * DEG_TO_RAD<T>;
    transform_cpos_posa_incl(
            xn, yn, zn, a.xpos[0], a.ypos[0], a.posa[0] * DEG_TO_RAD<T>,
            incli);

    // The side, the shell, the azimuth and the distance from the
    // mid-surface of the walls
    if (!((zn >= 0 && a.positive) || (zn <= 0 && a.negative)))
        return;
    const T rn = std::sqrt(xn * xn + yn * yn + zn * zn);
    int rnidx = -1;
    if (!disk_info(rnidx, rn, a.nrnodes, a.rnodes))
        return;
    const T theta = std::atan2(yn, xn);
    const T wall = a.wall[0] * DEG_TO_RAD<T>;
    const T dn = std::sqrt(xn * xn + yn * yn) * std::cos(wall)
            - std::abs(zn) * std::sin(wall);

    // Selection traits
    if (!gmodel_selected(a.spt, rnidx, a.rnodes, a.nrnodes, xn, yn, rn, theta))
        return;

    // Density traits, integrated along the spatial z axis
    T rvalue = gmodel_traits_sum<rp_trait<T>, rh_trait<T>>(
            a.rpt, a.rht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, dn, [](T*) {});
    if (!rvalue)
        return;
    rvalue *= g.spat_step[2];

    // Velocity traits: their speeds, projected on the line of sight, and
    // the systemic velocity
    T vvalue = gmodel_traits_sum<vp_trait<T>, vh_trait<T>>(
            a.vpt, a.vht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, dn,
            [&](T* values) {
                vp_cone_project_traits(
                        values, a.vpt.n, a.vpt_directions,
                        xn, yn, zn, rn, incli);
            });
    vvalue += a.vsys ? a.vsys[0] : 0;

    // Dispersion traits (positive, as that of a disk)
    const T dvalue = std::abs(gmodel_traits_sum<dp_trait<T>, dh_trait<T>>(
            a.dpt, a.dht, rnidx, a.rnodes, a.nrnodes,
            xn, yn, rn, theta, dn, [](T*) {}));

    // The opacity in front of the light, and the light that it lets
    // through (that of the image, or of the first line)
    VoxelOpacity<T> opacity;
    if (HasOpacity)
        opacity = gmodel_voxel_opacity(x, y, z, g);
    const T orvalue = HasOpacity
            ? rvalue * gmodel_attenuation(
                    opacity, g.nopacity, g.opacity_ratios)
            : rvalue;
    gmodel_light_write<AddFunT>(
            x, y, rvalue, orvalue, vvalue, dvalue,
            HasOpacity ? &opacity : nullptr, g);

    const int idx = index_3d_to_1d(x, y, z, g.spat_size[0], g.spat_size[1]);
    gmodel_maps_write(idx, rvalue, orvalue, vvalue, dvalue, g);
    if (HasFields)
        gmodel_volume_fields_write(idx, rvalue, vvalue, dvalue, g, f);
}

// The polar traits of the set of a DiskTraitsArgs at its point j, in the
// ring rnidx, into values
template<auto PTraitFunT, typename T> constexpr void
gmodel_disk_ptraits_point(
        T* values, int j, int rnidx, const DiskTraitsArgs<T>& a)
{
    const TraitSet<T>& t = a.traits;
    const T r = a.r[j];
    const T theta = a.theta[j] * DEG_TO_RAD<T>;
    p_traits<PTraitFunT>(
            values, t.n, t.uids, t.cvalues, t.ccounts, t.pvalues, t.pcounts,
            rnidx, a.rnodes, a.nrnodes,
            r * std::cos(theta), r * std::sin(theta), r, theta);
}

// The height traits of the set of a DiskTraitsArgs at its point j, in the
// ring rnidx, into values: at |z|, as the disk kernels evaluate them
template<auto HTraitFunT, typename T> constexpr void
gmodel_disk_htraits_point(
        T* values, int j, int rnidx, const DiskTraitsArgs<T>& a)
{
    const TraitSet<T>& t = a.traits;
    h_traits<HTraitFunT>(
            values, t.n, t.uids, t.cvalues, t.ccounts, t.pvalues, t.pcounts,
            rnidx, a.rnodes, a.nrnodes, a.r[j], std::abs(a.z[j]));
}

// Evaluate each trait of the set of a DiskTraitsArgs at its point j, as
// the disk kernels do: in the ring of the radius of the point, with the
// node-wise parameters interpolated linearly between the (sub)nodes. The
// values are those of the traits before the kernels combine them (e.g.
// a velocity trait projected, without the systemic velocity). The disk
// is closed: at the radius of its last node, the values are those its
// last ring tends to there. Off the disk (a radius outside [rnodes[0],
// rnodes[nrnodes - 1]]), where the kernels evaluate nothing, the values
// are NaN.
template<typename T> constexpr void
gmodel_disk_traits_point(int j, const DiskTraitsArgs<T>& a)
{
    T values[TRAIT_NUM_MAX] = {0};
    int rnidx = a.nrnodes - 1;
    if (a.r[j] != a.rnodes[rnidx]
            && !disk_info(rnidx, a.r[j], a.nrnodes, a.rnodes)) {
        for (int i = 0; i < a.traits.n; ++i)
            values[i] = NAN;
    } else {
        switch (a.kind)
        {
        case TraitKind::RPT:
            gmodel_disk_ptraits_point<rp_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::VPT:
            gmodel_disk_ptraits_point<vp_trait<T>>(values, j, rnidx, a);
            vp_disk_project_traits(
                    values, a.traits.n, a.directions, a.r[j],
                    a.theta[j] * DEG_TO_RAD<T>, a.incl * DEG_TO_RAD<T>);
            break;
        case TraitKind::DPT:
            gmodel_disk_ptraits_point<dp_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::ZPT:
            gmodel_disk_ptraits_point<zp_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::SPT:
            gmodel_disk_ptraits_point<sp_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::OPT:
            gmodel_disk_ptraits_point<rp_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::RHT:
            gmodel_disk_htraits_point<rh_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::VHT:
            gmodel_disk_htraits_point<vh_trait<T>>(values, j, rnidx, a);
            break;
        case TraitKind::DHT:
            gmodel_disk_htraits_point<dh_trait<T>>(values, j, rnidx, a);
            break;
        }
    }
    for (int i = 0; i < a.traits.n; ++i)
        a.out[i * a.npoints + j] = values[i];
}

} // namespace gbkfit
