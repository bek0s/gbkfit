#pragma once

#include "gbkfit/constants.hpp"
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
// 0. The channels share their edges, so each channel needs one cdf.
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
    T cdf_lo = gauss_1d_cdf(
            spec_zero + (cmin - T{0.5}) * spec_step, centre, sigma);
    for (int c = cmin; c <= cmax; ++c)
    {
        const T cdf_hi = gauss_1d_cdf(
                spec_zero + (c + T{0.5}) * spec_step, centre, sigma);
        add(c, flux * (cdf_hi - cdf_lo) / spec_step);
        cdf_lo = cdf_hi;
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
// (see DiskArgs::opacity): the optical depth of the voxels in front of it
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
gmodel_voxel_opacity(int x, int y, int z, const DiskArgs<T>& a)
{
    const long long nspat = 1LL * a.spat_size[0] * a.spat_size[1]
            * a.spat_size[2];
    VoxelOpacity<T> o;
    for (int k = 0; k < a.nopacity; ++k)
    {
        const T* opacity = a.opacity + k * nspat;
        o.front[k] = z > 0
                ? opacity[index_3d_to_1d(
                        x, y, z - 1, a.spat_size[0], a.spat_size[1])]
                : T{0};
        o.own[k] = opacity[index_3d_to_1d(
                x, y, z, a.spat_size[0], a.spat_size[1])] - o.front[k];
    }
    return o;
}

// The opacity of a thin layer at the position pz along the line of sight
// of the spaxel (x, y) (see gmodel_optical_depth): all of it in front
template<typename T> constexpr VoxelOpacity<T>
gmodel_layer_opacity(int x, int y, T pz, const DiskArgs<T>& a)
{
    const long long nspat = 1LL * a.spat_size[0] * a.spat_size[1]
            * a.spat_size[2];
    VoxelOpacity<T> o;
    for (int k = 0; k < a.nopacity; ++k)
    {
        o.front[k] = gmodel_optical_depth(
                x, y, pz, a.opacity + k * nspat, a.spat_size);
        o.own[k] = 0;
    }
    return o;
}

// The optical depth of the opacity o of n cubes in front of the light and
// of its own voxel, at the wavelength of the light: ratios has the ratio
// of each cube there (a row of DiskArgs::opacity_ratios)
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

// Add the emission lines (see DiskArgs::lines) of light of the given flux,
// velocity and dispersion to the spectrum (x, y) of a spectral cube. The
// opacity o of nopacity cubes (if not null) dims each line with its row of
// ratios (see DiskArgs::opacity_ratios).
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

template<typename T> constexpr T
rnode_radius(
        T x, T y, bool loose, bool tilted, int index,
        const T* xpos, const T* ypos, const T* posa, const T* incl)
{
    T xposi = loose ? xpos[index] : xpos[0];
    T yposi = loose ? ypos[index] : ypos[0];
    T posai = tilted ? posa[index] : posa[0];
    T incli = tilted ? incl[index] : incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;
    transform_cpos_posa_incl(x, y, xposi, yposi, posai, incli);
    return std::sqrt(x * x + y * y);
}

template<typename T> constexpr T
rnode_radius(
        T x, T y, T z, bool loose, bool tilted, int index,
        const T* xpos, const T* ypos, const T* posa, const T* incl)
{
    T xposi = loose ? xpos[index] : xpos[0];
    T yposi = loose ? ypos[index] : ypos[0];
    T posai = tilted ? posa[index] : posa[0];
    T incli = tilted ? incl[index] : incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;
    transform_cpos_posa_incl(x, y, z, xposi, yposi, posai, incli);
    return std::sqrt(x * x + y * y);
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
        const T* xpos, const T* ypos, const T* posa, const T* incl)
{
    T radius_min = rnode_radius(
            x, y, loose, tilted, 0, xpos, ypos, posa, incl);

    // Ignore anything smaller than the first node
    if (radius_min < nodes[0])
        return false;

    T radius_max = rnode_radius(
            x, y, loose, tilted, nnodes-1, xpos, ypos, posa, incl);

    // Ignore anything larger than the last node
    if (radius_max >= nodes[nnodes-1])
        return false;
#if 1
    // Linear search
    for(index = 1; index < nnodes; ++index)
    {
        T radius_cur = rnode_radius(
                x, y, loose, tilted, index, xpos, ypos, posa, incl);
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
                x, y, loose, tilted, index, xpos, ypos, posa, incl);
        if(nodes[index] > radius_cur)
            ihi = index;
        else
            ilo = index;
    }
#endif
    // The radius is calculated using the velfi strategy
    T d1 = nodes[index-1] - rnode_radius(
            x, y, loose, tilted, index-1, xpos, ypos, posa, incl);
    T d2 = nodes[index] - rnode_radius(
            x, y, loose, tilted, index, xpos, ypos, posa, incl);
    radius = (nodes[index] * d1 - nodes[index - 1] * d2) / (d1 - d2);

    return true;
}

template<typename T> constexpr bool
ring_info(
        int& index, T& radius,
        T x, T y, T z,
        bool loose, bool tilted, int nnodes, const T* nodes,
        const T* xpos, const T* ypos, const T* posa, const T* incl)
{
    T radius_min = rnode_radius(
            x, y, z, loose, tilted, 0, xpos, ypos, posa, incl);

    // Ignore anything smaller than the first node
    if (radius_min < nodes[0])
        return false;

    T radius_max = rnode_radius(
            x, y, z, loose, tilted, nnodes-1, xpos, ypos, posa, incl);

    // Ignore anything larger than the last node
    if (radius_max >= nodes[nnodes-1])
        return false;
#if 1
    // Linear search
    for(index = 1; index < nnodes; ++index)
    {
        T radius_cur = rnode_radius(
                x, y, z, loose, tilted, index, xpos, ypos, posa, incl);
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
                x, y, z, loose, tilted, index, xpos, ypos, posa, incl);
        if(nodes[index] > radius_cur)
            ihi = index;
        else
            ilo = index;
    }
#endif
    // The radius is calculated using the velfi strategy
    T d1 = nodes[index-1] - rnode_radius(
            x, y, z, loose, tilted, index-1, xpos, ypos, posa, incl);
    T d2 = nodes[index] - rnode_radius(
            x, y, z, loose, tilted, index, xpos, ypos, posa, incl);
    radius = (nodes[index] * d1 - nodes[index - 1] * d2) / (d1 - d2);

    return true;
}

// The number of planes of line opacity in the fields of a thick disk (see
// FieldsArgs::fields): one for each line with line opacity, none without
template<typename T> constexpr int
gmodel_fields_lines(const DiskArgs<T>& a, const FieldsArgs<T>& f)
{
    return f.opt.uids ? a.nlines : 0;
}

// The coefficient of the line opacity of each line (see FieldsArgs::opt)
// at a point of the plane of the disk, in the ring rnidx, into
// coefficients (one for each line; there are at most TRAIT_NUM_MAX, one
// for each trait)
template<typename T> constexpr void
gmodel_line_opacity(
        T* coefficients, int rnidx, T x, T y, T r, T theta,
        const DiskArgs<T>& a, const FieldsArgs<T>& f)
{
    T values[TRAIT_NUM_MAX] = {0};
    p_traits<rp_trait<T>>(
            values,
            f.opt.n, f.opt.uids,
            f.opt.cvalues, f.opt.ccounts,
            f.opt.pvalues, f.opt.pcounts,
            rnidx, a.rnodes, a.nrnodes,
            x, y, r, theta);
    for (int l = 0; l < a.nlines; ++l)
        coefficients[l] = 0;
    for (int i = 0; i < f.opt.n; ++i)
        if (f.opt_rows[i] >= 0)
            coefficients[f.opt_rows[i]] += values[i];
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
// opacity dims the disk (a.opacity is not null), HasFields whether it
// writes the fields of the march instead of its light (f.fields is not
// null), and HasLineOpacity whether they have its line opacity (f.opt has
// traits): template parameters, so that the other kernels have none of
// their code (nor its registers and local memory, on a GPU).
template<
        auto AtomicAddFunT, bool HasOpacity, bool HasFields,
        bool HasLineOpacity, typename T>
constexpr void
gmodel_mcdisk_evaluate_cloud(
        RNG<T>& rng, int ci, const DiskArgs<T>& a, const MCDiskArgs<T>& mc,
        const FieldsArgs<T>& f)
{
    // This is a placeholder in case we decide to explicitly
    // add a Monte Carlo based thin disk in the future.
    const bool is_thin = false;

    int rnidx=0, tidx=0;
    const T* rpt_cptr = a.rpt.cvalues;
    const T* rpt_pptr = a.rpt.pvalues;
    const T* rht_cptr = a.rht.cvalues;
    const T* rht_pptr = a.rht.pvalues;
    T xd=0, yd=0, zd=0, rd=0, theta=0, sign=1;
    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T ptvalues[TRAIT_NUM_MAX] = {0};
    T htvalues[TRAIT_NUM_MAX] = {0};
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
    rvalue /= a.spat_step[0] * a.spat_step[1];

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
    if (a.spt.uids)
    {
        p_traits<sp_trait<T>>(
                ptvalues,
                a.spt.n, a.spt.uids,
                a.spt.cvalues, a.spt.ccounts,
                a.spt.pvalues, a.spt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xd, yd, rd, theta);

        // A point is kept only if every trait selects it
        for (int i = 0; i < a.spt.n; ++i)
            if (!ptvalues[i])
                return;
    }

    // Vertical distortion traits: the displacement of the midplane
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
    }

    // Density height trait: the height of the cloud above the midplane,
    // which the distortion displaces (a cloud of a profile that is none,
    // at a NaN height, is discarded below)
    rh_trait_rnd<T>(
                zd, rng, a.rht.uids[tidx], rht_cptr, rht_pptr,
                rnidx, a.rnodes, a.nrnodes, rd, zvalue);
    const T height = zd;
    zd += zvalue;

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
    transform_rh_rotate_z(xn, yn, xn, yn, -a.spat_rota * DEG_TO_RAD<T>);

    // The voxel of the cloud. Discard the clouds outside the cube, while
    // their positions are floats: converting one out of the range of int
    // (e.g. of a heavy-tailed height), or NaN, is undefined.
    const T px = std::rint((xn - a.spat_zero[0]) / a.spat_step[0]);
    const T py = std::rint((yn - a.spat_zero[1]) / a.spat_step[1]);
    const T pz = std::rint((zn - a.spat_zero[2]) / a.spat_step[2])
            - a.slab_zero;
    if (!(px >= 0 && px < a.spat_size[0] &&
          py >= 0 && py < a.spat_size[1] &&
          pz >= 0 && pz < a.spat_size[2])) {
        return;
    }
    const int x = px;
    const int y = py;
    const int z = pz;

    // Velocity traits
    if (a.vpt.uids)
    {
        p_traits<vp_trait<T>>(
                ptvalues,
                a.vpt.n, a.vpt.uids,
                a.vpt.cvalues, a.vpt.ccounts,
                a.vpt.pvalues, a.vpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xd, yd, rd, theta, incli);
    }
    if (a.vht.uids)
    {
        h_traits<vh_trait<T>>(
                htvalues,
                a.vpt.n, a.vht.uids,
                a.vht.cvalues, a.vht.ccounts,
                a.vht.pvalues, a.vht.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                rd, std::abs(height));
    }
    for (int i = 0; i < a.vpt.n; ++i)
        vvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Apply systemic velocity
    vvalue += vsysi;

    // Dispersion traits
    if (a.dpt.uids)
    {
        p_traits<dp_trait<T>>(
                ptvalues,
                a.dpt.n, a.dpt.uids,
                a.dpt.cvalues, a.dpt.ccounts,
                a.dpt.pvalues, a.dpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xd, yd, rd, theta);
    }
    if (a.dht.uids)
    {
        h_traits<dh_trait<T>>(
                htvalues,
                a.dpt.n, a.dht.uids,
                a.dht.cvalues, a.dht.ccounts,
                a.dht.pvalues, a.dht.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                rd, std::abs(height));
    }
    for (int i = 0; i < a.dpt.n; ++i)
        dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Ensure positive dispersion
    // Ideally, we want the caller to never use parameter values that result in
    // a negative dispersion. Unfortunatelly, this is hard to achieve.
    // Forcing a positive dispersion is a reasonable solution.
    dvalue = std::abs(dvalue);


    // The opacity in front of the light, and the light that it lets
    // through (that of the image, or of the first line)
    VoxelOpacity<T> opacity;
    if (HasOpacity) {
        opacity = gmodel_voxel_opacity(x, y, z, a);
    }
    const T orvalue = HasOpacity
            ? rvalue * gmodel_attenuation(
                    opacity, a.nopacity, a.opacity_ratios)
            : rvalue;

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AtomicAddFunT>(
                a.image, x, y, orvalue * a.image_scale,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AtomicAddFunT>(
                a.scube, x, y, rvalue, vvalue, dvalue,
                a.nlines, a.lines,
                HasOpacity ? &opacity : nullptr, a.nopacity, a.opacity_ratios,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
    }

    const int idx = index_3d_to_1d(x, y, z, a.spat_size[0], a.spat_size[1]);

    if (a.rdata) {
        AtomicAddFunT(&a.rdata[idx], rvalue * a.rdata_scale);
    }
    if (a.rdata_cmp) {
        AtomicAddFunT(&a.rdata_cmp[idx], rvalue);
    }
    if (a.ordata) {
        AtomicAddFunT(&a.ordata[idx], orvalue);
    }
    if (a.ordata_cmp) {
        AtomicAddFunT(&a.ordata_cmp[idx], orvalue);
    }
    // The clouds of a voxel add to the means of its velocity and dispersion
    if (a.vdata_cmp) {
        AtomicAddFunT(&a.vdata_cmp[idx], std::abs(rvalue) * vvalue);
    }
    if (a.ddata_cmp) {
        AtomicAddFunT(&a.ddata_cmp[idx], std::abs(rvalue) * dvalue);
    }
    if (a.vdweight_cmp) {
        AtomicAddFunT(&a.vdweight_cmp[idx], std::abs(rvalue));
    }
    // The moments of the clouds of the voxel (see
    // gmodel_mcdisk_fields_moments), and their light times the coefficient
    // of each line
    if (HasFields) {
        const long long nspat = 1LL * a.spat_size[0] * a.spat_size[1]
                * a.spat_size[2];
        const int nk = HasLineOpacity ? a.nlines : 0;
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
                    coefficients, rnidx, xd, yd, rd, theta, a, f);
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
        long long idx, const DiskArgs<T>& a, const FieldsArgs<T>& f)
{
    const long long nspat = 1LL * a.spat_size[0] * a.spat_size[1]
            * a.spat_size[2];
    T* v = &f.fields[nspat + idx];
    T* s = &f.fields[2 * nspat + idx];
    const T weight = f.fields[(3 + gmodel_fields_lines(a, f)) * nspat + idx];
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
gmodel_smdisk_size_z(const DiskArgs<T>& a)
{
    return a.rht.uids ? a.spat_size[2] : 1;
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
        int x, int y, int z, const DiskArgs<T>& a, const FieldsArgs<T>& f)
{
    bool is_thin = a.rht.uids == nullptr;
    bool has_depth = a.spat_step[2] != 0;

    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T xn=x, yn=y, zn=z, rn=0, theta=0;
    int rnidx = -1;

    // image-to-world transform
    xn = a.spat_zero[0] + x * a.spat_step[0];
    yn = a.spat_zero[1] + y * a.spat_step[1];
    zn = a.spat_zero[2] + (z + a.slab_zero) * a.spat_step[2];
    transform_rh_rotate_z(xn, yn, xn, yn, a.spat_rota * DEG_TO_RAD<T>);

    // If the disk is loose or tilted, we need to calculate the pixel's
    // radial node index and radius now.
    if (a.loose || a.tilted)
    {
        bool is_on_disk = is_thin
                ? ring_info(
                    rnidx, rn, xn, yn, a.loose, a.tilted,
                    a.nrnodes, a.rnodes, a.xpos, a.ypos, a.posa, a.incl)
                : ring_info(
                    rnidx, rn, xn, yn, zn, a.loose, a.tilted,
                    a.nrnodes, a.rnodes, a.xpos, a.ypos, a.posa, a.incl);
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
        pz = (yn * std::sin(incli) - a.spat_zero[2]) / a.spat_step[2];
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
    T htvalues[TRAIT_NUM_MAX] = {0};
    T rvalue=0, vvalue=0, dvalue=0, zvalue=0;

    // Selection traits
    if (a.spt.uids)
    {
        p_traits<sp_trait<T>>(
                ptvalues,
                a.spt.n, a.spt.uids,
                a.spt.cvalues, a.spt.ccounts,
                a.spt.pvalues, a.spt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta);

        // A point is kept only if every trait selects it
        for (int i = 0; i < a.spt.n; ++i)
            if (!ptvalues[i])
                return;
    }

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
    if (a.rpt.uids)
    {
        p_traits<rp_trait<T>>(
                ptvalues,
                a.rpt.n, a.rpt.uids,
                a.rpt.cvalues, a.rpt.ccounts,
                a.rpt.pvalues, a.rpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta);
    }
    if (a.rht.uids)
    {
        h_traits<rh_trait<T>>(
                htvalues,
                a.rpt.n, a.rht.uids,
                a.rht.cvalues, a.rht.ccounts,
                a.rht.pvalues, a.rht.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                rn, zn, zvalue);
    }
    for (int i = 0; i < a.rpt.n; ++i)
        rvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Discart pixels with zero density
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
        rvalue *= a.spat_step[2];
    }

    // Velocity traits
    if (a.vpt.uids)
    {
        p_traits<vp_trait<T>>(
                ptvalues,
                a.vpt.n, a.vpt.uids,
                a.vpt.cvalues, a.vpt.ccounts,
                a.vpt.pvalues, a.vpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta, incli);
    }
    if (a.vht.uids)
    {
        h_traits<vh_trait<T>>(
                htvalues,
                a.vpt.n, a.vht.uids,
                a.vht.cvalues, a.vht.ccounts,
                a.vht.pvalues, a.vht.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                rn, std::abs(zn));
    }
    for (int i = 0; i < a.vpt.n; ++i)
        vvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Apply systemic velocity
    vvalue += vsysi;

    // Dispersion traits
    if (a.dpt.uids)
    {
        p_traits<dp_trait<T>>(
                ptvalues,
                a.dpt.n, a.dpt.uids,
                a.dpt.cvalues, a.dpt.ccounts,
                a.dpt.pvalues, a.dpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta);
    }
    if (a.dht.uids)
    {
        h_traits<dh_trait<T>>(
                htvalues,
                a.dpt.n, a.dht.uids,
                a.dht.cvalues, a.dht.ccounts,
                a.dht.pvalues, a.dht.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                rn, std::abs(zn));
    }
    for (int i = 0; i < a.dpt.n; ++i)
        dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Ensure positive dispersion
    // Ideally, we want the caller to never use parameter values that result in
    // a negative dispersion. Unfortunatelly, this is hard to achieve.
    // Forcing a positive dispersion is a reasonable solution.
    dvalue = std::abs(dvalue);


    // The opacity in front of the light (that of a thin disk is a layer,
    // which only the opacity in front of it dims), and the light that it
    // lets through (that of the image, or of the first line)
    VoxelOpacity<T> opacity;
    if (HasOpacity) {
        opacity = is_thin
                ? gmodel_layer_opacity(x, y, pz, a)
                : gmodel_voxel_opacity(x, y, z, a);
    }
    const T orvalue = HasOpacity
            ? rvalue * gmodel_attenuation(
                    opacity, a.nopacity, a.opacity_ratios)
            : rvalue;

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AddFunT>(
                a.image, x, y, orvalue * a.image_scale,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AddFunT>(
                a.scube, x, y, rvalue, vvalue, dvalue,
                a.nlines, a.lines,
                HasOpacity ? &opacity : nullptr, a.nopacity, a.opacity_ratios,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
    }

    // The fields of the layer of a thin disk, wherever its plane crosses
    // the line of sight (see FieldsArgs::fields)
    if (HasFields && is_thin) {
        const long long nimage = 1LL * a.spat_size[0] * a.spat_size[1];
        const int idx = index_2d_to_1d(x, y, a.spat_size[0]);
        f.fields[idx] = rvalue;
        f.fields[nimage + idx] = vvalue;
        f.fields[2 * nimage + idx] = dvalue;
        f.fields[3 * nimage + idx] = pz;
    }

    // The voxel of the 3d outputs (a thin disk outside the grid has none).
    // The position is checked while it is a float: converting one out of
    // the range of int, or NaN, is undefined.
    const T vz = std::rint(pz);
    if (!(vz >= 0 && vz < a.spat_size[2])) {
        return;
    }
    const int idx = index_3d_to_1d(
            x, y, int(vz), a.spat_size[0], a.spat_size[1]);

    if (a.rdata) {
        a.rdata[idx] += rvalue * a.rdata_scale;
    }
    if (a.rdata_cmp) {
        a.rdata_cmp[idx] = rvalue;
    }
    if (a.ordata) {
        a.ordata[idx] += orvalue;
    }
    if (a.ordata_cmp) {
        a.ordata_cmp[idx] = orvalue;
    }
    // The means of the velocity and dispersion of the voxel (of one value)
    if (a.vdata_cmp) {
        a.vdata_cmp[idx] = std::abs(rvalue) * vvalue;
    }
    if (a.ddata_cmp) {
        a.ddata_cmp[idx] = std::abs(rvalue) * dvalue;
    }
    if (a.vdweight_cmp) {
        a.vdweight_cmp[idx] = std::abs(rvalue);
    }
    // The fields of the voxel of a thick disk, and its light times the
    // coefficient of each line
    if (HasFields && !is_thin) {
        const long long nspat = 1LL * a.spat_size[0] * a.spat_size[1]
                * a.spat_size[2];
        f.fields[idx] = rvalue;
        f.fields[nspat + idx] = vvalue;
        f.fields[2 * nspat + idx] = dvalue;
        if (HasLineOpacity) {
            T coefficients[TRAIT_NUM_MAX];
            gmodel_line_opacity(
                    coefficients, rnidx, xn, yn, rn, theta, a, f);
            for (int l = 0; l < a.nlines; ++l)
                f.fields[(3 + l) * nspat + idx] = rvalue * coefficients[l];
        }
    }
}

// The polar traits of the set of a DiskTraitsArgs at its point j, in the
// ring rnidx, into values. extra has the further arguments of PTraitFunT:
// the inclination (radians) of the velocity traits.
template<auto PTraitFunT, typename T, typename... Ts> constexpr void
gmodel_disk_ptraits_point(
        T* values, int j, int rnidx, const DiskTraitsArgs<T>& a, Ts... extra)
{
    const TraitSet<T>& t = a.traits;
    const T r = a.r[j];
    const T theta = a.theta[j] * DEG_TO_RAD<T>;
    p_traits<PTraitFunT>(
            values, t.n, t.uids, t.cvalues, t.ccounts, t.pvalues, t.pcounts,
            rnidx, a.rnodes, a.nrnodes,
            r * std::cos(theta), r * std::sin(theta), r, theta, extra...);
}

// The height traits of the set of a DiskTraitsArgs at its point j, in the
// ring rnidx, into values: at |z|, as the disk kernels evaluate them, with
// the extra arguments of the traits (e.g. the displacement of the midplane
// of the density height traits)
template<auto HTraitFunT, typename T, typename ...Ts> constexpr void
gmodel_disk_htraits_point(
        T* values, int j, int rnidx, const DiskTraitsArgs<T>& a,
        Ts ...extra)
{
    const TraitSet<T>& t = a.traits;
    h_traits<HTraitFunT>(
            values, t.n, t.uids, t.cvalues, t.ccounts, t.pvalues, t.pcounts,
            rnidx, a.rnodes, a.nrnodes, a.r[j], std::abs(a.z[j]), extra...);
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
            gmodel_disk_ptraits_point<vp_trait<T>>(
                    values, j, rnidx, a, a.incl * DEG_TO_RAD<T>);
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
            // (above a midplane on the plane of the galaxy)
            gmodel_disk_htraits_point<rh_trait<T>>(
                    values, j, rnidx, a, T{0});
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
