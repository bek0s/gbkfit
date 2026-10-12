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

// Add a line of the given flux, centre and dispersion (in the units of
// the spectral axis) to the spectrum (x, y) of a spectral cube
template<auto AddFunT, typename T> void constexpr
gmodel_scube_line(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size_z,
        T spec_step,
        T spec_zero)
{
    // A line of a velocity or dispersion that is not a number (e.g. of
    // parameters out of their range) makes the spectrum NaN, as a flux
    // that is not a number does, so that it is seen
    if (std::isnan(vvalue) || std::isnan(dvalue))
    {
        for (int z = 0; z < spec_size_z; ++z)
        {
            int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
            AddFunT(&scube[idx], T(NAN));
        }
        return;
    }

    // Calculate a spectral range that encloses most of the flux, within
    // the spectral axis. The channels are clamped before they become
    // ints: converting a float out of the range of int is undefined.
    T zmin = vvalue - dvalue * LINE_WIDTH_MULTIPLIER<T>;
    T zmax = vvalue + dvalue * LINE_WIDTH_MULTIPLIER<T>;
    T zmin_pos = std::max<T>(std::rint((zmin - spec_zero) / spec_step), 0);
    T zmax_pos = std::min<T>(
            std::rint((zmax - spec_zero) / spec_step), spec_size_z - 1);
    if (zmin_pos > zmax_pos)
        return;
    int zmin_idx = zmin_pos;
    int zmax_idx = zmax_pos;

    // Evaluate the spectral line within the range specified above
    // Evaluating only within the range can result in huge speed increase.
    // Each channel gets the mean of the line over the channel (the cube
    // holds flux per unit velocity), which conserves the flux of lines of
    // any width, including 0. The channels share their edges, so each
    // channel needs one cdf.
    T cdf_lo = gauss_1d_cdf(
            spec_zero + (zmin_idx - T{0.5}) * spec_step, vvalue, dvalue);
    for (int z = zmin_idx; z <= zmax_idx; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        T cdf_hi = gauss_1d_cdf(
                spec_zero + (z + T{0.5}) * spec_step, vvalue, dvalue);
        AddFunT(&scube[idx], rvalue * (cdf_hi - cdf_lo) / spec_step);
        cdf_lo = cdf_hi;
    }
}

// Add the emission lines (see DiskArgs::lines) of light of the given flux,
// velocity and dispersion to the spectrum (x, y) of a spectral cube
template<auto AddFunT, typename T> void constexpr
gmodel_scube_evaluate(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int nlines, const T* lines,
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
        gmodel_scube_line<AddFunT>(
                scube, x, y,
                rvalue * flux, offset + scale * vvalue, scale * dvalue,
                spat_size_x, spat_size_y,
                spec_size_z, spec_step, spec_zero);
    }
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

// The fraction of the light of the voxel (x, y, z) that reaches the
// viewer, who is before the first voxel along z (the velocities along +z
// recede from the viewer). The opacity cube holds the optical depth from
// the viewer to the far side of each voxel (see
// gmodel_opacity_accumulate). The light crosses the voxels in front of its
// own, and the absorbers of its own voxel, which are mixed uniformly with
// the emitters: on average, (1 - exp(-tau)) / tau of the light of a voxel
// of optical depth tau leaves it.
template<typename T> constexpr T
gmodel_attenuation(int x, int y, int z, const T* opacity, const int* size)
{
    const T tau_front = z > 0
            ? opacity[index_3d_to_1d(x, y, z - 1, size[0], size[1])]
            : T{0};
    const T tau_own =
            opacity[index_3d_to_1d(x, y, z, size[0], size[1])] - tau_front;
    const T own = tau_own != 0 ? -std::expm1(-tau_own) / tau_own : T{1};
    return std::exp(-tau_front) * own;
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

template<auto AtomicAddFunT, typename T> constexpr void
gmodel_mcdisk_evaluate_cloud(
        RNG<T>& rng, int ci, const DiskArgs<T>& a, const MCDiskArgs<T>& mc)
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

    // Density height trait: the height of the cloud above the midplane
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
    transform_rh_rotate_z(xn, yn, xn, yn, -a.spat_rota * DEG_TO_RAD<T>);

    // The voxel of the cloud. Discard the clouds outside the cube, while
    // their positions are floats: converting one out of the range of int
    // (e.g. of a heavy-tailed height), or NaN, is undefined.
    const T px = std::rint((xn - a.spat_zero[0]) / a.spat_step[0]);
    const T py = std::rint((yn - a.spat_zero[1]) / a.spat_step[1]);
    const T pz = std::rint((zn - a.spat_zero[2]) / a.spat_step[2]);
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


    // The light that the opacity lets through
    T orvalue = a.opacity
            ? rvalue * gmodel_attenuation(x, y, z, a.opacity, a.spat_size)
            : rvalue;

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AtomicAddFunT>(
                a.image, x, y, orvalue,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AtomicAddFunT>(
                a.scube, x, y, orvalue, vvalue, dvalue,
                a.nlines, a.lines,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
    }

    const int idx = index_3d_to_1d(x, y, z, a.spat_size[0], a.spat_size[1]);

    if (a.rdata) {
        AtomicAddFunT(&a.rdata[idx], rvalue);
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
// writes to its other outputs.
template<auto AddFunT, typename T> constexpr void
gmodel_smdisk_evaluate_spaxel(int x, int y, int z, const DiskArgs<T>& a)
{
    bool is_thin = a.rht.uids == nullptr;
    bool has_depth = a.spat_step[2] != 0;

    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T xn=x, yn=y, zn=z, rn=0, theta=0;
    int rnidx = -1;

    // image-to-world transform
    xn = a.spat_zero[0] + x * a.spat_step[0];
    yn = a.spat_zero[1] + y * a.spat_step[1];
    zn = a.spat_zero[2] + z * a.spat_step[2];
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
                rn, std::abs(zn));
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


    // The light that the opacity lets through: that of a thin disk is a
    // layer, which only the opacity in front of it dims
    T orvalue = rvalue;
    if (a.opacity) {
        orvalue *= is_thin
                ? std::exp(-gmodel_optical_depth(
                        x, y, pz, a.opacity, a.spat_size))
                : gmodel_attenuation(x, y, z, a.opacity, a.spat_size);
    }

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AddFunT>(
                a.image, x, y, orvalue,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AddFunT>(
                a.scube, x, y, orvalue, vvalue, dvalue,
                a.nlines, a.lines,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
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
        a.rdata[idx] += rvalue;
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
