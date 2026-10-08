#pragma once

#include "gbkfit/constants.hpp"
#include "gbkfit/gmodel/disk.hpp"
#include "gbkfit/gmodel/traits.hpp"
#include "gbkfit/random.hpp"
#include "gbkfit/utilities/indexutils.hpp"

namespace gbkfit {

template<typename T> constexpr void
gmodel_wcube_pixel(
        int x, int y,
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_wcube,
        T* spec_wcube)
{
    T sum = 0;
    T mean = 0;
    T maximum = 0;

    // Find the maximum value and sum at position (x, y) of the spatial cube
    for(int z = 0; z < spat_size_z; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        T value = spat_wcube[idx];
        sum += value;
        maximum = std::max(maximum, value);
    }

    // Calculate mean weight
    mean = sum / spat_size_z;

    // Assign the same mean weight across the entire spectrum
    for(int z = 0; z < spec_size_z; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        spec_wcube[idx] = maximum > 0 ? mean / maximum : 0;
    }
}

template<auto AtomicAddFunT, typename T> void constexpr
gmodel_image_evaluate(T* image, int x, int y, T rvalue, int spat_size_x)
{
    const int idx = index_2d_to_1d(x, y, spat_size_x);
    AtomicAddFunT(&image[idx], rvalue);
}

template<auto AtomicAddFunT, typename T> void constexpr
gmodel_scube_evaluate(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size_z,
        T spec_step,
        T spec_zero)
{
    // Calculate a spectral range that encloses most of the flux.
    T zmin = vvalue - dvalue * LINE_WIDTH_MULTIPLIER<T>;
    T zmax = vvalue + dvalue * LINE_WIDTH_MULTIPLIER<T>;
    int zmin_idx = std::max<T>(std::rint(
            (zmin - spec_zero)/spec_step), 0);
    int zmax_idx = std::min<T>(std::rint(
            (zmax - spec_zero)/spec_step), spec_size_z - 1);

    // Evaluate the spectral line within the range specified above
    // Evaluating only within the range can result in huge speed increase
    for (int z = zmin_idx; z <= zmax_idx; ++z)
    {
        int idx = index_3d_to_1d(x, y, z, spat_size_x, spat_size_y);
        T zvel = spec_zero + z * spec_step;
        T flux = rvalue * gauss_1d_pdf(zvel, vvalue, dvalue); // * spec_step;
        AtomicAddFunT(&scube[idx], flux);
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

    // Linear search
    // TODO: Why 1?
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

template<auto AtomicAssignFunT, auto AtomicAddFunT, typename T> constexpr void
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
    T rvalue=0, vvalue=0, dvalue=0, zvalue=0, svalue=0, wvalue=1;

    // All clouds have equal flux
    rvalue = mc.cflux / a.spat_step[2];

    // Find which cumulative sum the cloud belongs to.
    while(ci >= mc.ncloudscsum[rnidx]) {
        rnidx++;
    }

    // Find which trait and subring the cloud belongs to.
    for(tidx = 0; tidx < a.rpt.n; ++tidx)
    {
        int size = mc.hasordint[tidx] ? 1 : a.nrnodes - 2; // -2 ?
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

    if (sign < 0) {
        rvalue = -rvalue;
    }

    // Integrate along z dimension
    rvalue *= a.spat_step[2];

    // Convert to surface brightness
    rvalue /= a.spat_step[0] * a.spat_step[1];

    // Calculate cartesian coordinates on disk plane.
    xd = rd * std::cos(theta);
    yd = rd * std::sin(theta);

    // Recalculate radial node index.
    // This is done in order to account for:
    //  - rptraits with ordinary integral (no subrings).
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

        for (int i = 0; i < a.spt.n; ++i)
            svalue += ptvalues[i];

        if (!svalue)
            return;
    }

    // Density height trait
    rh_trait_rnd<T>(
                zd, rng, a.rht.uids[tidx], rht_cptr, rht_pptr,
                rnidx, a.rnodes, a.nrnodes, rd);

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

    //
    int x = std::rint((xn - a.spat_zero[0])/a.spat_step[0]);
    int y = std::rint((yn - a.spat_zero[1])/a.spat_step[1]);
    int z = std::rint((zn - a.spat_zero[2])/a.spat_step[2]);

    // Discard pixels outside the image/cube
    if (x < 0 || x >= a.spat_size[0] ||
        y < 0 || y >= a.spat_size[1] ||
        z < 0 || z >= a.spat_size[2]) {
        return;
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
                rd, std::abs(zd));
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
                rd, std::abs(zd));
    }
    for (int i = 0; i < a.dpt.n; ++i)
        dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Ensure positive dispersion
    // Ideally, we want the caller to never use parameter values that result in
    // a negative dispersion. Unfortunatelly, this is hard to achieve.
    // Forcing a positive dispersion is a reasonable solution.
    dvalue = std::abs(dvalue);

    // Weight polar traits
    if (a.wpt.uids && a.wdata)
    {
        p_traits<wp_trait<T>>(
                ptvalues,
                a.wpt.n, a.wpt.uids,
                a.wpt.cvalues, a.wpt.ccounts,
                a.wpt.pvalues, a.wpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xd, yd, rd, theta);

        for (int i = 0; i < a.wpt.n; ++i)
            wvalue *= ptvalues[i];
    }

    // Apply opacity to the calculated density.
    T orvalue = rvalue;
    if (a.opacity)
    {
        // Apply the opacity of all the spaxels between the current spatial
        // position and the viewer. Do not include the current spatial position.
        for(int oz = z + 1; oz < a.spat_size[2]; ++oz)
        {
            const auto idx = index_3d_to_1d(
                    x, y, oz, a.spat_size[0], a.spat_size[1]);
            orvalue -= orvalue * a.opacity[idx];
        }
    }

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AtomicAddFunT>(
                a.image, x, y, orvalue,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AtomicAddFunT>(
                a.scube, x, y, orvalue, vvalue, dvalue,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
    }

    const int idx = index_3d_to_1d(x, y, z, a.spat_size[0], a.spat_size[1]);

    if (a.wdata) {
        // TODO
        AtomicAssignFunT(&a.wdata[idx], wvalue);
    }
    if (a.wdata_cmp) {
        // For overlapping clouds, keep the last weight
        // Storing the mean would be too much effort with little reward
        AtomicAssignFunT(&a.wdata_cmp[idx], wvalue);
    }
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
    if (a.vdata_cmp) {
        // For overlapping clouds, keep the last velocity
        // Storing the mean would be too much effort with little reward
        AtomicAssignFunT(&a.vdata_cmp[idx], vvalue);
    }
    if (a.ddata_cmp) {
        // For overlapping clouds, keep the last dispersion
        // Storing the mean would be too much effort with little reward
        AtomicAssignFunT(&a.ddata_cmp[idx], dvalue);
    }
}

template<auto AtomicAddFunT, typename T> constexpr void
gmodel_smdisk_evaluate_spaxel(int x, int y, int z, const DiskArgs<T>& a)
{
    bool is_thin = a.rht.uids == nullptr;

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
    theta = std::atan2(yn, xn);

    // If the disk is not loose or tilted, we need to calculate the pixel's
    // radial node index and radius now.
    if (!(a.loose || a.tilted)) {
        rn = std::sqrt(xn * xn + yn * yn);
        bool is_on_disk = disk_info(rnidx, rn, a.nrnodes, a.rnodes);
        if (!is_on_disk) {
            return;
        }
    }

    // These are needed for trait evaluation
    T ptvalues[TRAIT_NUM_MAX] = {0};
    T htvalues[TRAIT_NUM_MAX] = {0};
    T rvalue=0, vvalue=0, dvalue=0, zvalue=0, svalue=0, wvalue=1;

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

        for (int i = 0; i < a.spt.n; ++i)
            svalue += ptvalues[i];

        if (!svalue)
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

        zn += zvalue;
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

    // Thin disk requires density correction for inclination
    if (is_thin) {
        rvalue /= std::cos(incli);
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

    // Weight polar traits
    if (a.wpt.uids && a.wdata)
    {
        p_traits<wp_trait<T>>(
                ptvalues,
                a.wpt.n, a.wpt.uids,
                a.wpt.cvalues, a.wpt.ccounts,
                a.wpt.pvalues, a.wpt.pcounts,
                rnidx, a.rnodes, a.nrnodes,
                xn, yn, rn, theta);

        for (int i = 0; i < a.wpt.n; ++i)
            wvalue *= ptvalues[i];
    }

    // Apply opacity to the calculated density.
    T orvalue = rvalue;
    if (a.opacity)
    {
        // Apply the opacity of all the spaxels between the current spatial
        // position and the viewer. Do not include the current spatial position.
        for(int oz = z + 1; oz < a.spat_size[2]; ++oz)
        {
            const auto idx = index_3d_to_1d(
                    x, y, oz, a.spat_size[0], a.spat_size[1]);
            orvalue -= orvalue * a.opacity[idx];
        }
    }

    if (a.image) {
        gbkfit::gmodel_image_evaluate<AtomicAddFunT>(
                a.image, x, y, orvalue,
                a.spat_size[0]);
    }

    if (a.scube) {
        gbkfit::gmodel_scube_evaluate<AtomicAddFunT>(
                a.scube, x, y, orvalue, vvalue, dvalue,
                a.spat_size[0], a.spat_size[1],
                a.spec_size,
                a.spec_step,
                a.spec_zero);
    }

    const int idx = index_3d_to_1d(x, y, z, a.spat_size[0], a.spat_size[1]);

    if (a.wdata) {
        // TODO
        a.wdata[idx] += wvalue;
    }
    if (a.wdata_cmp) {
        a.wdata_cmp[idx] = wvalue;
    }
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
    if (a.vdata_cmp) {
        a.vdata_cmp[idx] = vvalue;
    }
    if (a.ddata_cmp) {
        a.ddata_cmp[idx] = dvalue;
    }
}

} // namespace gbkfit
