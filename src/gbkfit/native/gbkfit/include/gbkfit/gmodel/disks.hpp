#pragma once

#include "gbkfit/gmodel/traits.hpp"

namespace gbkfit {

constexpr int TRAIT_NUM_MAX = 4;

template<typename T> constexpr void
transform_cpos(T& x, T& y, T xpos, T ypos)
{
    x -= xpos;
    y -= ypos;
}

template<typename T> constexpr void
transform_posa(T& x, T& y, T posa)
{
    transform_lh_rotate_z(x, y, x, y, posa);
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

template<typename T> constexpr void
transform_incl_posa_cpos(T& x, T& y, T& z, T xposi, T yposi, T posai, T incli)
{
    transform_incl(y, z, incli);
    transform_posa(x, y, posai);
    transform_cpos(x, y, xposi, yposi);
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
#if 0
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
#if 0
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

template<typename T> constexpr bool
gmodel_mcdisk_evaluate_cloud(
        int& x, int& y, int& z,
        T& bvalue, T& vvalue, T& dvalue,
        RNG<T>& rng, int ci,
        T cflux, int nclouds,
        const int* ncloudscsum, int ncloudscsum_len,
        const bool* hasordint,
        bool loose, bool tilted,
        int nrnodes, const T* rnodes,
        const T* vsys,
        const T* xpos, const T* ypos,
        const T* posa, const T* incl,
        int nrt,
        const int* rpt_uids,
        const T* rpt_cvalues, const int* rpt_ccounts,
        const T* rpt_pvalues, const int* rpt_pcounts,
        const int* rht_uids,
        const T* rht_cvalues, const int* rht_ccounts,
        const T* rht_pvalues, const int* rht_pcounts,
        int nvt,
        const int* vpt_uids,
        const T* vpt_cvalues, const int* vpt_ccounts,
        const T* vpt_pvalues, const int* vpt_pcounts,
        const int* vht_uids,
        const T* vht_cvalues, const int* vht_ccounts,
        const T* vht_pvalues, const int* vht_pcounts,
        int ndt,
        const int* dpt_uids,
        const T* dpt_cvalues, const int* dpt_ccounts,
        const T* dpt_pvalues, const int* dpt_pcounts,
        const int* dht_uids,
        const T* dht_cvalues, const int* dht_ccounts,
        const T* dht_pvalues, const int* dht_pcounts,
        int nwt,
        const int* wpt_uids,
        const T* wpt_cvalues, const int* wpt_ccounts,
        const T* wpt_pvalues, const int* wpt_pcounts,
        int nst,
        const int* spt_uids,
        const T* spt_cvalues, const int* spt_ccounts,
        const T* spt_pvalues, const int* spt_pcounts,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero)
{
    // This is a placeholder in case we decide to explicitly
    // add a Monte Carlo based thin disk in the future.
    bool is_thin = false;

    int rnidx = 0, tidx=0;
    const T* rpt_cptr = rpt_cvalues;
    const T* rpt_pptr = rpt_pvalues;
    const T* rht_cptr = rht_cvalues;
    const T* rht_pptr = rht_pvalues;
    T xd=0, yd=0, zd=0, rd=0, theta=0, sign=0;
    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T ptvalues[] = {0, 0, 0, 0};
    T htvalues[] = {0, 0, 0, 0};
    T wvalue = 0;
    T svalue = 0;
    bvalue = cflux;
    vvalue = 0;
    dvalue = 0;

    // Find which comulative sum the cloud belongs to.
    while(ci >= ncloudscsum[rnidx]) {
        rnidx++;
    }

    // Find which trait and subring the cloud belongs to.
    for(tidx = 0; tidx < nrt; ++tidx)
    {
        int size = hasordint[tidx] ? 1 : nrnodes - 2;
        if (rnidx < size)
            break;
        rnidx -= size;
        rpt_cptr += rpt_ccounts[tidx];
        rpt_pptr += rpt_pcounts[tidx];
        rht_cptr += rht_ccounts[tidx];
        rht_pptr += rht_pcounts[tidx];
    }

    // Density polar trait
    // The first and last radial nodes must be ignored.
    rp_trait_rnd<T>(
            sign, rd, theta, rng,
            rpt_uids[tidx], rpt_cptr, rpt_pptr,
            rnidx, &rnodes[1], nrnodes - 2);

    if (sign < 0) {
        bvalue = -bvalue;
        //std::cout << bvalue << std::endl;
    }

    // Calculate cartesian coordinates on disk plane.
    xd = rd * std::cos(theta);
    yd = rd * std::sin(theta);

    // Recalculate radial node index.
    // This is done in order to account for:
    //  - rptraits with ordinary integral (no subrings).
    //  - pixels in the first and last half subrings.
    if (!disk_info(rnidx, rd, nrnodes, rnodes)) {
        return false;
    }

    // =====================================================================

    // Selection traits
    if (spt_uids)
    {
        p_traits(
                sp_trait<T>,
                ptvalues,
                nst, spt_uids,
                spt_cvalues, spt_ccounts,
                spt_pvalues, spt_pcounts,
                rnidx, rnodes, nrnodes,
                xd, yd, rd, theta);

        for (int i = 0; i < nst; ++i)
            svalue += ptvalues[i];

        if (!svalue)
            return false;
    }

    // Density height trait
    rh_trait_rnd<T>(zd, rng, rht_uids[tidx], rht_cptr, rht_pptr);

    // Warp traits
    if (wpt_uids)
    {
        p_traits(
                wp_trait<T>,
                ptvalues,
                nwt, wpt_uids,
                wpt_cvalues, wpt_ccounts,
                wpt_pvalues, wpt_pcounts,
                rnidx, rnodes, nrnodes,
                xd, yd, rd, theta);

        for (int i = 0; i < nwt; ++i)
            wvalue += ptvalues[i];

        zd += wvalue;
    }

    vsysi = loose ? lerp(rd, rnidx, rnodes, vsys) : (vsys ? vsys[0] : 0);
    xposi = loose ? lerp(rd, rnidx, rnodes, xpos) : xpos[0];
    yposi = loose ? lerp(rd, rnidx, rnodes, ypos) : ypos[0];
    posai = tilted ? lerp(rd, rnidx, rnodes, posa) : posa[0];
    incli = tilted ? lerp(rd, rnidx, rnodes, incl) : incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;

    T xn = xd, yn = yd, zn = zd;
    transform_incl_posa_cpos(xn, yn, zn, xposi, yposi, posai, incli);

    x = std::rint(xn - spat_zero_x);
    y = std::rint(yn - spat_zero_y);
    z = std::rint(zn - spat_zero_z);

    // Discard pixels outside the image/cube
    if (x < 0 || x >= spat_size_x ||
        y < 0 || y >= spat_size_y ||
        z < 0 || z >= spat_size_z) {
        return false;
    }

    // Velocity traits
    if (vpt_uids)
    {
        p_traits(
                vp_trait<T>,
                ptvalues,
                nvt, vpt_uids,
                vpt_cvalues, vpt_ccounts,
                vpt_pvalues, vpt_pcounts,
                rnidx, rnodes, nrnodes,
                xd, yd, rd, theta, incli);
    }
    if (vht_uids)
    {
        h_traits(
                vh_trait<T>,
                htvalues,
                nvt, vht_uids,
                vht_cvalues, vht_ccounts,
                vht_pvalues, vht_pcounts,
                std::abs(zd));
    }
    for (int i = 0; i < nvt; ++i)
        vvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Dispersion traits
    if (dpt_uids)
    {
        p_traits(
                dp_trait<T>,
                ptvalues,
                ndt, dpt_uids,
                dpt_cvalues, dpt_ccounts,
                dpt_pvalues, dpt_pcounts,
                rnidx, rnodes, nrnodes,
                xd, yd, rd, theta);
    }
    if (dht_uids)
    {
        h_traits(
                dh_trait<T>,
                htvalues,
                ndt, dht_uids,
                dht_cvalues, dht_ccounts,
                dht_pvalues, dht_pcounts,
                std::abs(zd));
    }
    for (int i = 0; i < ndt; ++i)
        dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Ensure positive dispersion
    dvalue = std::abs(dvalue);

    return true;
}


template<typename T> constexpr bool
gmodel_smdisk_evaluate_pixel(
        T& bvalue, T& vvalue, T& dvalue,
        int x, int y, int z,
        bool loose, bool tilted,
        int nrnodes, const T* rnodes,
        const T* vsys,
        const T* xpos, const T* ypos,
        const T* posa, const T* incl,
        int nrt,
        const int* rpt_uids,
        const T* rpt_cvalues, const int* rpt_ccounts,
        const T* rpt_pvalues, const int* rpt_pcounts,
        const int* rht_uids,
        const T* rht_cvalues, const int* rht_ccounts,
        const T* rht_pvalues, const int* rht_pcounts,
        int nvt,
        const int* vpt_uids,
        const T* vpt_cvalues, const int* vpt_ccounts,
        const T* vpt_pvalues, const int* vpt_pcounts,
        const int* vht_uids,
        const T* vht_cvalues, const int* vht_ccounts,
        const T* vht_pvalues, const int* vht_pcounts,
        int ndt,
        const int* dpt_uids,
        const T* dpt_cvalues, const int* dpt_ccounts,
        const T* dpt_pvalues, const int* dpt_pcounts,
        const int* dht_uids,
        const T* dht_cvalues, const int* dht_ccounts,
        const T* dht_pvalues, const int* dht_pcounts,
        int nwt,
        const int* wpt_uids,
        const T* wpt_cvalues, const int* wpt_ccounts,
        const T* wpt_pvalues, const int* wpt_pcounts,
        int nst,
        const int* spt_uids,
        const T* spt_cvalues, const int* spt_ccounts,
        const T* spt_pvalues, const int* spt_pcounts,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero)
{
    bool is_thin = rht_uids == nullptr;

    T vsysi=0, xposi=0, yposi=0, posai=0, incli=0;
    T xn = x, yn = y, zn = z, rn=0, theta=0;
    int rnidx = -1;

    xn = spat_zero_x + x * spat_step_x;
    yn = spat_zero_y + y * spat_step_y;
    zn = spat_zero_z + z * spat_step_z;

    if (loose || tilted)
    {
        bool success = false;
        if (is_thin)
            success = ring_info(
                    rnidx, rn, xn, yn, loose, tilted,
                    nrnodes, rnodes, xpos, ypos, posa, incl);
        else
            success = ring_info(
                    rnidx, rn, xn, yn, zn, loose, tilted,
                    nrnodes, rnodes, xpos, ypos, posa, incl);
        if (!success)
            return false;
    }

    vsysi = loose ? lerp(rn, rnidx, rnodes, xpos) : vsys[0];
    xposi = loose ? lerp(rn, rnidx, rnodes, xpos) : xpos[0];
    yposi = loose ? lerp(rn, rnidx, rnodes, ypos) : ypos[0];
    posai = tilted ? lerp(rn, rnidx, rnodes, posa) : posa[0];
    incli = tilted ? lerp(rn, rnidx, rnodes, incl) : incl[0];
    posai *= DEG_TO_RAD<T>;
    incli *= DEG_TO_RAD<T>;


    if (is_thin)
        transform_cpos_posa_incl(xn, yn, xposi, yposi, posai, incli);
    else
        transform_cpos_posa_incl(xn, yn, zn, xposi, yposi, posai, incli);

    theta = std::atan2(yn, xn);

    if (!(loose || tilted)) {
        rn = std::sqrt(xn * xn + yn * yn);
        if (!disk_info(rnidx, rn, nrnodes, rnodes))
            return false;
    }

    // These are needed for trait evaluation
//  T ptvalues[TRAIT_NUM_MAX];
//  T htvalues[TRAIT_NUM_MAX];
    T ptvalues[] = {0, 0, 0, 0};
    T htvalues[] = {0, 0, 0, 0};
    T wvalue = 0;
    T svalue = 0;
    bvalue = 0;
    vvalue = vsysi;
    dvalue = 0;

    // Selection traits
    if (spt_uids)
    {
        p_traits(
                sp_trait<T>,
                ptvalues,
                nst, spt_uids,
                spt_cvalues, spt_ccounts,
                spt_pvalues, spt_pcounts,
                rnidx, rnodes, nrnodes,
                xn, yn, rn, theta);

        for (int i = 0; i < nst; ++i)
            svalue += ptvalues[i];

        if (!svalue)
            return false;
    }

    // Warp traits
    if (wpt_uids)
    {
        p_traits(
                wp_trait<T>,
                ptvalues,
                nwt, wpt_uids,
                wpt_cvalues, wpt_ccounts,
                wpt_pvalues, wpt_pcounts,
                rnidx, rnodes, nrnodes,
                xn, yn, rn, theta);

        for (int i = 0; i < nwt; ++i)
            wvalue += ptvalues[i];

        zn += wvalue;
    }

    // Brightness traits
    if (rpt_uids)
    {
        p_traits(
                rp_trait<T>,
                ptvalues,
                nrt, rpt_uids,
                rpt_cvalues, rpt_ccounts,
                rpt_pvalues, rpt_pcounts,
                rnidx, rnodes, nrnodes,
                xn, yn, rn, theta);
    }
    if (rht_uids)
    {
        h_traits(
                rh_trait<T>,
                htvalues,
                nrt, rht_uids,
                rht_cvalues, rht_ccounts,
                rht_pvalues, rht_pcounts,
                std::abs(zn));
    }
    for (int i = 0; i < nrt; ++i)
        bvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i] * spat_step_z);

    // Velocity traits
    if (vpt_uids)
    {
        p_traits(
                vp_trait<T>,
                ptvalues,
                nvt, vpt_uids,
                vpt_cvalues, vpt_ccounts,
                vpt_pvalues, vpt_pcounts,
                rnidx, rnodes, nrnodes,
                xn, yn, rn, theta, incli);
    }
    if (vht_uids)
    {
        h_traits(
                vh_trait<T>,
                htvalues,
                nvt, vht_uids,
                vht_cvalues, vht_ccounts,
                vht_pvalues, vht_pcounts,
                std::abs(zn));
    }
    for (int i = 0; i < nvt; ++i)
        vvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Dispersion traits
    if (dpt_uids)
    {
        p_traits(
                dp_trait<T>,
                ptvalues,
                ndt, dpt_uids,
                dpt_cvalues, dpt_ccounts,
                dpt_pvalues, dpt_pcounts,
                rnidx, rnodes, nrnodes,
                xn, yn, rn, theta);
    }
    if (dht_uids)
    {
        h_traits(
                dh_trait<T>,
                htvalues,
                ndt, dht_uids,
                dht_cvalues, dht_ccounts,
                dht_pvalues, dht_pcounts,
                std::abs(zn));
    }
    for (int i = 0; i < ndt; ++i)
        dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

    // Ensure positive dispersion
    dvalue = std::abs(dvalue);

    return true;
}

} // namespace gbkfit
