#pragma once

#include <omp.h>

#include "kernels_misc.hpp"
#include "kernels_traits.hpp"

namespace gbkfit::openmp::kernels {

template<typename T> void
dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    #pragma omp parallel for collapse(3)
    for(int z = 0; z < dst_size_z; ++z)
    {
    for(int y = 0; y < dst_size_y; ++y)
    {
    for(int x = 0; x < dst_size_x; ++x)
    {
        // Src cube 3d index
        int nx = offset_x + x * scale_x;
        int ny = offset_y + y * scale_y;
        int nz = offset_z + z * scale_z;

        // Calculate average value under the current position
        T sum = 0;
        #pragma omp simd collapse(3)
        for(int dsz = 0; dsz < scale_z; ++dsz)
        {
        for(int dsy = 0; dsy < scale_y; ++dsy)
        {
        for(int dsx = 0; dsx < scale_x; ++dsx)
        {
            int idx = (nx + dsx)
                    + (ny + dsy) * src_size_x
                    + (nz + dsz) * src_size_x * src_size_y;

            sum += src_cube[idx];
        }
        }
        }

        // Dst cube 1d index
        int idx = x
                + y * dst_size_x
                + z * dst_size_x * dst_size_y;

        dst_cube[idx] = sum / (scale_x * scale_y * scale_z);
    }
    }
    }
}

template<typename T> void
dcube_moments(
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero,
        T nanval,
        const T* cube,
        int mcount, const int* morders, T* mmaps)
{
    int max_morder = morders[mcount - 1];

//  #pragma omp parallel for collapse(2)
    for (int y = 0; y < spat_size_y; ++y)
    {
    for (int x = 0; x < spat_size_x; ++x)
    {
        int m = 0;
        T m0, m1, m2, m0_sum=0, m1_sum=0, m2_sum=0;

        // Moment 0
        for (int z = 0; z < spec_size; ++z)
        {
            int idx = x
                    + y * spat_size_x
                    + z * spat_size_x * spat_size_y;

            T flx = std::max(T{0}, cube[idx]);
            m0_sum += flx;
        }
        m0 = m0_sum;

        if (morders[m] == 0)
        {
            int idx = x
                    + y * spat_size_x
                    + m * spat_size_x * spat_size_y;

            mmaps[idx] = m0;
            m++;
        }

        if (max_morder == 0)
            continue;

        // Moment 1
        for (int z = 0; z < spec_size; ++z)
        {
            int idx = x
                    + y * spat_size_x
                    + z*spat_size_x*spat_size_y;

            T flx = std::max(T{0}, cube[idx]);
            T vel = spec_zero + z * spec_step;
            m1_sum += flx * vel;
        }
        m1 = m0 > 0 ? m1_sum / m0 : nanval;

        if (morders[m] == 1)
        {
            int idx = x
                    + y * spat_size_x
                    + m * spat_size_x * spat_size_y;

            mmaps[idx] = m1;
            m++;
        }

        if (max_morder == 1)
            continue;

        // Moment 2
        for (int z = 0; z < spec_size; ++z)
        {
            int idx = x
                    + y * spat_size_x
                    + z * spat_size_x*spat_size_y;

            T flx = std::max(T{0}, cube[idx]);
            T vel = spec_zero + z * spec_step;
            m2_sum += flx * (vel - m1) * (vel - m1);
        }
        m2 = m0 > 0 ? std::sqrt(m2_sum / m0) : nanval;

        if (morders[m] == 2)
        {
            int idx = x
                    + y * spat_size_x
                    + m * spat_size_x * spat_size_y;

            mmaps[idx] = m2;
            m++;
        }

        if (max_morder == 2)
            continue;

        // Higher order moments
        for(; m < mcount; ++m)
        {
            T mn, mn_sum = 0;
            for (int z = 0; z < spec_size; ++z)
            {
                int idx = x
                        + y * spat_size_x
                        + z * spat_size_x*spat_size_y;

                T flx = std::max(T{0}, cube[idx]);
                T vel = spec_zero + z * spec_step;
                mn_sum += flx * std::pow(vel - m1, morders[m]);
            }
            mn = m0 > 0 ? mn_sum / m0 : nanval;

            int idx = x
                    + y * spat_size_x
                    + m * spat_size_x * spat_size_y;

            mmaps[idx] = mn;
        }
    }
    }
}

template<typename T> void
gmodel_smdisk_evaluate(
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
        T spec_zero,
        T* image, T* scube, T* bcube,
        T* bdata, T* vdata, T* ddata)
{
    bool is_thin = rht_uids == nullptr;

    #pragma omp parallel for collapse(2)
    for(int y = 0; y < spat_size_y; ++y)
    {
    for(int x = 0; x < spat_size_x; ++x)
    {
    for(int z = 0; z < spat_size_z; ++z)
    {
        T xposi, yposi, posai, incli;
        T xn = x, yn = y, zn = z, rn, theta;
        int rnidx = -1;

        xn = spat_zero_x + x * spat_step_x;
        yn = spat_zero_y + y * spat_step_y;
        zn = spat_zero_z + z * spat_step_z;

        if (loose || tilted)
        {
            bool success;
            if (is_thin)
                success = ring_info(
                        rnidx, rn, xn, yn, loose, tilted,
                        nrnodes, rnodes, xpos, ypos, posa, incl);
            else
                success = ring_info(
                        rnidx, rn, xn, yn, zn, loose, tilted,
                        nrnodes, rnodes, xpos, ypos, posa, incl);
            if (!success)
                continue;
        }

        xposi = loose ? interp_linear(rn, rnidx, rnodes, xpos) : xpos[0];
        yposi = loose ? interp_linear(rn, rnidx, rnodes, ypos) : ypos[0];
        posai = tilted ? interp_linear(rn, rnidx, rnodes, posa) : posa[0];
        incli = tilted ? interp_linear(rn, rnidx, rnodes, incl) : incl[0];
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
                continue;
        }

        // These are needed for trait evaluation
        T ptvalues[TRAIT_NUM_MAX];
        T htvalues[TRAIT_NUM_MAX];
        T bvalue = 0;
        T vvalue = 0;
        T dvalue = 0;
        T wvalue = 0;
        T svalue = 0;

        // Selection traits
        if (spt_uids)
        {
            p_traits<sp_trait<T>>(
                    ptvalues,
                    nst, spt_uids,
                    spt_cvalues, spt_ccounts,
                    spt_pvalues, spt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xn, yn, rn, theta);

            for (int i = 0; i < nst; ++i)
                svalue += ptvalues[i];

            if (!svalue)
                continue;
        }

        // Warp traits
        if (wpt_uids)
        {
            p_traits<wp_trait<T>>(
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
            p_traits<rp_trait<T>>(
                    ptvalues,
                    nrt, rpt_uids,
                    rpt_cvalues, rpt_ccounts,
                    rpt_pvalues, rpt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xn, yn, rn, theta);
        }
        if (rht_uids)
        {
            h_traits<rh_trait<T>>(
                    htvalues,
                    nrt, rht_uids,
                    rht_cvalues, rht_ccounts,
                    rht_pvalues, rht_pcounts,
                    std::abs(zn));
        }
        for (int i = 0; i < nrt; ++i)
            bvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i] * spat_step_z);

        // TODO: How to deal with negative bvalue?
        if (bvalue < 0)
            continue;

        // Velocity traits
        if (vpt_uids)
        {
            p_traits<vp_trait<T>>(
                    ptvalues,
                    nvt, vpt_uids,
                    vpt_cvalues, vpt_ccounts,
                    vpt_pvalues, vpt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xn, yn, rn, theta, incli);
        }
        if (vht_uids)
        {
            h_traits<vh_trait<T>>(
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
            p_traits<dp_trait<T>>(
                    ptvalues,
                    ndt, dpt_uids,
                    dpt_cvalues, dpt_ccounts,
                    dpt_pvalues, dpt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xn, yn, rn, theta);
        }
        if (dht_uids)
        {
            h_traits<dh_trait<T>>(
                    htvalues,
                    ndt, dht_uids,
                    dht_cvalues, dht_ccounts,
                    dht_pvalues, dht_pcounts,
                    std::abs(zn));
        }
        for (int i = 0; i < ndt; ++i)
            dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

        // TODO: How to deal with negative dvalue?
        if (dvalue < 0)
            continue;

        if (image)
            evaluate_image(
                    image, x, y, bvalue,
                    spat_size_x);

        if (scube)
            evaluate_scube(
                    scube, x, y, bvalue, vvalue, dvalue,
                    spat_size_x, spat_size_y,
                    spec_size,
                    spec_step,
                    spec_zero);

        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        if (bcube)
            bcube[idx] += bvalue;

        if (bdata)
            bdata[idx] = bvalue;

        if (vdata)
            vdata[idx] = vvalue;

        if (ddata)
            ddata[idx] = dvalue;
    }
    }
    }
}

template<typename T> void
gmodel_mcdisk_evaluate(
        T cflux, int nclouds, const int* ncloudspt,
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
        T spec_zero,
        T* image, T* scube, T* bcube,
        T* bdata, T* vdata, T* ddata)
{

    RNG<T> rngs[] = {RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1), RNG<T>(0, 1)};

    bool is_thin = false;

    //#pragma omp parallel for
    for(int ci = 0; ci < nclouds; ++ci)
    {

        RNG<T>& rng = rngs[omp_get_thread_num()];
        T ptvalues[TRAIT_NUM_MAX];
        T htvalues[TRAIT_NUM_MAX];
        T bvalue = cflux;
        T vvalue = 0;
        T dvalue = 0;
        T wvalue = 0;
        T svalue = 0;
        T vsysi, xposi, yposi, posai, incli;
        T xd, yd, zd, rd, theta;
        int rnidx = -1;

        // Calculate the density trait index of the current cloud
        int tidx = 0;
        int csum = 0;
        const T* rpt_cptr = rpt_cvalues;
        const T* rpt_pptr = rpt_pvalues;
        const T* rht_cptr = rht_cvalues;
        const T* rht_pptr = rht_pvalues;
        while((csum += ncloudspt[tidx]) <= ci)
        {
            rpt_cptr += rpt_ccounts[tidx];
            rpt_pptr += rpt_pcounts[tidx];
            rht_cptr += rht_ccounts[tidx];
            rht_pptr += rht_pcounts[tidx];
            tidx++;
        }

        // Density polar trait
        T sign;
        rp_trait_rnd<T>(
                sign, rd, theta, rng,
                rpt_uids[tidx], rpt_cptr, rpt_pptr, rnidx, rnodes, nrnodes);

        if (!disk_info(rnidx, rd, nrnodes, rnodes))
            continue;

        // Calculate cartesian coordinates on disk plane
        xd = rd * std::cos(theta);
        yd = rd * std::sin(theta);

        // =====================================================================

        // Selection traits
        if (spt_uids)
        {
            p_traits<sp_trait<T>>(
                    ptvalues,
                    nst, spt_uids,
                    spt_cvalues, spt_ccounts,
                    spt_pvalues, spt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xd, yd, rd, theta);

            for (int i = 0; i < nst; ++i)
                svalue += ptvalues[i];

            if (!svalue)
                continue;
        }

        // Density height trait
        // Only if needed (i.e., after passing the selection traits)
        rh_trait_rnd<T>(zd, rng, rht_uids[tidx], rht_cptr, rht_pptr);
    //  zd = 0;
    //  zd = 0;

        // Warp traits
        if (wpt_uids)
        {
            p_traits<wp_trait<T>>(
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

        vsysi = loose ? interp_linear(rd, rnidx, rnodes, vsys) : (vsys ? vsys[0] : 0);
        xposi = loose ? interp_linear(rd, rnidx, rnodes, xpos) : xpos[0];
        yposi = loose ? interp_linear(rd, rnidx, rnodes, ypos) : ypos[0];
        posai = tilted ? interp_linear(rd, rnidx, rnodes, posa) : posa[0];
        incli = tilted ? interp_linear(rd, rnidx, rnodes, incl) : incl[0];
        posai *= DEG_TO_RAD<T>;
        incli *= DEG_TO_RAD<T>;

        T xn = xd, yn = yd, zn = zd;

    //  std::cout << "rd: " << rd << std::endl;
    //  std::cout << "theta: " << theta << std::endl;
    //  std::cout << "1: xn, yn, zn: " << xn << ", " << yn << ", " << zn << std::endl;

        transform_incl_posa_cpos(xn, yn, zn, xposi, yposi, posai, incli);

    //  std::cout << xposi << " " << yposi << " " << posai << " " << incli << std::endl;
    //  std::cout << "2: xn, yn, zn: " << xn << ", " << yn << ", " << zn << std::endl;

        /*
        int x = std::rint(xn - spat_zero_x - 0.5);
        int y = std::rint(yn - spat_zero_y - 0.5);
        int z = std::rint(zn - spat_zero_z - 0.5);
        */

        int x = std::rint(xn - spat_zero_x - 0.5);
        int y = std::rint(yn - spat_zero_y - 0.5);
        int z = std::rint(zn - spat_zero_z - 0.5);

        //std::cout << "0: " << xn << " " << yn << " " << zn << std::endl;
        //std::cout << "3: " << x << " " << y << " " << z << std::endl;

        if (x < 0 || x >= spat_size_x ||
            y < 0 || y >= spat_size_y ||
            z < 0 || z >= spat_size_z)
            continue;

        //std::cout << "4: " << x << " " << y << " " << z << std::endl;

        // Velocity traits
        if (vpt_uids)
        {
            p_traits<vp_trait<T>>(
                    ptvalues,
                    nvt, vpt_uids,
                    vpt_cvalues, vpt_ccounts,
                    vpt_pvalues, vpt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xd, yd, rd, theta, incli);
        }
        if (vht_uids)
        {
            h_traits<vh_trait<T>>(
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
            p_traits<dp_trait<T>>(
                    ptvalues,
                    ndt, dpt_uids,
                    dpt_cvalues, dpt_ccounts,
                    dpt_pvalues, dpt_pcounts,
                    rnidx, rnodes, nrnodes,
                    xd, yd, rd, theta);
        }
        if (dht_uids)
        {
            h_traits<dh_trait<T>>(
                    htvalues,
                    ndt, dht_uids,
                    dht_cvalues, dht_ccounts,
                    dht_pvalues, dht_pcounts,
                    std::abs(zd));
        }
        for (int i = 0; i < ndt; ++i)
            dvalue += ptvalues[i] * (is_thin ? 1 : htvalues[i]);

        // TODO: How to deal with negative dvalue?
        if (dvalue < 0)
            continue;

        if (image)
            evaluate_image(
                    image, x, y, bvalue,
                    spat_size_x);

        if (scube)
            evaluate_scube(
                    scube, x, y, bvalue, vvalue, dvalue,
                    spat_size_x, spat_size_y,
                    spec_size,
                    spec_step,
                    spec_zero);

        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        if (bcube) {
            #pragma omp atomic update
            bcube[idx] += bvalue;
        }

        if (bdata) {
            #pragma omp atomic update
            bdata[idx] += bvalue;
        }

        if (vdata) {
            #pragma omp atomic write
            vdata[idx] = vvalue;
        }

        if (ddata) {
            #pragma omp atomic write
            ddata[idx] = dvalue;
        }
    }
}

} // namespace gbkfit::openmp::kernels
