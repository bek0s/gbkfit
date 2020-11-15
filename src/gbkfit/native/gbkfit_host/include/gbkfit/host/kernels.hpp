#pragma once

#include <random>
#include <iostream>
#include <omp.h>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/disks.hpp>
#include "gbkfit/host/fftutils.hpp"

namespace gbkfit {

template<typename T>
struct RNG
{
    RNG(T a, T b, int seed)
        : dis(a, b)
        , gen(seed) {}

    T
    operator()(void) {
        return dis(gen);
    }

    std::uniform_real_distribution<T> dis;
    std::mt19937 gen;
};

} // namespace gbkfit

namespace gbkfit::host::kernels {

template<typename T> void
complex_multiply_and_scale(
        typename fftw3<T>::complex* arr1,
        typename fftw3<T>::complex* arr2,
        int n, float scale)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i)
    {
        typename fftw3<T>::complex a, b;

        a[0] = arr1[i][0];
        a[1] = arr1[i][1];
        b[0] = arr2[i % n][0];
        b[1] = arr2[i % n][1];

        arr1[i][0] = (a[0]*b[0]-a[1]*b[1])*scale;
        arr1[i][1] = (a[0]*b[1]+a[1]*b[0])*scale;
    }
}

template<typename T> void
dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    const T nfactor = T{1} / (scale_x * scale_y * scale_z);

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

        dst_cube[idx] = sum * nfactor;
    }
    }
    }
}



template<typename T> void
dcube_moments(
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* scube,
        T* mmaps, const int* orders, int norders)
{
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size_y; ++y)
    {
    for (int x = 0; x < size_x; ++x)
    {
        moments(x, y,
                size_x, size_y, size_z,
                step_x, step_y, step_z,
                zero_x, zero_y, zero_z,
                scube,
                mmaps, orders, norders);
    }
    }
}

template<typename T> void
evaluate_image(T* image, int x, int y, T rvalue, int spat_size_x)
{
    int idx = x + y * spat_size_x;
    #pragma omp atomic update
    image[idx] += rvalue;
}

template<typename T> void
evaluate_scube(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero)
{
    // Calculate a spectral range that encloses most of the flux.
    T zmin = vvalue - dvalue * 3;
    T zmax = vvalue + dvalue * 3;
    int zmin_idx = std::max<T>(std::rint(
            (zmin - spec_zero)/spec_step), 0);
    int zmax_idx = std::min<T>(std::rint(
            (zmax - spec_zero)/spec_step), spec_size - 1);

    // Evaluate the spectral line within the range specified above
    // Evaluating only within the range results in huge speed increase
    for (int z = zmin_idx; z <= zmax_idx; ++z)
    {
        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;
        T zvel = spec_zero + z * spec_step;
        T flux = rvalue * gauss_1d_pdf<T>(zvel, vvalue, dvalue);
        #pragma omp atomic update
        scube[idx] += flux;
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
        T* image, T* scube, T* rcube,
        T* rdata, T* vdata, T* ddata)
{
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < spat_size_y; ++y)
    {
    for(int x = 0; x < spat_size_x; ++x)
    {
    for(int z = 0; z < spat_size_z; ++z)
    {
        T bvalue, vvalue, dvalue;
        bool success = gbkfit::gmodel_smdisk_evaluate_pixel(
                bvalue, vvalue, dvalue,
                x, y, z,
                loose, tilted,
                nrnodes, rnodes,
                vsys,
                xpos, ypos,
                posa, incl,
                nrt,
                rpt_uids,
                rpt_cvalues, rpt_ccounts,
                rpt_pvalues, rpt_pcounts,
                rht_uids,
                rht_cvalues, rht_ccounts,
                rht_pvalues, rht_pcounts,
                nvt,
                vpt_uids,
                vpt_cvalues, vpt_ccounts,
                vpt_pvalues, vpt_pcounts,
                vht_uids,
                vht_cvalues, vht_ccounts,
                vht_pvalues, vht_pcounts,
                ndt,
                dpt_uids,
                dpt_cvalues, dpt_ccounts,
                dpt_pvalues, dpt_pcounts,
                dht_uids,
                dht_cvalues, dht_ccounts,
                dht_pvalues, dht_pcounts,
                nwt,
                wpt_uids,
                wpt_cvalues, wpt_ccounts,
                wpt_pvalues, wpt_pcounts,
                nst,
                spt_uids,
                spt_cvalues, spt_ccounts,
                spt_pvalues, spt_pcounts,
                spat_size_x, spat_size_y, spat_size_z,
                spat_step_x, spat_step_y, spat_step_z,
                spat_zero_x, spat_zero_y, spat_zero_z,
                spec_size,
                spec_step,
                spec_zero);

        if (!success) {
            continue;
        }

        if (image) {
            evaluate_image(
                    image, x, y, bvalue,
                    spat_size_x);
        }

        if (scube) {
            evaluate_scube(
                    scube, x, y, bvalue, vvalue, dvalue,
                    spat_size_x, spat_size_y,
                    spec_size,
                    spec_step,
                    spec_zero);
        }

        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        if (rcube) {
            rcube[idx] += bvalue;
        }

        if (rdata) {
            rdata[idx] = bvalue;
        }

        if (vdata) {
            vdata[idx] = vvalue;
        }

        if (ddata) {
            ddata[idx] = dvalue;
        }
    }
    }
    }
}

template<typename T> void
gmodel_mcdisk_evaluate(
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
        T spec_zero,
        T* image, T* scube, T* rcube,
        T* rdata, T* vdata, T* ddata)
{
    // Each thread needs to have each own random number generator.
    std::vector<RNG<T>> rngs;
    for(int i = 0; i < omp_get_num_procs(); ++i)
        rngs.push_back(RNG<T>(0, 1, 42));

    #pragma omp parallel for
    for(int ci = 0; ci < nclouds; ++ci)
    {
        int x, y, z;
        T bvalue, vvalue, dvalue;
        RNG<T>& rng = rngs[omp_get_thread_num()];
        bool success = gbkfit::gmodel_mcdisk_evaluate_cloud(
                x, y, z,
                bvalue, vvalue, dvalue,
                rng, ci,
                cflux, nclouds,
                ncloudscsum, ncloudscsum_len,
                hasordint,
                loose, tilted,
                nrnodes, rnodes,
                vsys,
                xpos, ypos,
                posa, incl,
                nrt,
                rpt_uids,
                rpt_cvalues, rpt_ccounts,
                rpt_pvalues, rpt_pcounts,
                rht_uids,
                rht_cvalues, rht_ccounts,
                rht_pvalues, rht_pcounts,
                nvt,
                vpt_uids,
                vpt_cvalues, vpt_ccounts,
                vpt_pvalues, vpt_pcounts,
                vht_uids,
                vht_cvalues, vht_ccounts,
                vht_pvalues, vht_pcounts,
                ndt,
                dpt_uids,
                dpt_cvalues, dpt_ccounts,
                dpt_pvalues, dpt_pcounts,
                dht_uids,
                dht_cvalues, dht_ccounts,
                dht_pvalues, dht_pcounts,
                nwt,
                wpt_uids,
                wpt_cvalues, wpt_ccounts,
                wpt_pvalues, wpt_pcounts,
                nst,
                spt_uids,
                spt_cvalues, spt_ccounts,
                spt_pvalues, spt_pcounts,
                spat_size_x, spat_size_y, spat_size_z,
                spat_step_x, spat_step_y, spat_step_z,
                spat_zero_x, spat_zero_y, spat_zero_z,
                spec_size,
                spec_step,
                spec_zero);

        if (!success) {
            continue;
        }

        if (image) {
            evaluate_image(
                    image, x, y, bvalue,
                    spat_size_x);
        }

        if (scube) {
            evaluate_scube(
                    scube, x, y, bvalue, vvalue, dvalue,
                    spat_size_x, spat_size_y,
                    spec_size,
                    spec_step,
                    spec_zero);
        }

        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        if (rcube) {
            #pragma omp atomic update
            rcube[idx] += bvalue;
        }

        if (rdata) {
            #pragma omp atomic update
            rdata[idx] += bvalue;
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

} // namespace gbkfit::host::kernels
