#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

#include <omp.h>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>

#include "gbkfit/host/fftutils.hpp"
#include "gbkfit/host/random.hpp"

namespace gbkfit::host::kernels {

template<typename T> inline constexpr void
atomic_add(T* addr, T val)
{
    #pragma omp atomic update
    *addr += val;
}

template<typename T> inline constexpr void
atomic_set(T* addr, T val)
{
    #pragma omp atomic write
    *addr = val;
}

template<typename T> void
math_complex_multiply_and_scale(
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
dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_dcube, T* dst_dcube)
{
    // Parallelization: per 3d position in the dst dcube
    #pragma omp parallel for collapse(3)
    for(int z = 0; z < dst_size_z; ++z) {
    for(int y = 0; y < dst_size_y; ++y) {
    for(int x = 0; x < dst_size_x; ++x) {

    gbkfit::dmodel_dcube_downscale(
            x, y, z,
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_dcube, dst_dcube);

    }
    }
    }
}

template<typename T> void
dmodel_dcube_mask(
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    // Parallelization: per 3d position in the dcube
    #pragma omp parallel for collapse(3)
    for(int z = 0; z < size_z; ++z) {
    for(int y = 0; y < size_y; ++y) {
    for(int x = 0; x < size_x; ++x) {

    gbkfit::dmodel_dcube_mask(
            x, y, z,
            cutoff, apply,
            size_x, size_y, size_z,
            dcube_d, dcube_m, dcube_w);

    }
    }
    }
}

template<typename T> void
dmodel_moments(
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff,
        int norders,
        const int* orders,
        T* mmaps_d,
        T* mmaps_m,
        T* mmaps_w)
{
    // Parallelization: per 2d spatial position
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size_y; ++y) {
    for (int x = 0; x < size_x; ++x) {

    gbkfit::dmodel_mmaps_moments(
            x, y,
            size_x, size_y, size_z,
            step_x, step_y, step_z,
            zero_x, zero_y, zero_z,
            dcube_d, dcube_w,
            cutoff, norders, orders,
            mmaps_d, mmaps_w, mmaps_m);

    }
    }
}

template<typename T> void
gmodel_convolve(
        int data1_size_x, int data1_size_y, int data1_size_z,
        int data2_size_x, int data2_size_y, int data2_size_z,
        const T* data1, const T* data2, T* result)
{
    // Parallelization: per 3d position in cube
    #pragma omp parallel for collapse(3)
    for(int z1 = 0; z1 < data1_size_z; ++z1) {
    for(int y1 = 0; y1 < data1_size_y; ++y1) {
    for(int x1 = 0; x1 < data1_size_x; ++x1) {

    gbkfit::gmodel_convolve(
            x1, y1, z1,
            data1_size_x, data1_size_y, data1_size_z,
            data2_size_x, data2_size_y, data2_size_z,
            data1, data2, result);

    }
    }
    }
}

template<typename T> void
gmodel_wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_cube,
        T* spec_cube)
{
    // Parallelization: per 2d spatial position
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < spat_size_y; ++y) {
    for(int x = 0; x < spat_size_x; ++x) {

    gbkfit::gmodel_wcube_pixel(
            x, y,
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            spat_cube,
            spec_cube);

    }
    }
}

template<typename T> void
gmodel_smdisk_evaluate(
        bool loose, bool tilted,
        int nrnodes, const T* rnodes,
        T tauto,
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
        int nzt,
        const int* zpt_uids,
        const T* zpt_cvalues, const int* zpt_ccounts,
        const T* zpt_pvalues, const int* zpt_pcounts,
        int nst,
        const int* spt_uids,
        const T* spt_cvalues, const int* spt_ccounts,
        const T* spt_pvalues, const int* spt_pcounts,
        int nwt,
        const int* wpt_uids,
        const T* wpt_cvalues, const int* wpt_ccounts,
        const T* wpt_pvalues, const int* wpt_pcounts,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        T* image, T* scube, T* tdata, T* wdata,
        T* extra_rdata_tot,
        T* extra_rdata_cmp,
        T* extra_vdata_cmp,
        T* extra_ddata_cmp)
{
    // Parallelization: per 3d spatial position
    // TODO: Implement ray marching and revise parallelization scheme
    #pragma omp parallel for collapse(3)
    for(int y = 0; y < spat_size_y; ++y) {
    for(int x = 0; x < spat_size_x; ++x) {
    for(int z = 0; z < spat_size_z; ++z) {

    gbkfit::gmodel_smdisk_evaluate_spaxel<atomic_add<T>>(
            x, y, z,
            loose, tilted,
            nrnodes, rnodes,
            tauto,
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
            nzt,
            zpt_uids,
            zpt_cvalues, zpt_ccounts,
            zpt_pvalues, zpt_pcounts,
            nst,
            spt_uids,
            spt_cvalues, spt_ccounts,
            spt_pvalues, spt_pcounts,
            nwt,
            wpt_uids,
            wpt_cvalues, wpt_ccounts,
            wpt_pvalues, wpt_pcounts,
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            image, scube, tdata, wdata,
            extra_rdata_tot,
            extra_rdata_cmp,
            extra_vdata_cmp,
            extra_ddata_cmp);
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
        T tauto,
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
        int nzt,
        const int* zpt_uids,
        const T* zpt_cvalues, const int* zpt_ccounts,
        const T* zpt_pvalues, const int* zpt_pcounts,
        int nst,
        const int* spt_uids,
        const T* spt_cvalues, const int* spt_ccounts,
        const T* spt_pvalues, const int* spt_pcounts,
        int nwt,
        const int* wpt_uids,
        const T* wpt_cvalues, const int* wpt_ccounts,
        const T* wpt_pvalues, const int* wpt_pcounts,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        T* image, T* scube, T* tdata, T* wdata,
        T* extra_rdata_tot,
        T* extra_rdata_cmp,
        T* extra_vdata_cmp,
        T* extra_ddata_cmp)
{
    // Each thread needs to have each own random number generator.
    std::vector<RNG<T>> rngs;
    for(int i = 0; i < omp_get_num_procs(); ++i)
        rngs.push_back(RNG<T>(0, 1, 42));

    // Parallelization: per cloud
    #pragma omp parallel for
    for(int ci = 0; ci < nclouds; ++ci)
    {

    RNG<T>& rng = rngs[omp_get_thread_num()];
    gbkfit::gmodel_mcdisk_evaluate_cloud<atomic_set<T>, atomic_add<T>>(
            rng, ci,
            cflux, nclouds,
            ncloudscsum, ncloudscsum_len,
            hasordint,
            loose, tilted,
            nrnodes, rnodes,
            tauto,
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
            nzt,
            zpt_uids,
            zpt_cvalues, zpt_ccounts,
            zpt_pvalues, zpt_pcounts,
            nst,
            spt_uids,
            spt_cvalues, spt_ccounts,
            spt_pvalues, spt_pcounts,
            nwt,
            wpt_uids,
            wpt_cvalues, wpt_ccounts,
            wpt_pvalues, wpt_pcounts,
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            image, scube, tdata, wdata,
            extra_rdata_tot,
            extra_rdata_cmp,
            extra_vdata_cmp,
            extra_ddata_cmp);

    }
}

template<typename T> void
objective_count_pixels(
        const T* data1, const T* data2, int size, T epsilon, int* counts)
{
    #pragma omp parallel
    {
        int count_data1 = 0;
        int count_data2 = 0;
        int count_both = 0;

        #pragma omp for nowait
        for(int i = 0; i < size; ++i)
        {
            const bool has_data1 = std::abs(data1[i]) > epsilon;
            const bool has_data2 = std::abs(data2[i]) > epsilon;
            count_data1 += has_data1 && !has_data2;
            count_data2 += !has_data1 && has_data2;
            count_both += has_data1 && has_data2;
        }

        if (count_data1) {
            #pragma omp atomic update
            counts[0] += count_data1;
        }
        if (count_data2) {
            #pragma omp atomic update
            counts[1] += count_data2;
        }
        if (count_both) {
            #pragma omp atomic update
            counts[2] += count_both;
        }
    }
}

} // namespace gbkfit::host::kernels
