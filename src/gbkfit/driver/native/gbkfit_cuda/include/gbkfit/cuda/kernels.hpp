#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>

#include "gbkfit/cuda/fftutils.hpp"
#include "gbkfit/cuda/random.hpp"

namespace gbkfit::cuda::kernels {

template<typename T> inline constexpr void
atomic_add(T* addr, T val)
{
    atomicAdd(addr, val);
}

template<typename T> inline constexpr void
atomic_set(T* addr, T val)
{
    // TODO: revise this
    *addr = val;
}

template<typename T> __global__ void
math_complex_multiply_and_scale(
        typename cufft<T>::complex* arr1,
        typename cufft<T>::complex* arr2,
        int n, T scale)
{
    // Parallelization: per item in arr1/arr2
    const int nthreads = n;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    typename cufft<T>::complex a, b;

    a.x = arr1[tid].x;
    a.y = arr1[tid].y;
    b.x = arr2[tid % n].x;
    b.y = arr2[tid % n].y;

    arr1[tid].x = (a.x*b.x-a.y*b.y)*scale;
    arr1[tid].y = (a.x*b.y+a.y*b.x)*scale;
}

template<typename T> __global__ void
dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    // Parallelization: per 3d position in the dst dcube
    const int nthreads = dst_size_x * dst_size_y * dst_size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, dst_size_x, dst_size_y);

    gbkfit::dmodel_dcube_downscale(
            x, y, z,
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_cube, dst_cube);
}

template<typename T> __global__ void
dmodel_dcube_mask(
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    // Parallelization: per 3d position in the dcube
    const int nthreads = size_x * size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, size_x, size_y);

    gbkfit::dmodel_dcube_mask(
            x, y, z,
            cutoff, apply,
            size_x, size_y, size_z,
            dcube_d, dcube_m, dcube_w);
}

template<typename T> __global__ void
dcube_moments(
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
    const int nthreads = size_x * size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, size_x);

    gbkfit::dmodel_mmaps_moments(
            x, y,
            size_x, size_y, size_z,
            step_x, step_y, step_z,
            zero_x, zero_y, zero_z,
            dcube_d, dcube_w,
            cutoff, norders, orders,
            mmaps_d, mmaps_w, mmaps_m);
}

template<typename T> __global__ void
gmodel_wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_cube,
        T* spec_cube)
{
    // Parallelization: per 2d spatial position
    const int nthreads = spat_size_x * spat_size_y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, spat_size_x);

    gbkfit::gmodel_wcube_pixel(
            x, y,
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            spat_cube,
            spec_cube);
}

template<typename T> __global__ void
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
        const T* opacity,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        T* image, T* scube, T* wdata, T* rdata, T* ordata,
        T* rdata_cmp, T* vdata_cmp, T* ddata_cmp, T* ordata_cmp)
{
    // Parallelization: per cloud
    const int nthreads = nclouds;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    RNG<T> rng(tid);
    gbkfit::gmodel_mcdisk_evaluate_cloud<atomic_set<T>, atomic_add<T>>(
            rng, tid,
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
            opacity,
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            image, scube, wdata, rdata, ordata,
            rdata_cmp, vdata_cmp, ddata_cmp, ordata_cmp);
}

template<typename T> __global__ void
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
        const T* opacity,
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        T* image, T* scube, T* wdata, T* rdata, T* ordata,
        T* rdata_cmp, T* vdata_cmp, T* ddata_cmp, T* ordata_cmp)
{
    // Parallelization: per 3d spatial position
    // TODO: Implement ray marching and revise parallelization scheme
    const int nthreads = spat_size_x * spat_size_y * spat_size_z;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, spat_size_x, spat_size_y);

    gbkfit::gmodel_smdisk_evaluate_spaxel<atomic_add<T>>(
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
            opacity,
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            image, scube, wdata, rdata, ordata,
            rdata_cmp, vdata_cmp, ddata_cmp, ordata_cmp);
}

template<typename T> __global__ void
objective_count_pixels(
        const T* data1, const T* data2, int size, T epsilon, int* counts)
{
    // Instead of using as many threads as possible, use a fixed number of them
    // This way we do not get too much overhead from the atomic adds
    // Revise this decision in the future
    // Each thread is assigned size/nthreads positions in data1 and data2
    const int nthreads = 1024 * 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;

    int count_data1 = 0;
    int count_data2 = 0;
    int count_both = 0;

    for (int i = tid; i < size; i += nthreads)
    {
        const bool has_data1 = std::abs(data1[i]) > epsilon;
        const bool has_data2 = std::abs(data2[i]) > epsilon;
        count_data1 += has_data1 && !has_data2;
        count_data2 += !has_data1 && has_data2;
        count_both += has_data1 && has_data2;
    }

    if (count_data1) {
        atomicAdd(&counts[0], count_data1);
    }
    if (count_data2) {
        atomicAdd(&counts[1], count_data2);
    }
    if (count_both) {
        atomicAdd(&counts[2], count_both);
    }
}

} // namespace gbkfit::cuda::kernels
