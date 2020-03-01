#pragma once

#include <curand_kernel.h>

#include <gbkfit/gmodel/disks.hpp>
#include "gbkfit/cuda/fftutils.hpp"

namespace gbkfit {

template<typename T>
struct RNG
{
    __device__
    RNG(unsigned int tid) {
        curand_init(tid, 0, 0, &state);
    }

    __device__ T
    operator ()(void) {
        return curand_uniform(&state);
    }

    curandState state;
};

} // namespace gbkfit

namespace gbkfit { namespace cuda { namespace kernels {

constexpr void
index_1d_to_3d(
        int& out_x, int& out_y, int& out_z,
        int idx, int size_x, int size_y)
{
    out_z = idx/(size_x*size_y);
    idx -= out_z*size_x*size_y;
    out_y = idx/size_x;
    idx -= out_y*size_x;
    out_x = idx;
}

constexpr void
index_1d_to_2d(
        int& out_x, int& out_y,
        int idx, int size_x)
{
    out_y = idx/size_x;
    idx -= out_y*size_x;
    out_x = idx;
}

template<typename T> __global__ void
dmodel_dcube_complex_multiply_and_scale(
        typename cufft<T>::complex* ary1,
        typename cufft<T>::complex* ary2,
        int n, T scale)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    typename cufft<T>::complex a, b, c;

    a.x = ary1[tid].x;
    a.y = ary1[tid].y;
    b.x = ary2[tid % n].x;
    b.y = ary2[tid % n].y;

    c.x = (a.x*b.x-a.y*b.y)*scale;
    c.y = (a.x*b.y+a.y*b.x)*scale;

    ary1[tid].x = c.x;
    ary1[tid].y = c.y;
}

template<typename T> __global__ void
dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    int n = dst_size_x * dst_size_y * dst_size_z;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    const T nfactor = T{1} / (scale_x * scale_y * scale_z);

    int x, y, z;
    index_1d_to_3d(x, y, z, tid, dst_size_x, dst_size_y);

    // Src cube 3d index
    int nx = offset_x + x * scale_x;
    int ny = offset_y + y * scale_y;
    int nz = offset_z + z * scale_z;

    // Calculate average value under the current position
    T sum = 0;
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

template<typename T> constexpr void
evaluate_image(T* image, int x, int y, T rvalue, int spat_size_x)
{
    int idx = x + y * spat_size_x;
    atomicAdd(&image[idx], rvalue);
}

template<typename T> constexpr void
evaluate_scube(
        T* scube, int x, int y, T rvalue, T vvalue, T dvalue,
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero)
{
    // Calculate a spectral range that encloses most of the flux.
    T zmin = vvalue - dvalue * 4;
    T zmax = vvalue + dvalue * 4;
    int zmin_idx = fmax<T>(std::rint(
            (zmin - spec_zero)/spec_step), 0);
    int zmax_idx = fmin<T>(std::rint(
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
    //  scube[idx] += flux;
        atomicAdd(&scube[idx], flux);
    }
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
    int n = spat_size_x * spat_size_y * spat_size_z;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    int x, y;
    index_1d_to_2d(x, y, tid, spat_size_x);

    for (int z = 0; z < spat_size_z; ++z)
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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nclouds)
        return;

    int x, y, z;
    T bvalue, vvalue, dvalue;
    RNG<T> rng(tid);
    bool success = gbkfit::gmodel_mcdisk_evaluate_cloud(
            x, y, z,
            bvalue, vvalue, dvalue,
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
        return;
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

}}} // namespace gbkfit::cuda::kernels
