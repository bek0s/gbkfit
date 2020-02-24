#pragma once

#include "gbkfit/drivers/cuda/cufft.hpp"


namespace gbkfit { namespace cuda { namespace kernels {

__device__
void map_index_1d_to_3d(int& out_xidx,
                        int& out_yidx,
                        int& out_zidx,
                        int idx,
                        int width,
                        int height,
                        int depth)
{
    out_zidx = idx/(width*height);
    idx -= out_zidx*width*height;

    out_yidx = idx/width;
    idx -= out_yidx*width;

    out_xidx = idx/1;
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

    auto& a = ary1[tid];
    auto& b = ary2[tid % n];
    cufftComplex result;
    result.x = (a.x * b.x - a.y * b.y) * scale;
    result.y = (a.x * b.y + a.y * b.x) * scale;
    a.x = result.x;
    a.y = result.y;
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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= tid)
        return;

    const T nfactor = T{1} / (scale_x * scale_y * scale_z);

    int x, y, z;

    map_index_1d_to_3d(x, y, z, tid, dst_size_x, dst_size_y, dst_size_z);

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

    dst_cube[idx] = sum / (scale_x * scale_y * scale_z);
}


template<typename T> __global__ void
gmodel_mcdisk_evaluate(
        T cflux, int nclouds, const int* ncloudspt,
        bool loose, bool tilted,
        int nrnodes, const T* rnodes,
        const T* vsys,
        const T* xpos, const T* ypos,
        const T* posa, const T* incl,
        int nbt,
        const int* bpt_uids,
        const T* bpt_cvalues, const int* bpt_ccounts,
        const T* bpt_pvalues, const int* bpt_pcounts,
        const int* bht_uids,
        const T* bht_cvalues, const int* bht_ccounts,
        const T* bht_pvalues, const int* bht_pcounts,
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
    if (image)
        image[0] = 42;

    if (scube)
        scube[0] = 42;
}

template<typename T> __global__ void
gmodel_smdisk_evaluate(
        bool loose, bool tilted,
        int nrnodes, const T* rnodes,
        const T* vsys,
        const T* xpos, const T* ypos,
        const T* posa, const T* incl,
        int nbt,
        const int* bpt_uids,
        const T* bpt_cvalues, const int* bpt_ccounts,
        const T* bpt_pvalues, const int* bpt_pcounts,
        const int* bht_uids,
        const T* bht_cvalues, const int* bht_ccounts,
        const T* bht_pvalues, const int* bht_pcounts,
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
    if (image)
        image[0] = 42;

    if (scube)
        scube[0] = 42;
}

}}} // namespace gbkfit::cuda::kernels
