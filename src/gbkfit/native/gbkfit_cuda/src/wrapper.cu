
#include "gbkfit/cuda/kernels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit { namespace cuda {

template<typename T> void
Wrapper<T>::objective_count_pixels(
        const T* data, const T* model, int size, int* counts)
{
    unsigned int n = size;
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::objective_count_pixels<<<gsize, bsize>>>(
            data, model, size, counts);
    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::dmodel_dcube_complex_multiply_and_scale(
        typename cufft<T>::complex* arr1,
        typename cufft<T>::complex* arr2,
        int n, T scale)
{
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::dmodel_dcube_complex_multiply_and_scale<<<gsize, bsize>>>(
            arr1, arr2, n, scale);
    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    unsigned int n = dst_size_x * dst_size_y * dst_size_z;
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::dmodel_dcube_downscale<<<gsize, bsize>>>(
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_cube, dst_cube);
    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::dmodel_dcube_make_mask(
        bool mask_spat, bool mask_spec, T mask_coef,
        int size_x, int size_y, int size_z,
        T* cube, T* mask)
{
    unsigned int n = size_x * size_y;
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::dmodel_dcube_make_mask<<<gsize, bsize>>>(
            mask_spat, mask_spec, mask_coef,
            size_x, size_y, size_z,
            cube, mask);
    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::dmodel_mmaps_moments(
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* scube,
        T* mmaps, T* masks, const int* orders, int norders)
{
    unsigned int n = size_x * size_y;
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);

    kernels::dcube_moments<<<gsize, bsize>>>(
            size_x, size_y, size_z,
            step_x, step_y, step_z,
            zero_x, zero_y, zero_z,
            scube,
            mmaps, masks, orders, norders);

    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::gmodel_mcdisk_evaluate(
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
    const int n = nclouds;
    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::gmodel_mcdisk_evaluate<<<gsize, bsize>>>(
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
            spec_zero,
            image, scube, rcube,
            rdata, vdata, ddata);

    cudaDeviceSynchronize();
}

template<typename T> void
Wrapper<T>::gmodel_smdisk_evaluate(
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
    const int n = spat_size_x * spat_size_y;

    dim3 bsize(256);
    dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernels::gmodel_smdisk_evaluate<<<gsize, bsize>>>(
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
            spec_zero,
            image, scube, rcube,
            rdata, vdata, ddata);

    cudaDeviceSynchronize();
}

#define INSTANTIATE(T)          \
    template struct Wrapper<T>;
INSTANTIATE(float)
#undef INSTANTIATE

}} // namespace gbkfit::cuda::wrapper
