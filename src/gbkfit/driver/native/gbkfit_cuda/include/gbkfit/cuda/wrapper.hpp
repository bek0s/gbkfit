#pragma once

#include "gbkfit/cuda/fftutils.hpp"

namespace gbkfit { namespace cuda {

template<typename T>
struct Wrapper
{
    static void
    math_complex_multiply_and_scale(
            typename cufft<T>::complex* arr1,
            typename cufft<T>::complex* arr2,
            int n, T scale);

    static void
    dmodel_dcube_downscale(
            int scale_x, int scale_y, int scale_z,
            int offset_x, int offset_y, int offset_z,
            int src_size_x, int src_size_y, int src_size_z,
            int dst_size_x, int dst_size_y, int dst_size_z,
            const T* src_cube, T* dst_cube);

    static void
    dmodel_dcube_mask(
            T cutoff, bool apply,
            int size_x, int size_y, int size_z,
            T* dcube_d, T* dcube_m, T* dcube_w);

    static void
    dmodel_mmaps_moments(
            int size_x, int size_y, int size_z,
            T step_x, T step_y, T step_z,
            T zero_x, T zero_y, T zero_z,
            const T* dcube_d,
            const T* dcube_w,
            T cutoff, int norders, const int* orders,
            T* mmaps_d, T* mmaps_m, T* mmaps_w);

    static void
    gmodel_wcube_evaluate(
            int spat_size_x, int spat_size_y, int spat_size_z,
            int spec_size_z,
            const T* spat_data,
            T* spec_data);

    static void
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
            T* image, T* scube,
            T* wdata, T* wdata_cmp,
            T* rdata, T* rdata_cmp,
            T* ordata, T* ordata_cmp,
            T* vdata_cmp, T* ddata_cmp);

    static void
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
            T* image, T* scube,
            T* wdata, T* wdata_cmp,
            T* rdata, T* rdata_cmp,
            T* ordata, T* ordata_cmp,
            T* vdata_cmp, T* ddata_cmp);

    static void
    objective_count_pixels(
            const T* data1, const T* data2, int size, T epsilon, int* counts);

    static void
    objective_residual(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w, const T* mdl_m,
            int size, T weight, T* res);

    static void
    objective_residual(
            const T* data, const T*  data_w, const T* data_m, const T* model, const T* model_w, const T* model_m, int size, T* out);
};

}} // namespace gbkfit::cuda
