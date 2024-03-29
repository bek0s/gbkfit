#pragma once

#include "gbkfit/host/common.hpp"

namespace gbkfit::host {

template<typename T>
struct GModel
{
    void
    convolve(
            int data1_size_x, int data1_size_y, int data1_size_z,
            int data2_size_x, int data2_size_y, int data2_size_z,
            Ptr data1,
            Ptr data2,
            Ptr result) const;

    void
    wcube_evaluate(
            int spat_size_x, int spat_size_y, int spat_size_z,
            int spec_size_z,
            Ptr spat_cube,
            Ptr spec_cube) const;

    void
    mcdisk_evaluate(
            T cflux, int nclouds,
            Ptr ncloudscsum, int ncloudscsum_len,
            Ptr hasordint,
            bool loose, bool tilted,
            int nrnodes, Ptr rnodes,
            Ptr vsys,
            Ptr xpos, Ptr ypos,
            Ptr posa, Ptr incl,
            int nrt,
            Ptr rpt_uids,
            Ptr rpt_cvalues, Ptr rpt_ccounts,
            Ptr rpt_pvalues, Ptr rpt_pcounts,
            Ptr rht_uids,
            Ptr rht_cvalues, Ptr rht_ccounts,
            Ptr rht_pvalues, Ptr rht_pcounts,
            int nvt,
            Ptr vpt_uids,
            Ptr vpt_cvalues, Ptr vpt_ccounts,
            Ptr vpt_pvalues, Ptr vpt_pcounts,
            Ptr vht_uids,
            Ptr vht_cvalues, Ptr vht_ccounts,
            Ptr vht_pvalues, Ptr vht_pcounts,
            int ndt,
            Ptr dpt_uids,
            Ptr dpt_cvalues, Ptr dpt_ccounts,
            Ptr dpt_pvalues, Ptr dpt_pcounts,
            Ptr dht_uids,
            Ptr dht_cvalues, Ptr dht_ccounts,
            Ptr dht_pvalues, Ptr dht_pcounts,
            int nzt,
            Ptr zpt_uids,
            Ptr zpt_cvalues, Ptr zpt_ccounts,
            Ptr zpt_pvalues, Ptr zpt_pcounts,
            int nst,
            Ptr spt_uids,
            Ptr spt_cvalues, Ptr spt_ccounts,
            Ptr spt_pvalues, Ptr spt_pcounts,
            int nwt,
            Ptr wpt_uids,
            Ptr wpt_cvalues, Ptr wpt_ccounts,
            Ptr wpt_pvalues, Ptr wpt_pcounts,
            Ptr opacity,
            int spat_size_x, int spat_size_y, int spat_size_z,
            T spat_step_x, T spat_step_y, T spat_step_z,
            T spat_zero_x, T spat_zero_y, T spat_zero_z,
            int spec_size,
            T spec_step,
            T spec_zero,
            Ptr image, Ptr scube,
            Ptr wdata, Ptr wdata_cmp,
            Ptr rdata, Ptr rdata_cmp,
            Ptr ordata, Ptr ordata_cmp,
            Ptr vdata_cmp, Ptr ddata_cmp) const;

    void
    smdisk_evaluate(
            bool loose, bool tilted,
            int nrnodes, Ptr rnodes,
            Ptr vsys,
            Ptr xpos, Ptr ypos,
            Ptr posa, Ptr incl,
            int nrt,
            Ptr rpt_uids,
            Ptr rpt_cvalues, Ptr rpt_ccounts,
            Ptr rpt_pvalues, Ptr rpt_pcounts,
            Ptr rht_uids,
            Ptr rht_cvalues, Ptr rht_ccounts,
            Ptr rht_pvalues, Ptr rht_pcounts,
            int nvt,
            Ptr vpt_uids,
            Ptr vpt_cvalues, Ptr vpt_ccounts,
            Ptr vpt_pvalues, Ptr vpt_pcounts,
            Ptr vht_uids,
            Ptr vht_cvalues, Ptr vht_ccounts,
            Ptr vht_pvalues, Ptr vht_pcounts,
            int ndt,
            Ptr dpt_uids,
            Ptr dpt_cvalues, Ptr dpt_ccounts,
            Ptr dpt_pvalues, Ptr dpt_pcounts,
            Ptr dht_uids,
            Ptr dht_cvalues, Ptr dht_ccounts,
            Ptr dht_pvalues, Ptr dht_pcounts,
            int nzt,
            Ptr zpt_uids,
            Ptr zpt_cvalues, Ptr zpt_ccounts,
            Ptr zpt_pvalues, Ptr zpt_pcounts,
            int nst,
            Ptr spt_uids,
            Ptr spt_cvalues, Ptr spt_ccounts,
            Ptr spt_pvalues, Ptr spt_pcounts,
            int nwt,
            Ptr wpt_uids,
            Ptr wpt_cvalues, Ptr wpt_ccounts,
            Ptr wpt_pvalues, Ptr wpt_pcounts,
            Ptr opacity,
            int spat_size_x, int spat_size_y, int spat_size_z,
            T spat_step_x, T spat_step_y, T spat_step_z,
            T spat_zero_x, T spat_zero_y, T spat_zero_z,
            int spec_size,
            T spec_step,
            T spec_zero,
            Ptr image, Ptr scube,
            Ptr wdata, Ptr wdata_cmp,
            Ptr rdata, Ptr rdata_cmp,
            Ptr ordata, Ptr ordata_cmp,
            Ptr vdata_cmp, Ptr ddata_cmp) const;
};

} // namespace gbkfit::host
