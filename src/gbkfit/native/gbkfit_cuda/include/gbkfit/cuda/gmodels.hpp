#pragma once

#include <gbkfit/common.hpp>

namespace gbkfit { namespace cuda {

template<typename T>
struct GModelMCDisk
{
    void
    evaluate(
            T cflux, int nclouds, CPtr ncloudspt,
            bool loose, bool tilted,
            int nrnodes, CPtr rnodes,
            CPtr vsys,
            CPtr xpos, CPtr ypos,
            CPtr posa, CPtr incl,
            int nrt,
            CPtr rpt_uids,
            CPtr rpt_cvalues, CPtr rpt_ccounts,
            CPtr rpt_pvalues, CPtr rpt_pcounts,
            CPtr rht_uids,
            CPtr rht_cvalues, CPtr rht_ccounts,
            CPtr rht_pvalues, CPtr rht_pcounts,
            int nvt,
            CPtr vpt_uids,
            CPtr vpt_cvalues, CPtr vpt_ccounts,
            CPtr vpt_pvalues, CPtr vpt_pcounts,
            CPtr vht_uids,
            CPtr vht_cvalues, CPtr vht_ccounts,
            CPtr vht_pvalues, CPtr vht_pcounts,
            int ndt,
            CPtr dpt_uids,
            CPtr dpt_cvalues, CPtr dpt_ccounts,
            CPtr dpt_pvalues, CPtr dpt_pcounts,
            CPtr dht_uids,
            CPtr dht_cvalues, CPtr dht_ccounts,
            CPtr dht_pvalues, CPtr dht_pcounts,
            int nwt,
            CPtr wpt_uids,
            CPtr wpt_cvalues, CPtr wpt_ccounts,
            CPtr wpt_pvalues, CPtr wpt_pcounts,
            int nst,
            CPtr spt_uids,
            CPtr spt_cvalues, CPtr spt_ccounts,
            CPtr spt_pvalues, CPtr spt_pcounts,
            int spat_size_x, int spat_size_y, int spat_size_z,
            T spat_step_x, T spat_step_y, T spat_step_z,
            T spat_zero_x, T spat_zero_y, T spat_zero_z,
            int spec_size,
            T spec_step,
            T spec_zero,
            Ptr image, Ptr scube, Ptr bcube,
            Ptr bdata, Ptr vdata, Ptr ddata) const;
};

template<typename T>
struct GModelSMDisk
{
    void
    evaluate(
            bool loose, bool tilted,
            int nrnodes, CPtr rnodes,
            CPtr vsys,
            CPtr xpos, CPtr ypos,
            CPtr posa, CPtr incl,
            int nrt,
            CPtr rpt_uids,
            CPtr rpt_cvalues, CPtr rpt_ccounts,
            CPtr rpt_pvalues, CPtr rpt_pcounts,
            CPtr rht_uids,
            CPtr rht_cvalues, CPtr rht_ccounts,
            CPtr rht_pvalues, CPtr rht_pcounts,
            int nvt,
            CPtr vpt_uids,
            CPtr vpt_cvalues, CPtr vpt_ccounts,
            CPtr vpt_pvalues, CPtr vpt_pcounts,
            CPtr vht_uids,
            CPtr vht_cvalues, CPtr vht_ccounts,
            CPtr vht_pvalues, CPtr vht_pcounts,
            int ndt,
            CPtr dpt_uids,
            CPtr dpt_cvalues, CPtr dpt_ccounts,
            CPtr dpt_pvalues, CPtr dpt_pcounts,
            CPtr dht_uids,
            CPtr dht_cvalues, CPtr dht_ccounts,
            CPtr dht_pvalues, CPtr dht_pcounts,
            int nwt,
            CPtr wpt_uids,
            CPtr wpt_cvalues, CPtr wpt_ccounts,
            CPtr wpt_pvalues, CPtr wpt_pcounts,
            int nst,
            CPtr spt_uids,
            CPtr spt_cvalues, CPtr spt_ccounts,
            CPtr spt_pvalues, CPtr spt_pcounts,
            int spat_size_x, int spat_size_y, int spat_size_z,
            T spat_step_x, T spat_step_y, T spat_step_z,
            T spat_zero_x, T spat_zero_y, T spat_zero_z,
            int spec_size,
            T spec_step,
            T spec_zero,
            Ptr image, Ptr scube, Ptr bcube,
            Ptr bdata, Ptr vdata, Ptr ddata) const;
};

}} // namespace gbkfit::cuda
