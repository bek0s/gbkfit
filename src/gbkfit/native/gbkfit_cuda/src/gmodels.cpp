
#include "gbkfit/cuda/gmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit { namespace cuda {

template<typename T> void
GModelMCDisk<T>::evaluate(
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
        Ptr image, Ptr scube, Ptr rcube,
        Ptr rdata, Ptr vdata, Ptr ddata) const
{
    Wrapper<T>::gmodel_mcdisk_evaluate(
            cflux, nclouds,
            reinterpret_cast<const int*>(ncloudspt),
            loose, tilted,
            nrnodes,
            reinterpret_cast<const T*>(rnodes),
            reinterpret_cast<const T*>(vsys),
            reinterpret_cast<const T*>(xpos),
            reinterpret_cast<const T*>(ypos),
            reinterpret_cast<const T*>(posa),
            reinterpret_cast<const T*>(incl),
            nrt,
            reinterpret_cast<const int*>(rpt_uids),
            reinterpret_cast<const T*>(rpt_cvalues),
            reinterpret_cast<const int*>(rpt_ccounts),
            reinterpret_cast<const T*>(rpt_pvalues),
            reinterpret_cast<const int*>(rpt_pcounts),
            reinterpret_cast<const int*>(rht_uids),
            reinterpret_cast<const T*>(rht_cvalues),
            reinterpret_cast<const int*>(rht_ccounts),
            reinterpret_cast<const T*>(rht_pvalues),
            reinterpret_cast<const int*>(rht_pcounts),
            nvt,
            reinterpret_cast<const int*>(vpt_uids),
            reinterpret_cast<const T*>(vpt_cvalues),
            reinterpret_cast<const int*>(vpt_ccounts),
            reinterpret_cast<const T*>(vpt_pvalues),
            reinterpret_cast<const int*>(vpt_pcounts),
            reinterpret_cast<const int*>(vht_uids),
            reinterpret_cast<const T*>(vht_cvalues),
            reinterpret_cast<const int*>(vht_ccounts),
            reinterpret_cast<const T*>(vht_pvalues),
            reinterpret_cast<const int*>(vht_pcounts),
            ndt,
            reinterpret_cast<const int*>(dpt_uids),
            reinterpret_cast<const T*>(dpt_cvalues),
            reinterpret_cast<const int*>(dpt_ccounts),
            reinterpret_cast<const T*>(dpt_pvalues),
            reinterpret_cast<const int*>(dpt_pcounts),
            reinterpret_cast<const int*>(dht_uids),
            reinterpret_cast<const T*>(dht_cvalues),
            reinterpret_cast<const int*>(dht_ccounts),
            reinterpret_cast<const T*>(dht_pvalues),
            reinterpret_cast<const int*>(dht_pcounts),
            nwt,
            reinterpret_cast<const int*>(wpt_uids),
            reinterpret_cast<const T*>(wpt_cvalues),
            reinterpret_cast<const int*>(wpt_ccounts),
            reinterpret_cast<const T*>(wpt_pvalues),
            reinterpret_cast<const int*>(wpt_pcounts),
            nst,
            reinterpret_cast<const int*>(spt_uids),
            reinterpret_cast<const T*>(spt_cvalues),
            reinterpret_cast<const int*>(spt_ccounts),
            reinterpret_cast<const T*>(spt_pvalues),
            reinterpret_cast<const int*>(spt_pcounts),
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            reinterpret_cast<T*>(image),
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(rcube),
            reinterpret_cast<T*>(rdata),
            reinterpret_cast<T*>(vdata),
            reinterpret_cast<T*>(ddata));
}

template<typename T> void
GModelSMDisk<T>::evaluate(
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
        Ptr image, Ptr scube, Ptr rcube,
        Ptr rdata, Ptr vdata, Ptr ddata) const
{
    Wrapper<T>::gmodel_smdisk_evaluate(
            loose, tilted,
            nrnodes,
            reinterpret_cast<const T*>(rnodes),
            reinterpret_cast<const T*>(vsys),
            reinterpret_cast<const T*>(xpos),
            reinterpret_cast<const T*>(ypos),
            reinterpret_cast<const T*>(posa),
            reinterpret_cast<const T*>(incl),
            nrt,
            reinterpret_cast<const int*>(rpt_uids),
            reinterpret_cast<const T*>(rpt_cvalues),
            reinterpret_cast<const int*>(rpt_ccounts),
            reinterpret_cast<const T*>(rpt_pvalues),
            reinterpret_cast<const int*>(rpt_pcounts),
            reinterpret_cast<const int*>(rht_uids),
            reinterpret_cast<const T*>(rht_cvalues),
            reinterpret_cast<const int*>(rht_ccounts),
            reinterpret_cast<const T*>(rht_pvalues),
            reinterpret_cast<const int*>(rht_pcounts),
            nvt,
            reinterpret_cast<const int*>(vpt_uids),
            reinterpret_cast<const T*>(vpt_cvalues),
            reinterpret_cast<const int*>(vpt_ccounts),
            reinterpret_cast<const T*>(vpt_pvalues),
            reinterpret_cast<const int*>(vpt_pcounts),
            reinterpret_cast<const int*>(vht_uids),
            reinterpret_cast<const T*>(vht_cvalues),
            reinterpret_cast<const int*>(vht_ccounts),
            reinterpret_cast<const T*>(vht_pvalues),
            reinterpret_cast<const int*>(vht_pcounts),
            ndt,
            reinterpret_cast<const int*>(dpt_uids),
            reinterpret_cast<const T*>(dpt_cvalues),
            reinterpret_cast<const int*>(dpt_ccounts),
            reinterpret_cast<const T*>(dpt_pvalues),
            reinterpret_cast<const int*>(dpt_pcounts),
            reinterpret_cast<const int*>(dht_uids),
            reinterpret_cast<const T*>(dht_cvalues),
            reinterpret_cast<const int*>(dht_ccounts),
            reinterpret_cast<const T*>(dht_pvalues),
            reinterpret_cast<const int*>(dht_pcounts),
            nwt,
            reinterpret_cast<const int*>(wpt_uids),
            reinterpret_cast<const T*>(wpt_cvalues),
            reinterpret_cast<const int*>(wpt_ccounts),
            reinterpret_cast<const T*>(wpt_pvalues),
            reinterpret_cast<const int*>(wpt_pcounts),
            nst,
            reinterpret_cast<const int*>(spt_uids),
            reinterpret_cast<const T*>(spt_cvalues),
            reinterpret_cast<const int*>(spt_ccounts),
            reinterpret_cast<const T*>(spt_pvalues),
            reinterpret_cast<const int*>(spt_pcounts),
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            reinterpret_cast<T*>(image),
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(rcube),
            reinterpret_cast<T*>(rdata),
            reinterpret_cast<T*>(vdata),
            reinterpret_cast<T*>(ddata));
}

#define INSTANTIATE(T)               \
    template struct GModelMCDisk<T>; \
    template struct GModelSMDisk<T>;
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}} // namespace gbkfit::cuda
