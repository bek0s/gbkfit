
#include "gbkfit/host/gmodels.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T>
void GModel<T>::wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        Ptr spat_data,
        Ptr spec_data) const
{
    kernels::gmodel_wcube_evaluate(
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            reinterpret_cast<T*>(spat_data),
            reinterpret_cast<T*>(spec_data));
};

template<typename T> void
GModel<T>::mcdisk_evaluate(
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
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        Ptr image, Ptr scube, Ptr rcube, Ptr wcube,
        Ptr rdata, Ptr vdata, Ptr ddata) const
{
    kernels::gmodel_mcdisk_evaluate(
            cflux, nclouds,
            reinterpret_cast<const int*>(ncloudscsum),
            ncloudscsum_len,
            reinterpret_cast<const bool*>(hasordint),
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
            nzt,
            reinterpret_cast<const int*>(zpt_uids),
            reinterpret_cast<const T*>(zpt_cvalues),
            reinterpret_cast<const int*>(zpt_ccounts),
            reinterpret_cast<const T*>(zpt_pvalues),
            reinterpret_cast<const int*>(zpt_pcounts),
            nst,
            reinterpret_cast<const int*>(spt_uids),
            reinterpret_cast<const T*>(spt_cvalues),
            reinterpret_cast<const int*>(spt_ccounts),
            reinterpret_cast<const T*>(spt_pvalues),
            reinterpret_cast<const int*>(spt_pcounts),
            nwt,
            reinterpret_cast<const int*>(wpt_uids),
            reinterpret_cast<const T*>(wpt_cvalues),
            reinterpret_cast<const int*>(wpt_ccounts),
            reinterpret_cast<const T*>(wpt_pvalues),
            reinterpret_cast<const int*>(wpt_pcounts),
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            reinterpret_cast<T*>(image),
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(rcube),
            reinterpret_cast<T*>(wcube),
            reinterpret_cast<T*>(rdata),
            reinterpret_cast<T*>(vdata),
            reinterpret_cast<T*>(ddata));
}

template<typename T> void
GModel<T>::smdisk_evaluate(
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
        int spat_size_x, int spat_size_y, int spat_size_z,
        T spat_step_x, T spat_step_y, T spat_step_z,
        T spat_zero_x, T spat_zero_y, T spat_zero_z,
        int spec_size,
        T spec_step,
        T spec_zero,
        Ptr image, Ptr scube, Ptr rcube, Ptr wcube,
        Ptr rdata, Ptr vdata, Ptr ddata) const
{
    kernels::gmodel_smdisk_evaluate(
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
            nzt,
            reinterpret_cast<const int*>(zpt_uids),
            reinterpret_cast<const T*>(zpt_cvalues),
            reinterpret_cast<const int*>(zpt_ccounts),
            reinterpret_cast<const T*>(zpt_pvalues),
            reinterpret_cast<const int*>(zpt_pcounts),
            nst,
            reinterpret_cast<const int*>(spt_uids),
            reinterpret_cast<const T*>(spt_cvalues),
            reinterpret_cast<const int*>(spt_ccounts),
            reinterpret_cast<const T*>(spt_pvalues),
            reinterpret_cast<const int*>(spt_pcounts),
            nwt,
            reinterpret_cast<const int*>(wpt_uids),
            reinterpret_cast<const T*>(wpt_cvalues),
            reinterpret_cast<const int*>(wpt_ccounts),
            reinterpret_cast<const T*>(wpt_pvalues),
            reinterpret_cast<const int*>(wpt_pcounts),
            spat_size_x, spat_size_y, spat_size_z,
            spat_step_x, spat_step_y, spat_step_z,
            spat_zero_x, spat_zero_y, spat_zero_z,
            spec_size,
            spec_step,
            spec_zero,
            reinterpret_cast<T*>(image),
            reinterpret_cast<T*>(scube),
            reinterpret_cast<T*>(rcube),
            reinterpret_cast<T*>(wcube),
            reinterpret_cast<T*>(rdata),
            reinterpret_cast<T*>(vdata),
            reinterpret_cast<T*>(ddata));
}

#define INSTANTIATE(T)\
    template struct GModel<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::host
