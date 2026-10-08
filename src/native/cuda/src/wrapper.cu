
#include <stdexcept>
#include <string>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "gbkfit/cuda/kernels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

constexpr int BLOCK_SIZE = 256;

// Turn a cuda error into an exception, which nanobind raises in Python
void
check(cudaError_t error, const char* what)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
                std::string(what) + " failed: " + cudaGetErrorString(error));
    }
}

// Launch a kernel with one thread per item, wait for it to finish, and
// raise any error of its launch or execution
template<typename Kernel, typename... Args> void
launch(const char* name, int n, Kernel kernel, Args... args)
{
    if (n <= 0)
        return;
    const dim3 bsize(BLOCK_SIZE);
    const dim3 gsize((n + bsize.x - 1) / bsize.x);
    kernel<<<gsize, bsize>>>(args...);
    check(cudaGetLastError(), name);
    check(cudaDeviceSynchronize(), name);
}

template<typename T> void
Wrapper<T>::dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_cube, T* dst_cube)
{
    const int n = dst_size_x * dst_size_y * dst_size_z;
    launch("dmodel_dcube_downscale", n, kernels::dmodel_dcube_downscale<T>,
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_cube, dst_cube);
}

template<typename T> void
Wrapper<T>::dmodel_dcube_mask(
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    const int n = size_x * size_y * size_z;
    launch("dmodel_dcube_mask", n, kernels::dmodel_dcube_mask<T>,
            cutoff, apply,
            size_x, size_y, size_z,
            dcube_d, dcube_m, dcube_w);
}

template<typename T> void
Wrapper<T>::dmodel_mmaps_moments(
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m, T* mmaps_w)
{
    const int n = size_x * size_y;
    launch("dcube_moments", n, kernels::dcube_moments<T>,
            size_x, size_y, size_z,
            step_x, step_y, step_z,
            zero_x, zero_y, zero_z,
            dcube_d,
            dcube_w,
            cutoff, norders, orders,
            mmaps_d, mmaps_m, mmaps_w);
}

template<typename T> void
Wrapper<T>::gmodel_wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_cube,
        T* spec_cube)
{
    const int n = spat_size_x * spat_size_y;
    launch("gmodel_wcube_evaluate", n, kernels::gmodel_wcube_evaluate<T>,
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            spat_cube,
            spec_cube);
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
        T* vdata_cmp, T* ddata_cmp)
{
    const int n = nclouds;
    launch("gmodel_mcdisk_evaluate", n, kernels::gmodel_mcdisk_evaluate<T>,
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
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp);
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
        T* vdata_cmp, T* ddata_cmp)
{
    const int n = spat_size_x * spat_size_y * spat_size_z;
    launch("gmodel_smdisk_evaluate", n, kernels::gmodel_smdisk_evaluate<T>,
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
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp);
}

template<typename T> void
Wrapper<T>::objective_count_pixels(
        const T* data1, const T* data2, int size, T epsilon, int* counts)
{
    const int n = size;
    launch("objective_count_pixels", n, kernels::objective_count_pixels<T>,
            data1, data2, size, epsilon, counts);
}

template<typename T> void
Wrapper<T>::objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w, const T* mdl_m,
        int size, T weight, T* res)
{
    const int n = size;
    launch("objective_residual", n, kernels::objective_residual<T>,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w, mdl_m,
            size, weight, res);
}

// The term of a residual sum, in double precision
template<typename T>
struct ResidualSumTerm
{
    bool squared;

    __host__ __device__ double
    operator()(T residual) const
    {
        const double r = residual;
        return squared ? r * r : fabs(r);
    }
};

template<typename T> void
Wrapper<T>::objective_residual_sum(
        const T* residual, int size, bool squared, T* sum)
{
    // Accumulate in double precision: a cube can have millions of terms
    const T result = thrust::transform_reduce(
            thrust::device, residual, residual + size,
            ResidualSumTerm<T>{squared}, 0.0, thrust::plus<double>());
    check(cudaMemcpy(sum, &result, sizeof(T), cudaMemcpyHostToDevice),
          "objective_residual_sum");
}

#define INSTANTIATE(T)\
    template struct Wrapper<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::cuda
