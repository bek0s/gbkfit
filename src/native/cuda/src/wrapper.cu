
#include <limits>
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

// Launch a kernel with one thread per item (n items, which can be more
// than an int holds), wait for it to finish, and raise any error of its
// launch or execution
template<typename Kernel, typename... Args> void
launch(const char* name, long long n, Kernel kernel, Args... args)
{
    if (n <= 0)
        return;
    const long long nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nblocks > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
                std::string(name) + " failed: too many items ("
                + std::to_string(n) + ")");
    }
    const dim3 bsize(BLOCK_SIZE);
    const dim3 gsize(nblocks);
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
        T step_z, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m, T* mmaps_w)
{
    const int n = size_x * size_y;
    launch("dcube_moments", n, kernels::dcube_moments<T>,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d,
            dcube_w,
            cutoff, norders, orders,
            mmaps_d, mmaps_m, mmaps_w);
}

template<typename T> void
Wrapper<T>::dmodel_mmaps_gaussian(
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
    const int n = size_x * size_y;
    launch("mmaps_gaussian", n, kernels::dmodel_mmaps_gaussian<T>,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d,
            cutoff, norders, orders,
            mmaps_d, mmaps_m);
}

template<typename T> void
Wrapper<T>::dmodel_dcube_convolve_z(
        int size_x, int size_y, int size_z,
        int nk, const T* kernels, T* cube, T* scratch)
{
    const int n = size_x * size_y * size_z;
    launch("dcube_convolve_z", n, kernels::dmodel_dcube_convolve_z<T>,
            size_x, size_y, size_z, nk, kernels, cube, scratch);
    check(cudaMemcpy(cube, scratch, sizeof(T) * n, cudaMemcpyDeviceToDevice),
          "dcube_convolve_z");
}

template<typename T> void
Wrapper<T>::dmodel_lens_resample(
        int nx, int ny, int nz, int sx, int sy,
        const T* source_x, const T* source_y,
        const T* source, T* image)
{
    const int n = nx * ny * nz;
    launch("lens_resample", n, kernels::dmodel_lens_resample<T>,
            nx, ny, nz, sx, sy, source_x, source_y, source, image);
}

template<typename T> void
Wrapper<T>::dmodel_regions_sum(
        int nregions, int npix, int size_z,
        const int* indptr, const int* indices, const T* weights,
        const T* cube, T* out)
{
    const int n = nregions * size_z;
    launch("regions_sum", n, kernels::dmodel_regions_sum<T>,
            nregions, npix, size_z,
            indptr, indices, weights,
            cube, out);
}

template<typename T> void
Wrapper<T>::gmodel_mcdisk_evaluate(
        const DiskArgs<T>& a, const MCDiskArgs<T>& mc, const FieldsArgs<T>& f)
{
    // (the kernels of the fields of disks with line opacity and without, of
    // disks dimmed by opacity, and of the others: see
    // gmodel_mcdisk_evaluate_cloud)
    namespace k = kernels;
    launch("gmodel_mcdisk_evaluate", mc.nclouds,
            f.fields && f.opt.uids ? k::gmodel_mcdisk_evaluate<T, 0, 1, 1>
            : f.fields ? k::gmodel_mcdisk_evaluate<T, 0, 1, 0>
            : a.opacity ? k::gmodel_mcdisk_evaluate<T, 1, 0, 0>
            : k::gmodel_mcdisk_evaluate<T, 0, 0, 0>,
            a, mc, f);
    // The fields of each voxel, from the moments of its clouds
    if (f.fields) {
        const long long n = 1LL * a.spat_size[0] * a.spat_size[1]
                * a.spat_size[2];
        launch("gmodel_mcdisk_fields_moments", n,
                kernels::gmodel_mcdisk_fields_moments<T>, a, f);
    }
}

template<typename T> void
Wrapper<T>::gmodel_smdisk_evaluate(
        const DiskArgs<T>& a, const FieldsArgs<T>& f)
{
    // (the voxels need not be in memory, so there can be more than an int
    // holds, e.g. 1300 x 1300 x 1300)
    const long long n = 1LL * a.spat_size[0] * a.spat_size[1]
            * gbkfit::gmodel_smdisk_size_z(a);
    // (see gmodel_mcdisk_evaluate)
    namespace k = kernels;
    launch("gmodel_smdisk_evaluate", n,
            f.fields && f.opt.uids ? k::gmodel_smdisk_evaluate<T, 0, 1, 1>
            : f.fields ? k::gmodel_smdisk_evaluate<T, 0, 1, 0>
            : a.opacity ? k::gmodel_smdisk_evaluate<T, 1, 0, 0>
            : k::gmodel_smdisk_evaluate<T, 0, 0, 0>,
            a, f);
}

template<typename T> void
Wrapper<T>::gmodel_opacity_accumulate(
        int size_x, int size_y, int size_z, T* opacity)
{
    const int n = size_x * size_y;
    launch("gmodel_opacity_accumulate", n,
            kernels::gmodel_opacity_accumulate<T>,
            size_x, size_y, size_z, opacity);
}

template<typename T> void
Wrapper<T>::gmodel_transfer_march(const TransferArgs<T>& a)
{
    const long long n = 1LL * a.spat_size[0] * a.spat_size[1]
            * kernels::transfer_chunks(a);
    launch("gmodel_transfer_march", n,
            a.tau ? kernels::gmodel_transfer_march<T, true>
                  : kernels::gmodel_transfer_march<T, false>,
            a);
    if (a.opacity) {
        launch("gmodel_transfer_opacity_front",
                1LL * a.spat_size[0] * a.spat_size[1],
                kernels::gmodel_transfer_opacity_front<T>, a);
    }
}

template<typename T> void
Wrapper<T>::gmodel_disk_traits_evaluate(const DiskTraitsArgs<T>& a)
{
    launch("gmodel_disk_traits_evaluate", a.npoints,
            kernels::gmodel_disk_traits_evaluate<T>, a);
}

template<typename T> void
Wrapper<T>::objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w,
        int size, T weight, T missing, T* res)
{
    const int n = size;
    launch("objective_residual", n, kernels::objective_residual<T>,
            obs_d, obs_e, obs_m,
            mdl_d, mdl_w,
            size, weight, missing, res);
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
        const T* residual, int size, bool squared, double* sum)
{
    // Accumulate in double precision: a cube can have millions of terms
    const double result = thrust::transform_reduce(
            thrust::device, residual, residual + size,
            ResidualSumTerm<T>{squared}, 0.0, thrust::plus<double>());
    check(cudaMemcpy(sum, &result, sizeof(double), cudaMemcpyHostToDevice),
          "objective_residual_sum");
}

#define INSTANTIATE(T)\
    template struct Wrapper<T>;
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace gbkfit::cuda
