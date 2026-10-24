#pragma once

#include <gbkfit/gmodel/cone.hpp>
#include <gbkfit/gmodel/disk.hpp>
#include <gbkfit/gmodel/transfer.hpp>

namespace gbkfit { namespace cuda {

// While it lives, the calling thread launches its kernels on a device
// (that of their arrays: there can be several GPUs), and after, on the
// device it had before
class DeviceScope
{
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int m_device;
    int m_previous;
};

template<typename T>
struct Wrapper
{
    using DeviceScope = cuda::DeviceScope;

    static void
    dmodel_dcube_downscale(
            int scale_x, int scale_y, int scale_z,
            int offset_x, int offset_y, int offset_z,
            int src_size_x, int src_size_y, int src_size_z,
            int dst_size_x, int dst_size_y, int dst_size_z,
            const T* src_cube, T* dst_cube);

    static void
    dmodel_mmaps_moments(
            int size_x, int size_y, int size_z,
            T step_z, T zero_z,
            const T* dcube_d,
            const T* dcube_w,
            T cutoff, int norders, const int* orders,
            T* mmaps_d, T* mmaps_m, T* mmaps_w);

    static void
    dmodel_mmaps_gaussian(
            bool hermite,
            int size_x, int size_y, int size_z,
            T step_z, T zero_z,
            const T* dcube_d,
            T cutoff, int norders, const int* orders,
            T* mmaps_d, T* mmaps_m);

    static void
    dmodel_dcube_convolve_z(
            int size_x, int size_y, int size_z,
            int nk, const T* kernels, T* cube, T* scratch);

    static void
    dmodel_lens_resample(
            int nx, int ny, int nz, int sx, int sy,
            const T* source_x, const T* source_y,
            const T* source, T* image);

    static void
    dmodel_regions_sum(
            int nregions, int npix, int size_z,
            const int* indptr, const int* indices, const T* weights,
            const T* cube, T* out);

    static void
    gmodel_mcdisk_evaluate(
            const DiskArgs<T>& a, const GridArgs<T>& g,
            const MCDiskArgs<T>& mc, const FieldsArgs<T>& f);

    static void
    gmodel_smdisk_evaluate(
            const DiskArgs<T>& a, const GridArgs<T>& g,
            const FieldsArgs<T>& f);

    static void
    gmodel_smcone_evaluate(
            const ConeArgs<T>& a, const GridArgs<T>& g,
            const FieldsArgs<T>& f);

    static void
    gmodel_opacity_accumulate(int size_x, int size_y, int size_z, T* opacity);

    static void
    gmodel_transfer_march(const TransferArgs<T>& a);

    static void
    gmodel_disk_traits_evaluate(const DiskTraitsArgs<T>& a);

    static void
    objective_residual(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            int size, T weight, T missing, T limit, T limit_error,
            T scale, T jitter, T* res);

    static void
    objective_residual_sum(
            const T* residual, int size, bool squared, double* sum);

    static void
    objective_amplitude_sums(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            T scale, T jitter, int nouter, int nchannels, int ninner,
            double* a, double* b);

    static void
    objective_noise_sum(
            const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w,
            int size, T weight, T scale, T jitter, T scale_minimum,
            double* sum);
};

}} // namespace gbkfit::cuda
