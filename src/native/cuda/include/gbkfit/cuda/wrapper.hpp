#pragma once

#include <gbkfit/gmodel/disk.hpp>

namespace gbkfit { namespace cuda {

template<typename T>
struct Wrapper
{
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
            T step_z, T zero_z,
            const T* dcube_d,
            const T* dcube_w,
            T cutoff, int norders, const int* orders,
            T* mmaps_d, T* mmaps_m, T* mmaps_w);

    static void
    dmodel_mmaps_gaussian(
            int size_x, int size_y, int size_z,
            T step_z, T zero_z,
            const T* dcube_d,
            T cutoff, int norders, const int* orders,
            T* mmaps_d, T* mmaps_m);

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
    gmodel_wcube_evaluate(
            int spat_size_x, int spat_size_y, int spat_size_z,
            int spec_size_z,
            const T* spat_data,
            T* spec_data);

    static void
    gmodel_mcdisk_evaluate(const DiskArgs<T>& a, const MCDiskArgs<T>& mc);

    static void
    gmodel_smdisk_evaluate(const DiskArgs<T>& a);

    static void
    objective_residual(
            const T* obs_d, const T* obs_e, const T* obs_m,
            const T* mdl_d, const T* mdl_w, const T* mdl_m,
            int size, T weight, T* res);

    static void
    objective_residual_sum(
            const T* residual, int size, bool squared, double* sum);
};

}} // namespace gbkfit::cuda
