#pragma once

namespace gbkfit::cuda {

// The kernels of the data cubes (see gbkfit/dmodel)
template<typename T>
struct DModelWrapper
{
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
};

} // namespace gbkfit::cuda
