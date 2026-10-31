#include <algorithm>

#include "gbkfit/cuda/kernels_dmodel.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_dmodel.hpp"

namespace gbkfit::cuda {

template<typename T> void
DModelWrapper<T>::dmodel_dcube_downscale(
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
DModelWrapper<T>::dmodel_mmaps_moments(
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
DModelWrapper<T>::dmodel_mmaps_gaussian(
        bool hermite,
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
    const int n = size_x * size_y;
    auto kernel = hermite
            ? kernels::dmodel_mmaps_gaussian<true, T>
            : kernels::dmodel_mmaps_gaussian<false, T>;
    launch(hermite ? "mmaps_gauss_hermite" : "mmaps_gaussian", n, kernel,
            size_x, size_y, size_z,
            step_z, zero_z,
            dcube_d,
            cutoff, norders, orders,
            mmaps_d, mmaps_m);
}

template<typename T> void
DModelWrapper<T>::dmodel_dcube_convolve_z(
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
DModelWrapper<T>::dmodel_lens_resample(
        int nx, int ny, int nz, int sx, int sy,
        const T* source_x, const T* source_y,
        const T* source, T* image)
{
    const int n = nx * ny * nz;
    launch("lens_resample", n, kernels::dmodel_lens_resample<T>,
            nx, ny, nz, sx, sy, source_x, source_y, source, image);
}

template<typename T> void
DModelWrapper<T>::dmodel_regions_sum(
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

template struct DModelWrapper<float>;
template struct DModelWrapper<double>;

} // namespace gbkfit::cuda
