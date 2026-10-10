#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>
// #include <iostream>

#include <omp.h>

#include <gbkfit/dmodel/dmodels.hpp>
#include <gbkfit/gmodel/gmodels.hpp>
#include <gbkfit/objective/objective.hpp>

namespace gbkfit::host::kernels {

template<typename T> inline void
atomic_add(T* addr, T val)
{
    #pragma omp atomic update
    addr[0] += val;
}

// A plain add, for outputs that one thread writes to
template<typename T> inline void
add(T* addr, T val)
{
    addr[0] += val;
}

template<typename T> inline void
atomic_set(T* addr, T val)
{
    #pragma omp atomic write
    addr[0] = val;
}

template<typename T> void
math_complex_multiply_and_scale(
        std::complex<T>* arr1, const std::complex<T>* arr2, int n, T scale)
{
    // Parallelization: per item in arr1/arr2
    #pragma omp parallel for
    for(int i = 0; i < n; ++i)
    {
        const T ar = arr1[i].real(), ai = arr1[i].imag();
        const T br = arr2[i].real(), bi = arr2[i].imag();
        arr1[i] = {(ar*br - ai*bi) * scale, (ar*bi + ai*br) * scale};
    }
}

template<typename T> void
dmodel_dcube_downscale(
        int scale_x, int scale_y, int scale_z,
        int offset_x, int offset_y, int offset_z,
        int src_size_x, int src_size_y, int src_size_z,
        int dst_size_x, int dst_size_y, int dst_size_z,
        const T* src_dcube, T* dst_dcube)
{
    // Parallelization: per 3d position in the dst dcube
    #pragma omp parallel for collapse(3)
    for(int z = 0; z < dst_size_z; ++z) {
    for(int y = 0; y < dst_size_y; ++y) {
    for(int x = 0; x < dst_size_x; ++x) {

    gbkfit::dmodel_dcube_downscale(
            x, y, z,
            scale_x, scale_y, scale_z,
            offset_x, offset_y, offset_z,
            src_size_x, src_size_y, src_size_z,
            dst_size_x, dst_size_y, dst_size_z,
            src_dcube, dst_dcube);

    }
    }
    }
}

template<typename T> void
dmodel_dcube_mask(
        T cutoff, bool apply,
        int size_x, int size_y, int size_z,
        T* dcube_d, T* dcube_m, T* dcube_w)
{
    // Parallelization: per 3d position in the dcube
    #pragma omp parallel for collapse(3)
    for(int z = 0; z < size_z; ++z) {
    for(int y = 0; y < size_y; ++y) {
    for(int x = 0; x < size_x; ++x) {

    gbkfit::dmodel_dcube_mask(
            x, y, z,
            cutoff, apply,
            size_x, size_y, size_z,
            dcube_d, dcube_m, dcube_w);

    }
    }
    }
}

// The number of pixels of a row whose spectra are copied together by
// for_each_spectrum
constexpr int SPECTRA_TILE = 128;

// Call f(x, y, spectrum_d, spectrum_w) for each pixel (x, y) of a cube
// (and of its weights, if any) in parallel, with the spectra of the pixel
// in contiguous copies. Reading the spectra from the cube, a whole channel
// apart, misses the cache on every value when the size of a channel is a
// power of two (all values then map to the same cache set). So the
// spectra of a tile of pixels of a row are copied first, reading each
// channel of the tile contiguously, and one value more than the spectrum
// apart, for the same reason.
template<typename T, typename F> void
for_each_spectrum(
        int size_x, int size_y, int size_z,
        const T* cube_d, const T* cube_w, F f)
{
    const int ntiles = (size_x + SPECTRA_TILE - 1) / SPECTRA_TILE;
    const int pitch = size_z + 1;

    // Parallelization: per tile of a row
    #pragma omp parallel
    {
    std::vector<T> spectra_d(SPECTRA_TILE * pitch);
    std::vector<T> spectra_w(cube_w ? SPECTRA_TILE * pitch : 0);

    #pragma omp for collapse(2) schedule(dynamic)
    for (int y = 0; y < size_y; ++y) {
    for (int tile = 0; tile < ntiles; ++tile) {

    const int x0 = tile * SPECTRA_TILE;
    const int n = std::min(SPECTRA_TILE, size_x - x0);
    for (int z = 0; z < size_z; ++z)
    {
        const int idx = index_3d_to_1d(x0, y, z, size_x, size_y);
        for (int i = 0; i < n; ++i)
        {
            spectra_d[i * pitch + z] = cube_d[idx + i];
            if (cube_w)
                spectra_w[i * pitch + z] = cube_w[idx + i];
        }
    }
    for (int i = 0; i < n; ++i)
    {
        f(x0 + i, y,
          &spectra_d[i * pitch], cube_w ? &spectra_w[i * pitch] : nullptr);
    }

    }
    }
    }
}

template<typename T> void
dmodel_mmaps_moments(
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff,
        int norders,
        const int* orders,
        T* mmaps_d,
        T* mmaps_m,
        T* mmaps_w)
{
    for_each_spectrum(
            size_x, size_y, size_z, dcube_d, dcube_w,
            [&](int x, int y, const T* spectrum_d, const T* spectrum_w) {
        gbkfit::dmodel_mmaps_moments(
                x, y,
                size_x, size_y, size_z,
                step_z, zero_z,
                spectrum_d, spectrum_w, 1,
                cutoff, norders, orders,
                mmaps_d, mmaps_m, mmaps_w);
    });
}

template<typename T> void
dmodel_mmaps_gaussian(
        int size_x, int size_y, int size_z,
        T step_z, T zero_z,
        const T* dcube_d,
        T cutoff, int norders, const int* orders,
        T* mmaps_d, T* mmaps_m)
{
    for_each_spectrum(
            size_x, size_y, size_z, dcube_d, static_cast<const T*>(nullptr),
            [&](int x, int y, const T* spectrum, const T*) {
        gbkfit::dmodel_mmaps_gaussian(
                x, y,
                size_x, size_y, size_z,
                step_z, zero_z,
                spectrum, 1,
                cutoff, norders, orders,
                mmaps_d, mmaps_m);
    });
}

template<typename T> void
dmodel_dcube_convolve_z(
        int size_x, int size_y, int size_z,
        int nk, const T* kernels, T* cube, T* scratch)
{
    // Parallelization: per row of the dcube
    #pragma omp parallel for collapse(2)
    for (int z = 0; z < size_z; ++z) {
    for (int y = 0; y < size_y; ++y) {

    gbkfit::dmodel_dcube_convolve_z_row(
            y, z, size_x, size_y, size_z, nk, kernels, cube, scratch);

    }
    }

    const long long n = (long long)size_x * size_y * size_z;
    #pragma omp parallel for
    for (long long i = 0; i < n; ++i)
        cube[i] = scratch[i];
}

template<typename T> void
dmodel_lens_resample(
        int nx, int ny, int nz, int sx, int sy,
        const T* source_x, const T* source_y,
        const T* source, T* image)
{
    // Parallelization: per 3d position in the image cube
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {

    gbkfit::dmodel_lens_resample(
            x, y, z, nx, ny, sx, sy, source_x, source_y, source, image);

    }
    }
    }
}

template<typename T> void
dmodel_regions_sum(
        int nregions, int npix, int size_z,
        const int* indptr, const int* indices, const T* weights,
        const T* cube, T* out)
{
    // Parallelization: per region and channel
    #pragma omp parallel for collapse(2)
    for (int z = 0; z < size_z; ++z) {
    for (int r = 0; r < nregions; ++r) {

    gbkfit::dmodel_regions_sum(
            r, z,
            nregions, npix,
            indptr, indices, weights,
            cube, out);

    }
    }
}

template<typename T> void
gmodel_wcube_evaluate(
        int spat_size_x, int spat_size_y, int spat_size_z,
        int spec_size_z,
        const T* spat_cube,
        T* spec_cube)
{
    // Parallelization: per 2d spatial position
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < spat_size_y; ++y) {
    for(int x = 0; x < spat_size_x; ++x) {

    gbkfit::gmodel_wcube_pixel(
            x, y,
            spat_size_x, spat_size_y, spat_size_z,
            spec_size_z,
            spat_cube,
            spec_cube);

    }
    }
}

template<typename T> void
gmodel_mcdisk_evaluate(const DiskArgs<T>& a, const MCDiskArgs<T>& mc)
{
    // Parallelization: per cloud
    // Each cloud has its own stream of random numbers
    #pragma omp parallel for
    for(int ci = 0; ci < mc.nclouds; ++ci)
    {
        RNG<T> rng(mc.seed, ci);
        gbkfit::gmodel_mcdisk_evaluate_cloud<atomic_set<T>, atomic_add<T>>(
                rng, ci, a, mc);
    }
}

template<typename T> void
gmodel_smdisk_evaluate(const DiskArgs<T>& a)
{
    // Parallelization: per 2d spatial position. Each thread evaluates the
    // voxels of its spaxels in turn, so their light is added to the
    // spaxel without atomics, and in the same order on every run. The
    // spaxels off the disk take little time: chunks are handed out as
    // threads become free.
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for(int y = 0; y < a.spat_size[1]; ++y) {
    for(int x = 0; x < a.spat_size[0]; ++x) {
    for(int z = 0; z < a.spat_size[2]; ++z) {

    gbkfit::gmodel_smdisk_evaluate_spaxel<add<T>>(x, y, z, a);

    }
    }
    }
}

template<typename T> void
objective_residual(
        const T* obs_d, const T* obs_e, const T* obs_m,
        const T* mdl_d, const T* mdl_w, const T* mdl_m,
        int size, T weight, T* res)
{
    #pragma omp parallel for
    for(int i = 0; i < size; ++i) {

        gbkfit::objective_residual(
                i,
                obs_d, obs_e, obs_m,
                mdl_d, mdl_w, mdl_m,
                size, weight, res);

    }
}

template<typename T> void
objective_residual_sum(
        const T* residual, int size, bool squared, double* sum)
{
    // Accumulate in double precision: a cube can have millions of terms
    double sum_ = 0;
    #pragma omp parallel for reduction(+:sum_)
    for(int i = 0; i < size; ++i) {

    const double r = residual[i];
    sum_ += squared ? r * r : std::abs(r);

    }
    sum[0] = sum_;
}

} // namespace gbkfit::host::kernels

namespace gbkfit::host {

// The kernels of the host module, with the interface of the kernel
// wrapper of the cuda module (gbkfit::cuda::Wrapper)
template<typename T>
struct Wrapper
{
    static void
    dmodel_dcube_downscale(auto... args) {
        kernels::dmodel_dcube_downscale<T>(args...);
    }

    static void
    dmodel_dcube_mask(auto... args) {
        kernels::dmodel_dcube_mask<T>(args...);
    }

    static void
    dmodel_mmaps_moments(auto... args) {
        kernels::dmodel_mmaps_moments<T>(args...);
    }

    static void
    dmodel_mmaps_gaussian(auto... args) {
        kernels::dmodel_mmaps_gaussian<T>(args...);
    }

    static void
    dmodel_dcube_convolve_z(auto... args) {
        kernels::dmodel_dcube_convolve_z<T>(args...);
    }

    static void
    dmodel_lens_resample(auto... args) {
        kernels::dmodel_lens_resample<T>(args...);
    }

    static void
    dmodel_regions_sum(auto... args) {
        kernels::dmodel_regions_sum<T>(args...);
    }

    static void
    gmodel_wcube_evaluate(auto... args) {
        kernels::gmodel_wcube_evaluate<T>(args...);
    }

    static void
    gmodel_mcdisk_evaluate(const DiskArgs<T>& a, const MCDiskArgs<T>& mc) {
        kernels::gmodel_mcdisk_evaluate<T>(a, mc);
    }

    static void
    gmodel_smdisk_evaluate(const DiskArgs<T>& a) {
        kernels::gmodel_smdisk_evaluate<T>(a);
    }

    static void
    objective_residual(auto... args) {
        kernels::objective_residual<T>(args...);
    }

    static void
    objective_residual_sum(auto... args) {
        kernels::objective_residual_sum<T>(args...);
    }
};

} // namespace gbkfit::host
