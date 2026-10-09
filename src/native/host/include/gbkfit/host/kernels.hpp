#pragma once

#include <cassert>
#include <cmath>
#include <complex>
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

template<typename T> void
dmodel_mmaps_moments(
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* dcube_d,
        const T* dcube_w,
        T cutoff,
        int norders,
        const int* orders,
        T* mmaps_d,
        T* mmaps_m,
        T* mmaps_w)
{
    // Parallelization: per 2d spatial position
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size_y; ++y) {
    for (int x = 0; x < size_x; ++x) {

    gbkfit::dmodel_mmaps_moments(
            x, y,
            size_x, size_y, size_z,
            step_x, step_y, step_z,
            zero_x, zero_y, zero_z,
            dcube_d, dcube_w,
            cutoff, norders, orders,
            mmaps_d, mmaps_m, mmaps_w);

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
    // Parallelization: per 3d spatial position
    #pragma omp parallel for collapse(3)
    for(int y = 0; y < a.spat_size[1]; ++y) {
    for(int x = 0; x < a.spat_size[0]; ++x) {
    for(int z = 0; z < a.spat_size[2]; ++z) {

    gbkfit::gmodel_smdisk_evaluate_spaxel<atomic_add<T>>(x, y, z, a);

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
objective_residual_sum(const T* residual, int size, bool squared, T* sum)
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
