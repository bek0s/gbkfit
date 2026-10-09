#pragma once

#include "gbkfit/bindings/arrays.hpp"

namespace gbkfit::bindings {

// The data model operations of a native module. Kernels launches the
// kernels of the module, on the memory of Device.
template<typename T, typename Device, typename Kernels>
struct DModel
{
    using Cube = Array<Device, T, nb::ndim<3>>;
    using ConstCube = Array<Device, const T, nb::ndim<3>>;
    using Image = Array<Device, T, nb::ndim<2>>;
    using ConstImage = Array<Device, const T, nb::ndim<2>>;
    using Orders = Array<Device, const int, nb::ndim<1>>;
    using Indices = Array<Device, const int, nb::ndim<1>>;
    using Weights = Array<Device, const T, nb::ndim<1>>;

    // Average blocks of scale pixels of src, starting at offset, into dst
    static void
    dcube_downscale(
            std::array<int, 3> scale, std::array<int, 3> offset,
            ConstCube src, Cube dst)
    {
        const auto src_size = size_xyz(src);
        const auto dst_size = size_xyz(dst);
        for (int i = 0; i < 3; ++i) {
            require(offset[i] + dst_size[i] * scale[i] <= src_size[i],
                    "the downscaled cube does not fit in the source cube");
        }
        Kernels::dmodel_dcube_downscale(
                scale[0], scale[1], scale[2],
                offset[0], offset[1], offset[2],
                src_size[0], src_size[1], src_size[2],
                dst_size[0], dst_size[1], dst_size[2],
                src.data(), dst.data());
    }

    // Mark the pixels of dcube_d above the cutoff in dcube_m (optional),
    // and if apply is true, set the rest to NaN
    static void
    dcube_mask(T cutoff, bool apply, Cube dcube_d, Cube dcube_m, Cube dcube_w)
    {
        require_same_shape(dcube_m, dcube_d, "dcube_m", "dcube_d");
        require_same_shape(dcube_w, dcube_d, "dcube_w", "dcube_d");
        const auto size = size_xyz(dcube_d);
        Kernels::dmodel_dcube_mask(
                cutoff, apply,
                size[0], size[1], size[2],
                dcube_d.data(), data(dcube_m), data(dcube_w));
    }

    // The moment maps of the given orders of dcube_d, with the moment
    // map mask in mmaps_m and, with the weights dcube_w, the weights of
    // the moment maps in mmaps_w: one mask and one weight map for all the
    // orders
    static void
    mmaps_moments(
            std::array<T, 3> step, std::array<T, 3> zero,
            ConstCube dcube_d, ConstCube dcube_w,
            T cutoff, Orders orders,
            Cube mmaps_d, Image mmaps_m, Image mmaps_w)
    {
        const auto size = size_xyz(dcube_d);
        const int norders = int(orders.shape(0));
        require_same_shape(dcube_w, dcube_d, "dcube_w", "dcube_d");
        require(mmaps_d.shape(0) == size_t(norders)
                && int(mmaps_d.shape(1)) == size[1]
                && int(mmaps_d.shape(2)) == size[0],
                "mmaps_d must have shape (norders, ny, nx)");
        require(int(mmaps_m.shape(0)) == size[1]
                && int(mmaps_m.shape(1)) == size[0],
                "mmaps_m must have shape (ny, nx)");
        require(!dcube_w.is_valid() || mmaps_w.is_valid(),
                "mmaps_w is required with dcube_w");
        require_same_shape(mmaps_w, mmaps_m, "mmaps_w", "mmaps_m");
        Kernels::dmodel_mmaps_moments(
                size[0], size[1], size[2],
                step[2], zero[2],
                dcube_d.data(), data(dcube_w),
                cutoff, norders, orders.data(),
                mmaps_d.data(), mmaps_m.data(), data(mmaps_w));
    }

    // The moment maps of the given orders (0, 1 or 2) of a Gaussian fitted
    // to each spectrum of dcube_d, with the moment map mask in mmaps_m
    static void
    mmaps_gaussian(
            std::array<T, 3> step, std::array<T, 3> zero,
            ConstCube dcube_d, T cutoff, Orders orders,
            Cube mmaps_d, Image mmaps_m)
    {
        const auto size = size_xyz(dcube_d);
        const int norders = int(orders.shape(0));
        require(mmaps_d.shape(0) == size_t(norders)
                && int(mmaps_d.shape(1)) == size[1]
                && int(mmaps_d.shape(2)) == size[0],
                "mmaps_d must have shape (norders, ny, nx)");
        require(int(mmaps_m.shape(0)) == size[1]
                && int(mmaps_m.shape(1)) == size[0],
                "mmaps_m must have shape (ny, nx)");
        Kernels::dmodel_mmaps_gaussian(
                size[0], size[1], size[2],
                step[2], zero[2],
                dcube_d.data(),
                cutoff, norders, orders.data(),
                mmaps_d.data(), mmaps_m.data());
    }

    // The image-plane cube image (nz, ny, nx) lensed from the source-plane
    // cube source (nz, sy, sx): each pixel is the bilinear interpolation
    // of the source at its position on the source plane, in the pixel
    // coordinates of the source (source_x and source_y, of shape (ny, nx))
    static void
    lens_resample(
            ConstImage source_x, ConstImage source_y,
            ConstCube source, Cube image)
    {
        const auto src = size_xyz(source);
        const auto dst = size_xyz(image);
        require(src[2] == dst[2],
                "source and image must have the same number of channels");
        require(int(source_x.shape(0)) == dst[1]
                && int(source_x.shape(1)) == dst[0],
                "source_x must have shape (ny, nx) of the image");
        require_same_shape(source_y, source_x, "source_y", "source_x");
        Kernels::dmodel_lens_resample(
                dst[0], dst[1], dst[2], src[0], src[1],
                source_x.data(), source_y.data(),
                source.data(), image.data());
    }

    // The weighted sums of the pixels of regions in each channel of cube
    // (nz, ny, nx), into out (nz, nregions). The regions are a CSR matrix
    // (indptr, indices, weights) of nregions rows and nx * ny columns:
    // region r has the pixels indices[k] of a channel (flat indices),
    // with the weights weights[k], for k from indptr[r] to indptr[r + 1]
    // - 1. The indices must be smaller than nx * ny.
    static void
    regions_sum(
            Indices indptr, Indices indices, Weights weights,
            ConstCube cube, Image out)
    {
        const auto size = size_xyz(cube);
        const int nregions = int(out.shape(1));
        require(int(out.shape(0)) == size[2],
                "out must have shape (nz, nregions)");
        require(int(indptr.shape(0)) == nregions + 1,
                "indptr must have nregions + 1 values");
        require(indices.shape(0) == weights.shape(0),
                "indices and weights must have the same length");
        Kernels::dmodel_regions_sum(
                nregions, size[0] * size[1], size[2],
                indptr.data(), indices.data(), weights.data(),
                cube.data(), out.data());
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<DModel>(m, ("DModel" + suffix).c_str())
                .def(nb::init<>())
                .def_static("dcube_downscale", &dcube_downscale,
                        nb::arg("scale"), nb::arg("offset"),
                        nb::arg("src").noconvert(),
                        nb::arg("dst").noconvert())
                .def_static("dcube_mask", &dcube_mask,
                        nb::arg("cutoff"), nb::arg("apply"),
                        nb::arg("dcube_d").noconvert(),
                        nb::arg("dcube_m").noconvert().none(),
                        nb::arg("dcube_w").noconvert().none())
                .def_static("mmaps_moments", &mmaps_moments,
                        nb::arg("step"), nb::arg("zero"),
                        nb::arg("dcube_d").noconvert(),
                        nb::arg("dcube_w").noconvert().none(),
                        nb::arg("cutoff"),
                        nb::arg("orders").noconvert(),
                        nb::arg("mmaps_d").noconvert(),
                        nb::arg("mmaps_m").noconvert(),
                        nb::arg("mmaps_w").noconvert().none())
                .def_static("mmaps_gaussian", &mmaps_gaussian,
                        nb::arg("step"), nb::arg("zero"),
                        nb::arg("dcube_d").noconvert(),
                        nb::arg("cutoff"),
                        nb::arg("orders").noconvert(),
                        nb::arg("mmaps_d").noconvert(),
                        nb::arg("mmaps_m").noconvert())
                .def_static("lens_resample", &lens_resample,
                        nb::arg("source_x").noconvert(),
                        nb::arg("source_y").noconvert(),
                        nb::arg("source").noconvert(),
                        nb::arg("image").noconvert())
                .def_static("regions_sum", &regions_sum,
                        nb::arg("indptr").noconvert(),
                        nb::arg("indices").noconvert(),
                        nb::arg("weights").noconvert(),
                        nb::arg("cube").noconvert(),
                        nb::arg("out").noconvert());
    }
};

} // namespace gbkfit::bindings
