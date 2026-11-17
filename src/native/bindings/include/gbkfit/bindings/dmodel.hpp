#pragma once

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/bindings/gil.hpp"

namespace gbkfit::bindings {

// The data model operations of a native module. Kernels launches the
// kernels of the module, on the memory of Device: each function on the
// device of the array it writes (see DeviceScope of the Kernels).
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
        const typename Kernels::DeviceScope scope(dst.device_id());
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

    // Convolve the spectra of cube along z, in place, with a kernel for
    // each channel: kernels has shape (nz, nk), nk odd (see
    // dmodel_dcube_convolve_z). scratch, of the shape of cube, is working
    // memory.
    static void
    dcube_convolve_z(ConstImage kernels, Cube cube, Cube scratch)
    {
        const typename Kernels::DeviceScope scope(cube.device_id());
        require_same_shape(scratch, cube, "scratch", "cube");
        const auto size = size_xyz(cube);
        require(int(kernels.shape(0)) == size[2],
                "kernels must have a row for each channel of cube");
        require(kernels.shape(1) % 2 == 1,
                "the kernels must have an odd size");
        Kernels::dmodel_dcube_convolve_z(
                size[0], size[1], size[2], int(kernels.shape(1)),
                kernels.data(), cube.data(), scratch.data());
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
        const typename Kernels::DeviceScope scope(mmaps_d.device_id());
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
    // to each spectrum of dcube_d, with the moment map mask in mmaps_m;
    // with hermite, of a Gauss-Hermite series (also orders 3 and 4: h3,
    // h4)
    static void
    mmaps_gaussian(
            std::array<T, 3> step, std::array<T, 3> zero,
            ConstCube dcube_d, T cutoff, Orders orders,
            Cube mmaps_d, Image mmaps_m, bool hermite)
    {
        const typename Kernels::DeviceScope scope(mmaps_d.device_id());
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
                hermite,
                size[0], size[1], size[2],
                step[2], zero[2],
                dcube_d.data(),
                cutoff, norders, orders.data(),
                mmaps_d.data(), mmaps_m.data());
    }

    // The light that leaves a plane of a lensed scene toward the viewer,
    // into out (nz, ny, nx), on the image plane: the light the plane emits
    // (emission, (nz, sy, sx)) plus that from behind it (behind, (nz S, ny,
    // nx)) through its transmission (transmission, (nz S, sy, sx)),
    // averaged over the S sub-channels of each channel of out, each of
    // emission and transmission at the position of each pixel on the
    // plane, in the pixel coordinates of their grid (source_x and
    // source_y, of shape (ny, nx); see gbkfit::dmodel_lens_composite).
    // Without positions, the plane is on the image grid. emission,
    // transmission and behind are optional (none, 1, none); S is the
    // number of channels of behind (or transmission) over those of out.
    // out can be behind when S is 1.
    static void
    lens_composite(
            ConstImage source_x, ConstImage source_y,
            ConstCube emission, ConstCube transmission, ConstCube behind,
            Cube out)
    {
        const typename Kernels::DeviceScope scope(out.device_id());
        const auto dst = size_xyz(out);
        require(dst[2] >= 1, "out needs a channel");
        require(source_x.is_valid() == source_y.is_valid(),
                "source_x and source_y are given together");
        if (source_x.is_valid()) {
            require(int(source_x.shape(0)) == dst[1]
                    && int(source_x.shape(1)) == dst[0],
                    "source_x must have shape (ny, nx) of out");
            require_same_shape(source_y, source_x, "source_y", "source_x");
        }
        // (the grid of the plane: that of emission or of transmission, or
        // without positions the image plane's)
        std::array<int, 3> src = dst;
        if (emission.is_valid())
            src = size_xyz(emission);
        else if (transmission.is_valid())
            src = size_xyz(transmission);
        require(source_x.is_valid() || (src[0] == dst[0] && src[1] == dst[1]),
                "without positions, the plane must be on the grid of out");
        require(!emission.is_valid() || src[2] == dst[2],
                "emission must have the channels of out");
        // The sub-channels of each channel of out, of the light from behind
        int subchannels = 1;
        if (behind.is_valid()) {
            const auto from = size_xyz(behind);
            require(from[0] == dst[0] && from[1] == dst[1]
                    && from[2] % dst[2] == 0,
                    "behind must have shape (nz S, ny, nx) of out, of S "
                    "sub-channels of each of its channels");
            subchannels = from[2] / dst[2];
            require(subchannels == 1 || behind.data() != out.data(),
                    "out can be behind only without sub-channels");
        }
        if (transmission.is_valid()) {
            const auto through = size_xyz(transmission);
            require(behind.is_valid(),
                    "a transmission needs the light from behind");
            require(through[0] == src[0] && through[1] == src[1]
                    && through[2] == dst[2] * subchannels,
                    "transmission must be on the grid of emission, with a "
                    "channel for each of behind");
        }
        Kernels::dmodel_lens_composite(
                dst[0], dst[1], dst[2], src[0], src[1], subchannels,
                data(source_x), data(source_y),
                data(emission), data(transmission), data(behind),
                out.data());
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
        const typename Kernels::DeviceScope scope(out.device_id());
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
                        ReleaseGil(),
                        nb::arg("scale"), nb::arg("offset"),
                        nb::arg("src").noconvert(),
                        nb::arg("dst").noconvert())
                .def_static("dcube_convolve_z", &dcube_convolve_z,
                        ReleaseGil(),
                        nb::arg("kernels").noconvert(),
                        nb::arg("cube").noconvert(),
                        nb::arg("scratch").noconvert())
                .def_static("mmaps_moments", &mmaps_moments,
                        ReleaseGil(),
                        nb::arg("step"), nb::arg("zero"),
                        nb::arg("dcube_d").noconvert(),
                        nb::arg("dcube_w").noconvert().none(),
                        nb::arg("cutoff"),
                        nb::arg("orders").noconvert(),
                        nb::arg("mmaps_d").noconvert(),
                        nb::arg("mmaps_m").noconvert(),
                        nb::arg("mmaps_w").noconvert().none())
                .def_static("mmaps_gaussian", &mmaps_gaussian,
                        ReleaseGil(),
                        nb::arg("step"), nb::arg("zero"),
                        nb::arg("dcube_d").noconvert(),
                        nb::arg("cutoff"),
                        nb::arg("orders").noconvert(),
                        nb::arg("mmaps_d").noconvert(),
                        nb::arg("mmaps_m").noconvert(),
                        nb::arg("hermite"))
                .def_static("lens_composite", &lens_composite,
                        ReleaseGil(),
                        nb::arg("source_x").noconvert().none(),
                        nb::arg("source_y").noconvert().none(),
                        nb::arg("emission").noconvert().none(),
                        nb::arg("transmission").noconvert().none(),
                        nb::arg("behind").noconvert().none(),
                        nb::arg("out").noconvert())
                .def_static("regions_sum", &regions_sum,
                        ReleaseGil(),
                        nb::arg("indptr").noconvert(),
                        nb::arg("indices").noconvert(),
                        nb::arg("weights").noconvert(),
                        nb::arg("cube").noconvert(),
                        nb::arg("out").noconvert());
    }
};

} // namespace gbkfit::bindings
