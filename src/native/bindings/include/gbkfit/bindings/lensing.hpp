#pragma once

#include <array>
#include <string>

#include <gbkfit/lensing/layers.hpp>

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/bindings/gil.hpp"

namespace gbkfit::bindings {

// The deflections of the layers of the mass of a galaxy laid out on the
// sky (see gbkfit.lensing._layers), at points of its plane (kpc), without
// the factors of K = 4 G / c^2: of an elliptical layer over pi K, and of
// bands of rings over 2 pi K, on the device of the arrays. Kernels
// launches them.
template<typename T, typename Device, typename Kernels>
struct Lensing
{
    using Vector = Array<Device, T, nb::ndim<1>>;
    using ConstVector = Array<Device, const T, nb::ndim<1>>;
    using ConstMatrix = Array<Device, const T, nb::ndim<2>>;
    using ConstCube = Array<Device, const T, nb::ndim<3>>;

    static void
    require_points(
            const ConstVector& x, const ConstVector& y,
            const Vector& alpha_x, const Vector& alpha_y)
    {
        require_same_shape(y, x, "y", "x");
        require_same_shape(alpha_x, x, "alpha_x", "x");
        require_same_shape(alpha_y, x, "alpha_y", "x");
    }

    // The deflection of an elliptical layer (see
    // gbkfit::lens_elliptical_point) at the points (x, y), into alpha_x
    // and alpha_y: of its frame (the sine and cosine of its position angle,
    // its axis ratio and sigma1), its centre, the normal of its plane, the
    // tables of its slices (slices, n) of the first radius exp(start) and
    // the step of ln R, their heights, and the nodes of the quadrature (5,
    // nnodes; see gbkfit::LensNodes)
    static void
    elliptical_deflection(
            ConstVector x, ConstVector y,
            std::array<T, 4> frame, std::array<T, 2> centre,
            std::array<T, 2> normal, ConstMatrix tables, T start, T step,
            ConstVector heights, ConstMatrix nodes,
            Vector alpha_x, Vector alpha_y)
    {
        const typename Kernels::DeviceScope scope(alpha_x.device_id());
        require_points(x, y, alpha_x, alpha_y);
        require(tables.shape(1) >= 4, "a table needs 4 values at least");
        require(heights.shape(0) == tables.shape(0),
                "heights must have a value for each row of tables");
        require(nodes.shape(0) == 5, "nodes must have 5 rows");
        require(frame[3] > 0, "sigma1 must be positive");
        const LensEllipse<T> ellipse{
                frame[0], frame[1], frame[2], frame[3],
                centre[0], centre[1], normal[0], normal[1],
                int(tables.shape(0)), int(tables.shape(1)), start, step,
                tables.data(), heights.data(),
                int(nodes.shape(1)), nodes.data()};
        Kernels::lens_elliptical_deflection(
                int(x.shape(0)), x.data(), y.data(), ellipse,
                alpha_x.data(), alpha_y.data());
    }

    // The deflection of bands of rings (see gbkfit::lens_band_point) at
    // the points (x, y), into alpha_x and alpha_y: of their centres (bands,
    // 2), frames (bands, 5), normals (bands, 2), the coefficients of their
    // slices (slices, bands, 3) and the heights of the slices
    static void
    band_deflection(
            ConstVector x, ConstVector y,
            ConstMatrix centres, ConstMatrix frames, ConstMatrix normals,
            ConstCube coefficients, ConstVector heights,
            Vector alpha_x, Vector alpha_y)
    {
        const typename Kernels::DeviceScope scope(alpha_x.device_id());
        require_points(x, y, alpha_x, alpha_y);
        const size_t nbands = frames.shape(0);
        require(frames.shape(1) == 5, "frames must have shape (bands, 5)");
        require(centres.shape(0) == nbands && centres.shape(1) == 2
                && normals.shape(0) == nbands && normals.shape(1) == 2,
                "centres and normals must have shape (bands, 2)");
        require(coefficients.shape(1) == nbands
                && coefficients.shape(2) == 3,
                "coefficients must have shape (slices, bands, 3)");
        require(heights.shape(0) == coefficients.shape(0),
                "heights must have a value for each slice of coefficients");
        const LensBands<T> bands{
                int(nbands), centres.data(), frames.data(), normals.data(),
                int(coefficients.shape(0)), coefficients.data(),
                heights.data()};
        Kernels::lens_band_deflection(
                int(x.shape(0)), x.data(), y.data(), bands,
                alpha_x.data(), alpha_y.data());
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<Lensing>(m, ("Lensing" + suffix).c_str())
                .def(nb::init<>())
                .def_static("elliptical_deflection", &elliptical_deflection,
                        ReleaseGil(),
                        nb::arg("x").noconvert(), nb::arg("y").noconvert(),
                        nb::arg("frame"), nb::arg("centre"),
                        nb::arg("normal"),
                        nb::arg("tables").noconvert(),
                        nb::arg("start"), nb::arg("step"),
                        nb::arg("heights").noconvert(),
                        nb::arg("nodes").noconvert(),
                        nb::arg("alpha_x").noconvert(),
                        nb::arg("alpha_y").noconvert())
                .def_static("band_deflection", &band_deflection,
                        ReleaseGil(),
                        nb::arg("x").noconvert(), nb::arg("y").noconvert(),
                        nb::arg("centres").noconvert(),
                        nb::arg("frames").noconvert(),
                        nb::arg("normals").noconvert(),
                        nb::arg("coefficients").noconvert(),
                        nb::arg("heights").noconvert(),
                        nb::arg("alpha_x").noconvert(),
                        nb::arg("alpha_y").noconvert());
    }
};

} // namespace gbkfit::bindings
