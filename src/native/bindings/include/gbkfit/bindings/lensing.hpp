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
// bands of rings over 2 pi K, thin or thick, on the device of the arrays.
// Kernels launches them.
template<typename T, typename Device, typename Kernels>
struct Lensing
{
    using Vector = Array<Device, T, nb::ndim<1>>;
    using ConstVector = Array<Device, const T, nb::ndim<1>>;
    using ConstMatrix = Array<Device, const T, nb::ndim<2>>;
    using ConstCube = Array<Device, const T, nb::ndim<3>>;
    using ConstIndices = Array<Device, const int, nb::ndim<2>>;

    static void
    require_points(
            const ConstVector& x, const ConstVector& y,
            const Vector& alpha_x, const Vector& alpha_y)
    {
        require_same_shape(y, x, "y", "x");
        require_same_shape(alpha_x, x, "alpha_x", "x");
        require_same_shape(alpha_y, x, "alpha_y", "x");
    }

    // The vertical structure of a layer (see gbkfit::LensHeights): of its
    // knots, the shares of its terms there (terms, knots), the tables of
    // their pdfs (terms, knots, n), their tops and scale heights (thin, of
    // no terms), and the Gauss-Legendre rules of 1 to m nodes (m (m + 1) /
    // 2 nodes and weights)
    static LensHeights<T>
    heights(
            const ConstVector& knots, const ConstMatrix& shares,
            const ConstCube& tables, const ConstVector& tops,
            const ConstVector& scales, const ConstVector& gauss_nodes,
            const ConstVector& gauss_weights)
    {
        size_t m = 0;
        while ((m + 1) * (m + 2) / 2 <= gauss_nodes.shape(0)) {
            ++m;
        }
        require(m * (m + 1) / 2 == gauss_nodes.shape(0)
                && gauss_weights.shape(0) == gauss_nodes.shape(0)
                && m >= size_t(LENS_HEIGHT_FEWEST),
                "gauss_nodes and gauss_weights must hold the rules of 1 "
                "to m nodes, m >= " + std::to_string(LENS_HEIGHT_FEWEST));
        const LensGauss<T> gauss{
                int(m), gauss_nodes.data(), gauss_weights.data()};
        const size_t nterms = shares.shape(0), nknots = knots.shape(0);
        if (nterms == 0) {
            LensHeights<T> thin;
            thin.gauss = gauss;
            return thin;
        }
        require(nterms <= size_t(LENS_HEIGHT_TERMS),
                "a thick layer has at most "
                + std::to_string(LENS_HEIGHT_TERMS) + " terms");
        require(nknots >= 1 && shares.shape(1) == nknots,
                "shares must have shape (terms, knots)");
        require(tables.shape(0) == nterms && tables.shape(1) == nknots
                && tables.shape(2) >= 4,
                "tables must have shape (terms, knots, n), n >= 4");
        require(tops.shape(0) == nterms && scales.shape(0) == nterms,
                "tops and scales must have a value for each term");
        return {int(nterms), int(nknots), knots.data(), shares.data(),
                int(tables.shape(2)), tables.data(), tops.data(),
                scales.data(), gauss};
    }

    // The deflection of an elliptical layer (see
    // gbkfit::lens_elliptical_point) at the points (x, y), into alpha_x
    // and alpha_y: of its frame (the sine and cosine of its position angle,
    // its axis ratio and sigma1), its centre, the normal of its plane, the
    // table of its surface density of the first radius exp(start) and the
    // step of ln R, the nodes of the quadrature (5, nnodes; see
    // gbkfit::LensNodes) and its vertical structure (see heights)
    static void
    elliptical_deflection(
            ConstVector x, ConstVector y,
            std::array<T, 4> frame, std::array<T, 2> centre,
            std::array<T, 2> normal, ConstVector table, T start, T step,
            ConstMatrix nodes, ConstVector knots, ConstMatrix shares,
            ConstCube tables, ConstVector tops, ConstVector scales,
            ConstVector gauss_nodes, ConstVector gauss_weights,
            Vector alpha_x, Vector alpha_y)
    {
        const typename Kernels::DeviceScope scope(alpha_x.device_id());
        require_points(x, y, alpha_x, alpha_y);
        require(table.shape(0) >= 4, "a table needs 4 values at least");
        require(nodes.shape(0) == 5, "nodes must have 5 rows");
        require(frame[3] > 0, "sigma1 must be positive");
        const LensEllipse<T> ellipse{
                frame[0], frame[1], frame[2], frame[3],
                centre[0], centre[1], normal[0], normal[1],
                int(table.shape(0)), start, step, table.data(),
                int(nodes.shape(1)), nodes.data(),
                heights(knots, shares, tables, tops, scales, gauss_nodes,
                        gauss_weights)};
        Kernels::lens_elliptical_deflection(
                int(x.shape(0)), x.data(), y.data(), ellipse,
                alpha_x.data(), alpha_y.data());
    }

    // The deflection of bands of rings (see gbkfit::lens_band_point) at
    // the points (x, y), into alpha_x and alpha_y: of their centres (bands,
    // 2), frames (bands, 5), normals (bands, 2), coefficients (bands, 2, 3)
    // and the indices of their knots (bands, 2), and their vertical
    // structure (see heights)
    static void
    band_deflection(
            ConstVector x, ConstVector y,
            ConstMatrix centres, ConstMatrix frames, ConstMatrix normals,
            ConstCube coefficients, ConstIndices band_knots,
            ConstVector knots, ConstMatrix shares, ConstCube tables,
            ConstVector tops, ConstVector scales,
            ConstVector gauss_nodes, ConstVector gauss_weights,
            Vector alpha_x, Vector alpha_y)
    {
        const typename Kernels::DeviceScope scope(alpha_x.device_id());
        require_points(x, y, alpha_x, alpha_y);
        const size_t nbands = frames.shape(0);
        require(frames.shape(1) == 5, "frames must have shape (bands, 5)");
        require(centres.shape(0) == nbands && centres.shape(1) == 2
                && normals.shape(0) == nbands && normals.shape(1) == 2
                && band_knots.shape(0) == nbands
                && band_knots.shape(1) == 2,
                "centres, normals and band_knots must have shape "
                "(bands, 2)");
        require(coefficients.shape(0) == nbands
                && coefficients.shape(1) == 2
                && coefficients.shape(2) == 3,
                "coefficients must have shape (bands, 2, 3)");
        const LensBands<T> bands{
                int(nbands), centres.data(), frames.data(), normals.data(),
                coefficients.data(), band_knots.data(),
                heights(knots, shares, tables, tops, scales, gauss_nodes,
                        gauss_weights)};
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
                        nb::arg("table").noconvert(),
                        nb::arg("start"), nb::arg("step"),
                        nb::arg("nodes").noconvert(),
                        nb::arg("knots").noconvert(),
                        nb::arg("shares").noconvert(),
                        nb::arg("tables").noconvert(),
                        nb::arg("tops").noconvert(),
                        nb::arg("scales").noconvert(),
                        nb::arg("gauss_nodes").noconvert(),
                        nb::arg("gauss_weights").noconvert(),
                        nb::arg("alpha_x").noconvert(),
                        nb::arg("alpha_y").noconvert())
                .def_static("band_deflection", &band_deflection,
                        ReleaseGil(),
                        nb::arg("x").noconvert(), nb::arg("y").noconvert(),
                        nb::arg("centres").noconvert(),
                        nb::arg("frames").noconvert(),
                        nb::arg("normals").noconvert(),
                        nb::arg("coefficients").noconvert(),
                        nb::arg("band_knots").noconvert(),
                        nb::arg("knots").noconvert(),
                        nb::arg("shares").noconvert(),
                        nb::arg("tables").noconvert(),
                        nb::arg("tops").noconvert(),
                        nb::arg("scales").noconvert(),
                        nb::arg("gauss_nodes").noconvert(),
                        nb::arg("gauss_weights").noconvert(),
                        nb::arg("alpha_x").noconvert(),
                        nb::arg("alpha_y").noconvert());
    }
};

} // namespace gbkfit::bindings
