#pragma once

#include <array>
#include <complex>
#include <string>
#include <thread>
#include <vector>

#include <pocketfft_hdronly.h>

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

namespace nb = nanobind;

// The FFT convolution of real cubes of shape (z, y, x) by a kernel, in
// place: of the whole cubes (axes {0, 1, 2}), or of each channel (z) by
// its image (axes {1, 2}; an image for each channel, or one for all).
// The spectrum of the kernel is made once, and the kernel is not kept;
// the spectra of the cubes go in one buffer, kept for the next cube.
template<typename T>
class FFTConvolution
{
public:

    using RealType = T;
    using ComplexType = std::complex<T>;

    using Cube = bindings::Array<nb::device::cpu, RealType, nb::ndim<3>>;
    using Kernel = bindings::Array<
            nb::device::cpu, const RealType, nb::ndim<3>>;

    FFTConvolution(
            Kernel kernel, std::array<int, 3> shape, pocketfft::shape_t axes)
        : m_shape{std::size_t(shape[0]), std::size_t(shape[1]),
                  std::size_t(shape[2])}
        , m_axes(axes)
    {
        const auto whole = axes == pocketfft::shape_t{0, 1, 2};
        bindings::require(
                whole || axes == pocketfft::shape_t{1, 2},
                "axes must be (0, 1, 2) or (1, 2)");
        const pocketfft::shape_t kernel_shape{
                kernel.shape(0), kernel.shape(1), kernel.shape(2)};
        bindings::require(
                kernel_shape == m_shape || (!whole
                        && kernel_shape[0] == 1
                        && kernel_shape[1] == m_shape[1]
                        && kernel_shape[2] == m_shape[2]),
                "the kernel must have the shape of the cubes, or one image "
                "for all their channels (axes (1, 2))");
        m_nfactor = T{1} / (whole
                ? m_shape[0] * m_shape[1] * m_shape[2]
                : m_shape[1] * m_shape[2]);
        m_kernel_spectrum.resize(size(complex_shape(kernel_shape)));
        m_spectrum.resize(size(complex_shape(m_shape)));
        r2c(kernel_shape, kernel.data(), m_kernel_spectrum.data());
    }

    void
    convolve(Cube cube)
    {
        bindings::require(
                pocketfft::shape_t{cube.shape(0), cube.shape(1),
                                   cube.shape(2)} == m_shape,
                "the cube must have the shape of the convolution");
        r2c(m_shape, cube.data(), m_spectrum.data());
        // (one image of the kernel is for all the channels)
        const auto channels = m_shape[0];
        const auto plane = m_spectrum.size() / channels;
        if (m_kernel_spectrum.size() == m_spectrum.size())
        {
            kernels::math_complex_multiply_and_scale<T>(
                    m_spectrum.data(), m_kernel_spectrum.data(),
                    int(m_spectrum.size()), m_nfactor);
        }
        else
        {
            for (std::size_t z = 0; z < channels; ++z)
            {
                kernels::math_complex_multiply_and_scale<T>(
                        m_spectrum.data() + z * plane,
                        m_kernel_spectrum.data(), int(plane), m_nfactor);
            }
        }
        c2r(m_spectrum.data(), cube.data());
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<FFTConvolution>(m, ("FFTConvolution" + suffix).c_str())
                .def(nb::init<
                                Kernel, std::array<int, 3>,
                                pocketfft::shape_t>(),
                        nb::arg("kernel").noconvert(),
                        nb::arg("shape"),
                        nb::arg("axes"))
                .def("__call__", &FFTConvolution::convolve,
                        nb::arg("cube").noconvert());
    }

private:

    static std::size_t
    size(const pocketfft::shape_t& shape)
    {
        return shape[0] * shape[1] * shape[2];
    }

    // The shape of the non-redundant half of the spectrum of a real array
    static pocketfft::shape_t
    complex_shape(const pocketfft::shape_t& shape)
    {
        return {shape[0], shape[1], shape[2] / 2 + 1};
    }

    // The strides (in bytes) of a row-major array of a shape
    template<typename U>
    static pocketfft::stride_t
    strides(const pocketfft::shape_t& shape)
    {
        const auto row = shape[2] * sizeof(U);
        const auto plane = shape[1] * row;
        return {
            std::ptrdiff_t(plane),
            std::ptrdiff_t(row),
            std::ptrdiff_t(sizeof(U))};
    }

    static std::size_t
    nthreads()
    {
        return std::thread::hardware_concurrency();
    }

    // The transform of a real array of a shape along the axes (the last
    // one is the axis of the non-redundant half)
    void
    r2c(const pocketfft::shape_t& shape,
        const RealType* data_r, ComplexType* data_c) const
    {
        pocketfft::r2c(
                shape, strides<RealType>(shape),
                strides<ComplexType>(complex_shape(shape)),
                m_axes, pocketfft::FORWARD, data_r, data_c, T{1}, nthreads());
    }

    // The inverse of r2c (unnormalised) for the cubes: along the axes but
    // the last in place, then along the last one into the real cube
    // (pocketfft's c2r along several axes copies the spectrum first)
    void
    c2r(ComplexType* data_c, RealType* data_r) const
    {
        const auto shape_c = complex_shape(m_shape);
        const auto strides_c = strides<ComplexType>(shape_c);
        const pocketfft::shape_t others(m_axes.begin(), m_axes.end() - 1);
        pocketfft::c2c(
                shape_c, strides_c, strides_c, others, pocketfft::BACKWARD,
                data_c, data_c, T{1}, nthreads());
        pocketfft::c2r(
                m_shape, strides_c, strides<RealType>(m_shape),
                m_axes.back(), pocketfft::BACKWARD, data_c, data_r, T{1},
                nthreads());
    }

    pocketfft::shape_t m_shape;
    pocketfft::shape_t m_axes;
    T m_nfactor;
    std::vector<ComplexType> m_kernel_spectrum;
    std::vector<ComplexType> m_spectrum;
};

} // namespace gbkfit::host
