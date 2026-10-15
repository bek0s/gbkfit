#pragma once

#include <array>
#include <complex>
#include <string>
#include <thread>
#include <type_traits>
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
// The cubes are of type T, and the transforms of type U: T, or a more
// precise one; then each cube is copied into the buffer, as real rows
// padded to the length of its complex rows, transformed there in place,
// and copied back.
template<typename T, typename U>
class FFTConvolution
{
public:

    using RealType = U;
    using ComplexType = std::complex<U>;

    using Cube = bindings::Array<nb::device::cpu, T, nb::ndim<3>>;
    using Kernel = bindings::Array<nb::device::cpu, const U, nb::ndim<3>>;

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
        m_nfactor = U{1} / (whole
                ? m_shape[0] * m_shape[1] * m_shape[2]
                : m_shape[1] * m_shape[2]);
        m_kernel_spectrum.resize(size(complex_shape(kernel_shape)));
        m_spectrum.resize(size(complex_shape(m_shape)));
        r2c(kernel_shape, real_strides(kernel_shape, kernel_shape[2]),
            kernel.data(), m_kernel_spectrum.data());
    }

    void
    convolve(Cube cube)
    {
        bindings::require(
                pocketfft::shape_t{cube.shape(0), cube.shape(1),
                                   cube.shape(2)} == m_shape,
                "the cube must have the shape of the convolution");
        const auto columns = m_shape[2];
        if constexpr (std::is_same_v<T, U>)
        {
            const auto strides = real_strides(m_shape, columns);
            r2c(m_shape, strides, cube.data(), m_spectrum.data());
            multiply();
            c2r(strides, cube.data());
        }
        else
        {
            const auto padded = 2 * (columns / 2 + 1);
            const auto strides = real_strides(m_shape, padded);
            const auto buffer = reinterpret_cast<RealType*>(
                    m_spectrum.data());
            copy_rows(cube.data(), columns, buffer, padded);
            r2c(m_shape, strides, buffer, m_spectrum.data());
            multiply();
            c2r(strides, buffer);
            copy_rows(buffer, padded, cube.data(), columns);
        }
    }

    static void
    bind(nb::module_& m, const std::string& name)
    {
        nb::class_<FFTConvolution>(m, name.c_str())
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

    // The strides (in bytes) of a row-major array of a shape whose rows
    // are row values apart
    template<typename V>
    static pocketfft::stride_t
    strides(const pocketfft::shape_t& shape, std::size_t row)
    {
        return {
            std::ptrdiff_t(shape[1] * row * sizeof(V)),
            std::ptrdiff_t(row * sizeof(V)),
            std::ptrdiff_t(sizeof(V))};
    }

    static pocketfft::stride_t
    real_strides(const pocketfft::shape_t& shape, std::size_t row)
    {
        return strides<RealType>(shape, row);
    }

    static pocketfft::stride_t
    complex_strides(const pocketfft::shape_t& shape)
    {
        const auto shape_c = complex_shape(shape);
        return strides<ComplexType>(shape_c, shape_c[2]);
    }

    static std::size_t
    nthreads()
    {
        return std::thread::hardware_concurrency();
    }

    // Copy the rows of a cube of the shape between two arrays whose rows
    // are src_row and dst_row values apart, converting their type
    template<typename From, typename To>
    void
    copy_rows(
            const From* src, std::size_t src_row,
            To* dst, std::size_t dst_row) const
    {
        const auto rows = m_shape[0] * m_shape[1];
        const auto columns = m_shape[2];
        // Parallelization: per row
        #pragma omp parallel for
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t x = 0; x < columns; ++x)
            {
                dst[i * dst_row + x] = To(src[i * src_row + x]);
            }
        }
    }

    // The transform of a real array of a shape along the axes (the last
    // one is the axis of the non-redundant half); in place if data_r is
    // in the memory of data_c, as rows padded to their length
    void
    r2c(const pocketfft::shape_t& shape, const pocketfft::stride_t& strides_r,
        const RealType* data_r, ComplexType* data_c) const
    {
        pocketfft::r2c(
                shape, strides_r, complex_strides(shape), m_axes,
                pocketfft::FORWARD, data_r, data_c, U{1}, nthreads());
    }

    // The product of the spectrum with that of the kernel, scaled by the
    // normalisation of the inverse transform (one image of the kernel is
    // for all the channels)
    void
    multiply()
    {
        if (m_kernel_spectrum.size() == m_spectrum.size())
        {
            kernels::math_complex_multiply_and_scale<U>(
                    m_spectrum.data(), m_kernel_spectrum.data(),
                    int(m_spectrum.size()), m_nfactor);
            return;
        }
        const auto channels = m_shape[0];
        const auto plane = m_spectrum.size() / channels;
        for (std::size_t z = 0; z < channels; ++z)
        {
            kernels::math_complex_multiply_and_scale<U>(
                    m_spectrum.data() + z * plane,
                    m_kernel_spectrum.data(), int(plane), m_nfactor);
        }
    }

    // The inverse of r2c (unnormalised) for the cubes: along the axes but
    // the last in place, then along the last one into the real array
    // (pocketfft's c2r along several axes copies the spectrum first)
    void
    c2r(const pocketfft::stride_t& strides_r, RealType* data_r)
    {
        const auto strides_c = complex_strides(m_shape);
        const pocketfft::shape_t others(m_axes.begin(), m_axes.end() - 1);
        pocketfft::c2c(
                complex_shape(m_shape), strides_c, strides_c, others,
                pocketfft::BACKWARD, m_spectrum.data(), m_spectrum.data(),
                U{1}, nthreads());
        pocketfft::c2r(
                m_shape, strides_c, strides_r, m_axes.back(),
                pocketfft::BACKWARD, m_spectrum.data(), data_r, U{1},
                nthreads());
    }

    pocketfft::shape_t m_shape;
    pocketfft::shape_t m_axes;
    U m_nfactor;
    std::vector<ComplexType> m_kernel_spectrum;
    std::vector<ComplexType> m_spectrum;
};

} // namespace gbkfit::host
