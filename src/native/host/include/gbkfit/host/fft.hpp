#pragma once

#include <complex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <pocketfft_hdronly.h>

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

namespace nb = nanobind;

template<typename T>
class FFT
{
public:

    using SizeType = std::array<int, 3>;
    using RealType = T;
    using ComplexType = std::complex<T>;
    using DataCacheKeyType = std::pair<SizeType, const RealType*>;
    using DataCacheValueType = std::vector<ComplexType>;

    // Real cubes of shape (z, y, x), and the non-redundant half of
    // their spectra, of shape (z, y, x / 2 + 1)
    using Real = bindings::Array<nb::device::cpu, RealType, nb::ndim<3>>;
    using ConstReal = bindings::Array<
            nb::device::cpu, const RealType, nb::ndim<3>>;
    using Complex = bindings::Array<
            nb::device::cpu, ComplexType, nb::ndim<3>>;

    struct DataCacheKeyHashType {
        std::size_t operator()(const DataCacheKeyType& k) const {
            const auto size = k.first;
            const auto data = k.second;
            return size[0] ^ size[1] ^ size[2] ^
                    (std::uintptr_t)data;
        }
    };

    using DataCacheMappingContainer = std::unordered_map<
        DataCacheKeyType, DataCacheValueType, DataCacheKeyHashType>;

    FFT() {}

    void
    clear_cache()
    {
        m_data_cache.clear();
        m_xy_cache.clear();
    }

    void
    fft_r2c(ConstReal data_r, Complex data_c)
    {
        const auto size = bindings::size_xyz(data_r);
        require_spectrum_shape(size, data_c);
        fft_r2c_exec(size, data_r.data(), data_c.data());
    }

    void
    fft_c2r(Complex data_c, Real data_r)
    {
        const auto size = bindings::size_xyz(data_r);
        require_spectrum_shape(size, data_c);
        fft_c2r_exec(size, data_c.data(), data_r.data());
    }

    // Convolve data1_r with data2_r, in place. The spectra of data2_r
    // and of the buffer of data1_r are cached by their address.
    void
    fft_convolve_cached(Real data1_r, ConstReal data2_r)
    {
        const auto size = bindings::size_xyz(data1_r);
        bindings::require(
                bindings::size_xyz(data2_r) == size,
                "data1_r and data2_r must have the same shape");
        const auto data1_key = std::pair{size, (const RealType*)data1_r.data()};
        const auto data2_key = std::pair{size, data2_r.data()};
        const auto [n2, n1, n0] = size;
        const auto len = n0 * n1 * (n2 / 2 + 1);

        auto& data1_c = m_data_cache[data1_key];
        if (data1_c.empty())
        {
            data1_c.resize(len);
        }

        auto& data2_c = m_data_cache[data2_key];
        if (data2_c.empty())
        {
            data2_c.resize(len);
            fft_r2c_exec(size, data2_r.data(), data2_c.data());
        }

        fft_convolve_impl(
                size, data1_r.data(), data1_c.data(), data2_c.data());
    }

    // Convolve each channel (z) of data1_r with the image of the same
    // channel of data2_r, or with its only image, in place: 2D transforms
    // along y and x. The spectra are cached as by fft_convolve_cached.
    void
    fft_convolve_xy_cached(Real data1_r, ConstReal data2_r)
    {
        const auto size = bindings::size_xyz(data1_r);
        const auto size2 = bindings::size_xyz(data2_r);
        bindings::require(
                size2[0] == size[0] && size2[1] == size[1]
                && (size2[2] == size[2] || size2[2] == 1),
                "data2_r must have an image for each channel of data1_r, "
                "or one for all");
        const auto data1_key = std::pair{size, (const RealType*)data1_r.data()};
        const auto data2_key = std::pair{size2, data2_r.data()};
        const auto [nx, ny, nz] = size;
        const auto plane = ny * (nx / 2 + 1);

        auto& data1_c = m_xy_cache[data1_key];
        if (data1_c.empty())
        {
            data1_c.resize(std::size_t(nz) * plane);
        }

        auto& data2_c = m_xy_cache[data2_key];
        if (data2_c.empty())
        {
            data2_c.resize(std::size_t(size2[2]) * plane);
            fft_r2c_exec(size2, data2_r.data(), data2_c.data(), {1, 2});
        }

        fft_r2c_exec(size, data1_r.data(), data1_c.data(), {1, 2});
        const auto nfactor = T{1} / (nx * ny);
        for (int z = 0; z < nz; ++z)
        {
            // (the only image of data2_r is for all the channels)
            const auto z2 = size2[2] == 1 ? 0 : z;
            kernels::math_complex_multiply_and_scale<T>(
                    data1_c.data() + std::size_t(z) * plane,
                    data2_c.data() + std::size_t(z2) * plane,
                    plane, nfactor);
        }
        fft_c2r_exec(size, data1_c.data(), data1_r.data(), {1, 2});
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<FFT>(m, ("FFT" + suffix).c_str())
                .def(nb::init<>())
                .def("fft_r2c", &FFT::fft_r2c,
                        nb::arg("data_r").noconvert(),
                        nb::arg("data_c").noconvert())
                .def("fft_c2r", &FFT::fft_c2r,
                        nb::arg("data_c").noconvert(),
                        nb::arg("data_r").noconvert())
                .def("fft_convolve_cached", &FFT::fft_convolve_cached,
                        nb::arg("data1_r").noconvert(),
                        nb::arg("data2_r").noconvert())
                .def("fft_convolve_xy_cached", &FFT::fft_convolve_xy_cached,
                        nb::arg("data1_r").noconvert(),
                        nb::arg("data2_r").noconvert());
    }

private:

    static void
    require_spectrum_shape(SizeType size, const Complex& data_c)
    {
        bindings::require(
                bindings::size_xyz(data_c)
                        == SizeType{size[0] / 2 + 1, size[1], size[2]},
                "data_c must have shape (z, y, x / 2 + 1) of data_r");
    }

    // The size is given as (x, y, z), while the data is stored in
    // row-major order, i.e., with shape (z, y, x).
    static pocketfft::shape_t
    real_shape(SizeType size)
    {
        const auto [n2, n1, n0] = size;
        return {std::size_t(n0), std::size_t(n1), std::size_t(n2)};
    }

    // Strides (in bytes) of a row-major array with the given shape
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

    // Shape of the non-redundant half of the r2c output
    static pocketfft::shape_t
    complex_shape(const pocketfft::shape_t& shape)
    {
        return {shape[0], shape[1], shape[2] / 2 + 1};
    }

    static std::size_t
    nthreads()
    {
        return std::thread::hardware_concurrency();
    }

    // The transforms along the given axes (of the shape (z, y, x)); the
    // last one is the axis of the non-redundant half
    void
    fft_r2c_exec(
            SizeType size, const RealType* data_r, ComplexType* data_c,
            const pocketfft::shape_t& axes = {0, 1, 2})
    {
        const auto shape_r = real_shape(size);
        const auto shape_c = complex_shape(shape_r);
        pocketfft::r2c(
                shape_r,
                strides<RealType>(shape_r),
                strides<ComplexType>(shape_c),
                axes, pocketfft::FORWARD,
                data_r, data_c, T{1}, nthreads());
    }

    void
    fft_c2r_exec(
            SizeType size, const ComplexType* data_c, RealType* data_r,
            const pocketfft::shape_t& axes = {0, 1, 2})
    {
        const auto shape_r = real_shape(size);
        const auto shape_c = complex_shape(shape_r);
        pocketfft::c2r(
                shape_r,
                strides<ComplexType>(shape_c),
                strides<RealType>(shape_r),
                axes, pocketfft::BACKWARD,
                data_c, data_r, T{1}, nthreads());
    }

    void
    complex_multiply_and_scale(
            const std::array<int, 3> size,
            ComplexType* data1, const ComplexType* data2)
    {
        const auto [n2, n1, n0] = size;
        const auto n = n0 * n1 * (n2 / 2 + 1);
        const auto nfactor = T{1} / (n0 * n1 * n2);
        kernels::math_complex_multiply_and_scale<T>(data1, data2, n, nfactor);
    }

    void
    fft_convolve_impl(
            const std::array<int, 3> size,
            RealType* data1_r, ComplexType* data1_c, ComplexType* data2_c)
    {
        fft_r2c_exec(size, data1_r, data1_c);
        complex_multiply_and_scale(size, data1_c, data2_c);
        fft_c2r_exec(size, data1_c, data1_r);
    }

    DataCacheMappingContainer m_data_cache;
    // The spectra of the 2D transforms (see fft_convolve_xy_cached)
    DataCacheMappingContainer m_xy_cache;
};

} // namespace gbkfit::host
