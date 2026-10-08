#pragma once

#include <complex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <pocketfft_hdronly.h>

#include "gbkfit/host/common.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T>
class FFT
{
public:

    using SizeType = std::array<int, 3>;
    using RealType = T;
    using ComplexType = std::complex<T>;
    using DataCacheKeyType = std::pair<SizeType, RealType*>;
    using DataCacheValueType = std::vector<ComplexType>;

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
    }

    void
    fft_r2c(SizeType size, Ptr data_r, Ptr data_c)
    {
        auto* data_r_ptr = reinterpret_cast<RealType*>(data_r);
        auto* data_c_ptr = reinterpret_cast<ComplexType*>(data_c);
        fft_r2c_exec(size, data_r_ptr, data_c_ptr);
    }

    void
    fft_c2r(SizeType size, Ptr data_c, Ptr data_r)
    {
        auto* data_c_ptr = reinterpret_cast<ComplexType*>(data_c);
        auto* data_r_ptr = reinterpret_cast<RealType*>(data_r);
        fft_c2r_exec(size, data_c_ptr, data_r_ptr);
    }

    void
    fft_convolve(
            const std::array<int, 3> size,
            Ptr data1_r, Ptr data1_c, Ptr data2_c)
    {
        auto* data1_r_ptr = reinterpret_cast<RealType*>(data1_r);
        auto* data1_c_ptr = reinterpret_cast<ComplexType*>(data1_c);
        auto* data2_c_ptr = reinterpret_cast<ComplexType*>(data2_c);
        fft_convolve_impl(size, data1_r_ptr, data1_c_ptr, data2_c_ptr);
    }

    void
    fft_convolve_cached(
            const std::array<int, 3> size,
            Ptr data1_r, Ptr data2_r)
    {
        auto* data1_r_ptr = reinterpret_cast<RealType*>(data1_r);
        auto* data2_r_ptr = reinterpret_cast<RealType*>(data2_r);
        const auto data1_key = std::pair{size, data1_r_ptr};
        const auto data2_key = std::pair{size, data2_r_ptr};
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
            fft_r2c_exec(size, data2_r_ptr, data2_c.data());
        }

        fft_convolve_impl(size, data1_r_ptr, data1_c.data(), data2_c.data());
    }

private:

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

    void
    fft_r2c_exec(SizeType size, const RealType* data_r, ComplexType* data_c)
    {
        const auto shape_r = real_shape(size);
        const auto shape_c = complex_shape(shape_r);
        pocketfft::r2c(
                shape_r,
                strides<RealType>(shape_r),
                strides<ComplexType>(shape_c),
                {0, 1, 2}, pocketfft::FORWARD,
                data_r, data_c, T{1}, nthreads());
    }

    void
    fft_c2r_exec(SizeType size, const ComplexType* data_c, RealType* data_r)
    {
        const auto shape_r = real_shape(size);
        const auto shape_c = complex_shape(shape_r);
        pocketfft::c2r(
                shape_r,
                strides<ComplexType>(shape_c),
                strides<RealType>(shape_r),
                {0, 1, 2}, pocketfft::BACKWARD,
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
};

} // namespace gbkfit::host
