#pragma once

#include "gbkfit/cuda/common.hpp"
#include "gbkfit/cuda/fftutils.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T>
class FFT
{
public:

    enum class FFTType { R2C, C2R };

    using SizeType = std::array<int, 3>;
    using PlanType = cufftHandle;
    using RealType = typename cufft<T>::real;
    using ComplexType = typename cufft<T>::complex;
    using DataCacheKeyType = std::pair<SizeType, RealType*>;
    using DataCacheValueType = ComplexType*;
    using PlanCacheKeyType = std::pair<SizeType, FFTType>;
    using PlanCacheValueType = PlanType;

    struct DataCacheKeyHashType {
        std::size_t operator()(const DataCacheKeyType& k) const {
            const auto size = k.first;
            const auto data = k.second;
            return size[0] ^ size[1] ^ size[2] ^
                    (std::uintptr_t)data;
        }
    };

    struct PlanCacheKeyHashType {
        std::size_t operator()(const PlanCacheKeyType& k) const {
            const auto size = k.first;
            const auto type = k.second;
            return size[0] ^ size[1] ^ size[2] ^
                    static_cast<std::underlying_type_t<FFTType>>(type);
        }
    };

    using DataCacheMappingContainer = std::unordered_map<
        DataCacheKeyType, DataCacheValueType, DataCacheKeyHashType>;

    using PlanCacheMappingContainer = std::unordered_map<
        PlanCacheKeyType, PlanCacheValueType, PlanCacheKeyHashType>;

    FFT() {}

    ~FFT()
    {
        clear_cache();
    }

    void
    clear_cache()
    {
        for(auto& [_, data] : m_data_cache) {
            cudaFree(data);
        }
        m_data_cache.clear();

        for(auto& [_, plan] : m_plan_cache) {
            cufftDestroy(plan);
        }
        m_plan_cache.clear();
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
        const auto len = n0 * n1 * n2;

        if (m_data_cache.find(data1_key) == m_data_cache.end())
        {
            cudaMalloc((void**)&m_data_cache[data1_key], len * sizeof(ComplexType));
        }

        if (m_data_cache.find(data2_key) == m_data_cache.end())
        {
            cudaMalloc((void**)&m_data_cache[data2_key], len * sizeof(ComplexType));
            fft_r2c_exec(size, data2_r_ptr, m_data_cache[data2_key]);
        }

        auto* data1_c_ptr = m_data_cache[data1_key];
        auto* data2_c_ptr = m_data_cache[data2_key];

        fft_convolve_impl(size, data1_r_ptr, data1_c_ptr, data2_c_ptr);
    }

private:

    PlanType
    fft_r2c_plan(SizeType size, RealType* data_r, ComplexType* data_c)
    {
        auto [n2, n1, n0] = size;
        cufftHandle plan;
        cufftPlan3d(&plan, n0, n1, n2, cufft<T>::R2C);
        return plan;
    }

    PlanType
    fft_c2r_plan(SizeType size, ComplexType* data_c, RealType* data_r)
    {
        auto [n2, n1, n0] = size;
        cufftHandle plan;
        cufftPlan3d(&plan, n0, n1, n2, cufft<T>::C2R);
        return plan;
    }

    void
    fft_r2c_exec(SizeType size, RealType* data_r, ComplexType* data_c)
    {
        const auto key = std::pair{size, FFTType::R2C};

        if (m_plan_cache.find(key) == m_plan_cache.end())
        {
            m_plan_cache[key] = fft_r2c_plan(size, data_r, data_c);
        }

        cufft<T>::execR2C(m_plan_cache[key], data_r, data_c);
    }

    void
    fft_c2r_exec(SizeType size, ComplexType* data_c, RealType* data_r)
    {
        const auto key = std::pair{size, FFTType::C2R};

        if (m_plan_cache.find(key) == m_plan_cache.end())
        {
            m_plan_cache[key] = fft_c2r_plan(size, data_c, data_r);
        }

        cufft<T>::execC2R(m_plan_cache[key], data_c, data_r);
    }

    void
    complex_multiply_and_scale(
            const std::array<int, 3> size,
            ComplexType* data1, ComplexType* data2)
    {
        const auto [n2, n0, n1] = size;
        const auto n = n0 * n1 * (n2 / 2 + 1);
        const auto nfactor = T{1} / (n0 * n1 * n2);
        Wrapper<T>::math_complex_multiply_and_scale(
                reinterpret_cast<typename cufft<T>::complex*>(data1),
                reinterpret_cast<typename cufft<T>::complex*>(data2),
                n, nfactor);
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
    PlanCacheMappingContainer m_plan_cache;
};

} // namespace gbkfit::cuda
