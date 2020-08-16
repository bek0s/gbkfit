#pragma once

#include <cufft.h>

namespace gbkfit {

template<typename T>
struct cufft;

template<>
struct cufft<float>
{
    using real = cufftReal;
    using complex = cufftComplex;
    static constexpr auto R2C = CUFFT_R2C;
    static constexpr auto C2R = CUFFT_C2R;
    static constexpr auto execR2C = ::cufftExecR2C;
    static constexpr auto execC2R = ::cufftExecC2R;
};

template<>
struct cufft<double>
{
    using real = cufftDoubleReal;
    using complex = cufftDoubleComplex;
    static constexpr auto R2C = CUFFT_D2Z;
    static constexpr auto C2R = CUFFT_Z2D;
    static constexpr auto execR2C = ::cufftExecD2Z;
    static constexpr auto execC2R = ::cufftExecZ2D;
};

} // namespace gbkfit
