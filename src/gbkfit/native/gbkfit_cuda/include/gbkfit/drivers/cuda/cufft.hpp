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
    static constexpr auto cufftTypeR2C = CUFFT_R2C;
    static constexpr auto cufftTypeC2R = CUFFT_C2R;
    static constexpr auto cufftExecR2C = ::cufftExecR2C;
    static constexpr auto cufftExecC2R = ::cufftExecC2R;
};

template<>
struct cufft<double>
{
    using real = cufftDoubleReal;
    using complex = cufftDoubleComplex;
    static constexpr auto cufftTypeR2C = CUFFT_D2Z;
    static constexpr auto cufftTypeC2R = CUFFT_Z2D;
    static constexpr auto cufftExecR2C = ::cufftExecD2Z;
    static constexpr auto cufftExecC2R = ::cufftExecZ2D;
};

} // namespace gbkfit
