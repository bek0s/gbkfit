#pragma once

#include <random>

// This is on purpose under the namespace ::gbkfit and not ::gbkfit::host
// This hack allow us to pass any Random Number Generator type down to all the
// functions that need one without using a template parameter.
namespace gbkfit {

template<typename T>
struct RNG
{
    RNG(T a, T b, int seed)
        : dis(a, b)
        , gen(seed) {}

    T
    operator()(void) {
        return dis(gen);
    }

    std::uniform_real_distribution<T> dis;
    std::mt19937 gen;
};

} // namespace gbkfit
