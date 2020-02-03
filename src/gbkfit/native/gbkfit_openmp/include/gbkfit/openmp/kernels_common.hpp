#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

template<typename T>
struct RNG
{
    RNG(T a, T b)
        : gen(std::random_device()())
        , dis(a, b) {}

    T operator ()(void) {
        return dis(gen);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<T> dis;
};
