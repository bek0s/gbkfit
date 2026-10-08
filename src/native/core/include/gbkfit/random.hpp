#pragma once

#include <cstdint>
#include <type_traits>

namespace gbkfit {

// Philox4x32-10, the counter-based random number generator of
// Salmon et al. (2011), as in the Random123 library and cuRAND.
// It maps a 128-bit counter and a 64-bit key to 128 random bits.
struct Philox4x32
{
    static constexpr std::uint32_t M0 = 0xD2511F53;
    static constexpr std::uint32_t M1 = 0xCD9E8D57;
    static constexpr std::uint32_t W0 = 0x9E3779B9;
    static constexpr std::uint32_t W1 = 0xBB67AE85;

    std::uint32_t values[4];

    constexpr Philox4x32(const std::uint32_t (&counter)[4],
                         std::uint32_t key0, std::uint32_t key1)
        : values{counter[0], counter[1], counter[2], counter[3]}
    {
        round(key0, key1);
        for (int i = 1; i < 10; ++i)
        {
            key0 += W0;
            key1 += W1;
            round(key0, key1);
        }
    }

private:

    constexpr void
    round(std::uint32_t key0, std::uint32_t key1)
    {
        const std::uint64_t product0 = std::uint64_t{M0} * values[0];
        const std::uint64_t product1 = std::uint64_t{M1} * values[2];
        const auto hi0 = std::uint32_t(product0 >> 32);
        const auto lo0 = std::uint32_t(product0);
        const auto hi1 = std::uint32_t(product1 >> 32);
        const auto lo1 = std::uint32_t(product1);
        const std::uint32_t c1 = values[1];
        const std::uint32_t c3 = values[3];
        values[0] = hi1 ^ c1 ^ key0;
        values[1] = lo1;
        values[2] = hi0 ^ c3 ^ key1;
        values[3] = lo0;
    }
};

// A stream of uniform random numbers in the open interval (0, 1).
//
// Each stream is identified by a seed and a stream index (e.g., the
// index of a Monte Carlo cloud). Because the numbers are a function of
// the seed, the stream index, and the position in the stream only, the
// results do not depend on how the work is split between threads, and
// the host and cuda code draw the same numbers.
template<typename T>
struct RNG
{
    static_assert(std::is_floating_point_v<T>);

    constexpr RNG(std::uint32_t seed, std::uint32_t stream)
        : seed(seed), stream(stream) {}

    constexpr T
    operator()(void)
    {
        if constexpr (sizeof(T) <= 4) {
            // 24 random bits, centred in their interval: never 0 or 1
            return (T(next() >> 8) + T{0.5}) * T{0x1p-24};
        } else {
            // 53 random bits, centred in their interval: never 0 or 1
            const std::uint64_t hi = next() >> 5;
            const std::uint64_t lo = next() >> 6;
            return (T((hi << 26) | lo) + T{0.5}) * T{0x1p-53};
        }
    }

private:

    constexpr std::uint32_t
    next(void)
    {
        if (index == 4)
        {
            const std::uint32_t counter[4] = {block++, 0, stream, 0};
            const Philox4x32 philox(counter, seed, 0);
            for (int i = 0; i < 4; ++i)
                buffer[i] = philox.values[i];
            index = 0;
        }
        return buffer[index++];
    }

    std::uint32_t seed;
    std::uint32_t stream;
    std::uint32_t block = 0;
    std::uint32_t buffer[4] = {0, 0, 0, 0};
    int index = 4;
};

} // namespace gbkfit
