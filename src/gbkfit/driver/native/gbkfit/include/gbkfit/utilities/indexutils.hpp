#pragma once

namespace gbkfit
{

template<typename T> constexpr std::tuple<T, T>
index_1d_to_2d(T idx, T dim0)
{
    static_assert(std::is_integral_v<T>);

    auto idx1 = idx / dim0;
    idx -= idx1 * dim0;

    auto idx0 = idx;

    return {idx0, idx1};
}

template<typename T> constexpr void
index_1d_to_2d(T& idx0, T& idx1, T idx, T dim0)
{
    static_assert(std::is_integral_v<T>);

    idx1 = idx / dim0;
    idx -= idx1 * dim0;

    idx0 = idx;
}

template<typename T> constexpr std::tuple<T, T, T>
index_1d_to_3d(T idx, T dim0, T dim1)
{
    static_assert(std::is_integral_v<T>);

    auto idx2 = idx / (dim0 * dim1);
    idx -= idx2 * dim0 * dim1;

    auto idx1 = idx / dim0;
    idx -= idx1 * dim0;

    auto idx0 = idx;

    return {idx0, idx1, idx2};
}

template<typename T> constexpr void
index_1d_to_3d(T& idx0, T& idx1, T&idx2, T idx, T dim0, T dim1)
{
    static_assert(std::is_integral_v<T>);

    idx2 = idx / (dim0 * dim1);
    idx -= idx2 * dim0 * dim1;

    idx1 = idx / dim0;
    idx -= idx1 * dim0;

    idx0 = idx;
}

template<typename T> constexpr T
index_2d_to_1d(T idx0, T idx1, T dim0)
{
    static_assert(std::is_integral_v<T>);

    return idx0 + idx1 * dim0;
}

template<typename T> constexpr T
index_3d_to_1d(T idx0, T idx1, T idx2, T dim0, T dim1)
{
    static_assert(std::is_integral_v<T>);

    return idx0 + idx1 * dim0 + idx2 * dim0 * dim1;
}

} // namespace gbkfit
