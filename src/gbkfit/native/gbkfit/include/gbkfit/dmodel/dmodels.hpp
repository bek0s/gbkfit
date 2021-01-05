#pragma once

namespace gbkfit {

template<typename T> constexpr void
moments(int x, int y,
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* cube,
        T* mmaps, T* masks, const int* orders, int norders)
{
    (void)step_x;
    (void)step_y;
    (void)zero_x;
    (void)zero_y;

    // Orders are sorted
    int max_order = orders[norders - 1];

    int m = 0;
    T m0=0, m1=0, m2=0, m0_sum=0, m1_sum=0, m2_sum=0;

    // Moment 0
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = cube[idx];
        m0_sum += flx * step_z;
    }
    m0 = m0_sum == 0 ? NAN : m0_sum;
    masks[x + y * size_x] = m0_sum != 0;

    // Only output requested moments
    if (orders[m] == 0)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m0;
        m++;
    }

    // Max order reached
    if (max_order == 0)
        return;

    // Moment 1
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = cube[idx];
        T vel = zero_z + z * step_z;
        m1_sum += flx * vel * step_z;
    }
    m1 = m0_sum == 0 ? NAN : m1_sum / m0;

    // Only output requested moments
    if (orders[m] == 1)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m1;
        m++;
    }

    // Max order reached
    if (max_order == 1)
        return;

    // Moment 2
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = cube[idx];
        T vel = zero_z + z * step_z;
        m2_sum += flx * std::pow(vel - m1, 2) * step_z;
    }
    m2 = m0_sum == 0 ? NAN : std::sqrt(m2_sum / m0);

    // Only output requested moments
    if (orders[m] == 2)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m2;
        m++;
    }

    // Max order reached
    if (max_order == 2)
        return;

    // Higher order moments
    for(; m < norders; ++m)
    {
        T mn=0, mn_sum=0;
        for (int z = 0; z < size_z; ++z)
        {
            int idx = x
                    + y * size_x
                    + z * size_x * size_y;
            T flx = cube[idx];
            T vel = zero_z + z * step_z;
            mn_sum += flx * std::pow(vel - m1, orders[m]); // * step_z;
        }
        mn = m0_sum == 0 ? NAN : mn_sum / m0;

        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = mn;
    }
}

} // namespace gbkfit
