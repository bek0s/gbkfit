#pragma once

namespace gbkfit {

template<typename T> constexpr void
moments(int x, int y,
        int size_x, int size_y, int size_z,
        T step_x, T step_y, T step_z,
        T zero_x, T zero_y, T zero_z,
        const T* cube,
        T* mmaps, const int* orders, int norders)
{
    (void)step_x;
    (void)step_y;
    (void)zero_x;
    (void)zero_y;

    //
    // Check if spectrum has enough signal to extract moments
    // This is a very crude and primitive approach
    //

    constexpr T EPS1 = 1e-5;
    constexpr T EPS2 = 5e-2;

    int nbump = 0;
    T minimum = 0;
    T maximum = 0;
    T amplitude = 0;

    // Calculate amplitude range
    for(int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T value = cube[idx];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    amplitude = maximum - minimum;

    // Count the number of trend micro-changes in the spectrum
    T prev = cube[1] - cube[0];
    for(int z = 1; z < size_z - 1; ++z)
    {
        int idx1 = x
                + y * size_x
                + (z+0) * size_x * size_y;
        int idx2 = x
                + y * size_x
                + (z+1) * size_x * size_y;
        T diff = cube[idx2] - cube[idx1];
        if (diff == 0)
            continue;
        if (prev * diff < 0 && std::fabs(diff) / amplitude > EPS1)
            nbump += 1;
        prev = diff;
    }

    if (float(nbump) / size_z > EPS2)
        return;

    //
    // The spectrum signal is good enough to calculate moments
    //

    int max_order = orders[norders - 1];

    int m = 0;
    T m0=0, m1=0, m2=0, m0_sum=0, m1_sum=0, m2_sum=0;

    // Moment 0
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = std::max(T{0}, cube[idx]);
        m0_sum += flx * step_z;
    }
    m0 = m0_sum;

    // Only output requested moments
    if (orders[m] == 0)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m0;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 0)
        return;

    // Moment 1
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = std::max(T{0}, cube[idx]);
        T vel = zero_z + z * step_z;
        m1_sum += flx * vel  * step_z;
    }
    m1 = m0 > 0 ? m1_sum / m0 : NAN;

    // Only output requested moments
    if (orders[m] == 1)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m1;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 1)
        return;

    // Moment 2
    for (int z = 0; z < size_z; ++z)
    {
        int idx = x
                + y * size_x
                + z * size_x * size_y;
        T flx = std::max(T{0}, cube[idx]);
        T vel = zero_z + z * step_z;
        m2_sum += flx * std::pow(vel - m1, 2);
    }
    m2 = m0 > 0 ? std::sqrt(m2_sum / m0) : NAN;

    // Only output requested moments
    if (orders[m] == 2)
    {
        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = m2;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 2)
        return;
    // todo: fix the math
    // Higher order moments
    for(; m < norders; ++m)
    {
        T mn=0, mn_sum = 0;
        for (int z = 0; z < size_z; ++z)
        {
            int idx = x
                    + y * size_x
                    + z * size_x * size_y;
            T flx = std::max(T{0}, cube[idx]);
            T vel = zero_z + z * step_z;
            mn_sum += flx * std::pow(vel - m1, orders[m]);
        }
        mn = m0 > 0 ? mn_sum / m0 : NAN;

        int idx = x
                + y * size_x
                + m * size_x * size_y;
        mmaps[idx] = mn;
    }
}

} // namespace gbkfit
