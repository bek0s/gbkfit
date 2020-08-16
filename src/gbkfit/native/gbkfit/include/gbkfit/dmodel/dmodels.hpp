#pragma once

namespace gbkfit {

template<typename T> void
moments(int x, int y,
        int spat_size_x, int spat_size_y,
        int spec_size,
        T spec_step,
        T spec_zero,
        T nanval,
        const T* cube,
        T* mmaps, const int* orders, int norders)
{
    int max_order = orders[norders - 1];

    int m = 0;
    T m0, m1, m2, m0_sum=0, m1_sum=0, m2_sum=0;

    // Moment 0
    for (int z = 0; z < spec_size; ++z)
    {
        int idx = x
                + y * spat_size_x
                + z * spat_size_x * spat_size_y;

        T flx = std::max(T{0}, cube[idx]);
        m0_sum += flx;
    }
    m0 = m0_sum;

    // Only output requested moments
    if (orders[m] == 0)
    {
        int idx = x
                + y * spat_size_x
                + m * spat_size_x * spat_size_y;

    //  mmaps[idx] = 0;
        mmaps[idx] = m0;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 0)
        return;

    // Moment 1
    for (int z = 0; z < spec_size; ++z)
    {
        int idx = x
                + y * spat_size_x
                + z*spat_size_x*spat_size_y;

        T flx = std::max(T{0}, cube[idx]);
        T vel = spec_zero + z * spec_step;
        m1_sum += flx * vel;
    }
    m1 = m0 > 0 ? m1_sum / m0 : nanval;

    // Only output requested moments
    if (orders[m] == 1)
    {
        int idx = x
                + y * spat_size_x
                + m * spat_size_x * spat_size_y;

    //  mmaps[idx] = 1;
        mmaps[idx] = m1;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 1)
        return;

    // Moment 2
    for (int z = 0; z < spec_size; ++z)
    {
        int idx = x
                + y * spat_size_x
                + z * spat_size_x*spat_size_y;

        T flx = std::max(T{0}, cube[idx]);
        T vel = spec_zero + z * spec_step;
        m2_sum += flx * (vel - m1) * (vel - m1);
    }
    m2 = m0 > 0 ? std::sqrt(m2_sum / m0) : nanval;

    // Only output requested moments
    if (orders[m] == 2)
    {
        int idx = x
                + y * spat_size_x
                + m * spat_size_x * spat_size_y;

    //  mmaps[idx] = 2;
        mmaps[idx] = m2;
        m++;
    }

    // Max order just reached. We are done
    if (max_order == 2)
        return;

    /*
    // Higher order moments
    for(; m < norders; ++m)
    {
        T mn, mn_sum = 0;
        for (int z = 0; z < spec_size; ++z)
        {
            int idx = x
                    + y * spat_size_x
                    + z * spat_size_x*spat_size_y;

            T flx = std::max(T{0}, cube[idx]);
            T vel = spec_zero + z * spec_step;
            mn_sum += flx * std::pow(vel - m1, orders[m]);
        }
        mn = m0 > 0 ? mn_sum / m0 : nanval;

        int idx = x
                + y * spat_size_x
                + m * spat_size_x * spat_size_y;

        mmaps[idx] = mn;
    }
    */
}

} // namespace gbkfit
