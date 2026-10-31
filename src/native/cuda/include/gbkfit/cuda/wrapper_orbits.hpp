#pragma once

#include <gbkfit/orbits/orbit.hpp>

namespace gbkfit::cuda {

// The kernels of the orbits (see gbkfit/orbits)
template<typename T>
struct OrbitsWrapper
{
    static void
    orbits_potential(
            const PotentialArgs<T>& p, int n, const T* points,
            T* potential, T* forces);

    static void
    orbits_integrate(const OrbitArgs<T>& a);
};

} // namespace gbkfit::cuda
