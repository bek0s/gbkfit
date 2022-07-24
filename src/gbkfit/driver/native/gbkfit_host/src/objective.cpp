
#include "gbkfit/host/objective.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T> void
Objective<T>::count_pixels(
        Ptr data1, Ptr data2, int size, T epsilon, Ptr counts) const
{
    kernels::objective_count_pixels(
            reinterpret_cast<const T*>(data1),
            reinterpret_cast<const T*>(data2),
            size,
            epsilon,
            reinterpret_cast<int*>(counts));
}

#define INSTANTIATE(T)            \
    template struct Objective<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::host
