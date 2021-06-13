
#include "gbkfit/host/objective.hpp"
#include "gbkfit/host/kernels.hpp"

namespace gbkfit::host {

template<typename T> void
Objective<T>::count_pixels(Ptr data, Ptr model, int size, Ptr counts) const
{
    kernels::objective_count_pixels(
            reinterpret_cast<const T*>(data),
            reinterpret_cast<const T*>(model),
            size,
            reinterpret_cast<int*>(counts));
}

#define INSTANTIATE(T)            \
    template struct Objective<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::host
