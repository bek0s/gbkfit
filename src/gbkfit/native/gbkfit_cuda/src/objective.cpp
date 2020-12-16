
#include "gbkfit/cuda/objective.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T> void
Objective<T>::count_pixels(Ptr data, Ptr model, int size, Ptr counts) const
{
    Wrapper<T>::objective_count_pixels(
                reinterpret_cast<const T*>(data),
                reinterpret_cast<const T*>(model),
                size,
                reinterpret_cast<int*>(counts));
}

#define INSTANTIATE(T)            \
    template struct Objective<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::cuda
