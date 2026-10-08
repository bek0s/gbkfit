
#include "gbkfit/cuda/objective.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T> void
Objective<T>::count_pixels(
        Ptr data1, Ptr data2, int size, T epsilon, Ptr counts) const
{
    Wrapper<T>::objective_count_pixels(
            reinterpret_cast<const T*>(data1),
            reinterpret_cast<const T*>(data2),
            size,
            epsilon,
            reinterpret_cast<int*>(counts));
}

template<typename T> void
Objective<T>::residual(
        Ptr obs_d, Ptr obs_e, Ptr obs_m,
        Ptr mdl_d, Ptr mdl_w, Ptr mdl_m,
        int size, T weight, Ptr residual) const
{
    Wrapper<T>::objective_residual(
            reinterpret_cast<const T*>(obs_d),
            reinterpret_cast<const T*>(obs_e),
            reinterpret_cast<const T*>(obs_m),
            reinterpret_cast<const T*>(mdl_d),
            reinterpret_cast<const T*>(mdl_w),
            reinterpret_cast<const T*>(mdl_m),
            size, weight,
            reinterpret_cast<T*>(residual));
}

#define INSTANTIATE(T)\
    template struct Objective<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::cuda
