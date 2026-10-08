
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

template<typename T> void
Objective<T>::residual(
        Ptr obs_d, Ptr obs_e, Ptr obs_m,
        Ptr mdl_d, Ptr mdl_w, Ptr mdl_m,
        int size, T weight, Ptr residual) const
{
    kernels::objective_residual(
            reinterpret_cast<const T*>(obs_d),
            reinterpret_cast<const T*>(obs_e),
            reinterpret_cast<const T*>(obs_m),
            reinterpret_cast<const T*>(mdl_d),
            reinterpret_cast<const T*>(mdl_w),
            reinterpret_cast<const T*>(mdl_m),
            size, weight,
            reinterpret_cast<T*>(residual));

//    T* r = reinterpret_cast<T*>(residual);
//    T sum = 0;
//    #pragma omp parallel for reduction(+:sum)
//    for(int i = 0; i < size; ++i) {
//        sum = sum + std::fabs(r[i]);
//    }
//    r[0] = sum;
}

template<typename T> void
Objective<T>::residual_sum(Ptr residual, int size, bool squared, Ptr sum) const
{
    kernels::objective_residual_sum(
            reinterpret_cast<const T*>(residual),
            size, squared,
            reinterpret_cast<T*>(sum));
}

#define INSTANTIATE(T)\
    template struct Objective<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::host
