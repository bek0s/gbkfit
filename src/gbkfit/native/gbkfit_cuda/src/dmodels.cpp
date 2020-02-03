
#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit { namespace cuda {

#define INSTANTIATE(T)               \
    template struct DModelDCube<T>;
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}} // namespace gbkfit::cuda
