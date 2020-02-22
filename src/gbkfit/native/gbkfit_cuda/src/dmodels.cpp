
#include "gbkfit/drivers/cuda/dmodels.hpp"
#include "gbkfit/drivers/cuda/wrapper.hpp"

namespace gbkfit { namespace cuda {

#define INSTANTIATE(T)               \
    template struct DModelDCube<T>;
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}} // namespace gbkfit::cuda
