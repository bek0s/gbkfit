#pragma once

// The kernels of each family, launched on the device of their arrays
// (see DeviceScope), each family in a header and a source file of its
// own, so that they compile at once, and a change of one family compiles
// only its own
#include "gbkfit/cuda/wrapper_dmodel.hpp"
#include "gbkfit/cuda/wrapper_gmodel.hpp"
#include "gbkfit/cuda/wrapper_objective.hpp"
#include "gbkfit/cuda/wrapper_orbits.hpp"

namespace gbkfit { namespace cuda {

// While it lives, the calling thread launches its kernels on a device
// (that of their arrays: there can be several GPUs), and after, on the
// device it had before
class DeviceScope
{
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int m_device;
    int m_previous;
};

// All the kernels, as the bindings take them
template<typename T>
struct Wrapper
    : DModelWrapper<T>, GModelWrapper<T>, ObjectiveWrapper<T>,
      OrbitsWrapper<T>
{
    using DeviceScope = cuda::DeviceScope;
};

}} // namespace gbkfit::cuda
