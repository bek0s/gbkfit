#include "launch.cuh"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

DeviceScope::DeviceScope(int device)
    : m_device(device)
{
    check(cudaGetDevice(&m_previous), "cudaGetDevice");
    if (m_device != m_previous)
        check(cudaSetDevice(m_device), "cudaSetDevice");
}

DeviceScope::~DeviceScope()
{
    // (a destructor cannot throw; an error shows at the next call)
    if (m_device != m_previous)
        cudaSetDevice(m_previous);
}

} // namespace gbkfit::cuda
