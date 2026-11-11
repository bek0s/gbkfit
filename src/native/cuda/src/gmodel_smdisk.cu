#include <algorithm>

#include "gbkfit/cuda/kernels_gmodel.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_gmodel.hpp"

namespace gbkfit::cuda {

// The kernel of a smooth disk (see mcdisk_kernel, gmodel_mcdisk.cu)
template<typename T, bool FullKinematics> auto
smdisk_kernel(const DiskArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    namespace k = kernels;
    constexpr bool A = FullKinematics;
    return f.fields && a.opt.uids ? k::gmodel_smdisk_evaluate<T, 0, 1, 1, A>
            : f.fields ? k::gmodel_smdisk_evaluate<T, 0, 1, 0, A>
            : g.opacity ? k::gmodel_smdisk_evaluate<T, 1, 0, 0, A>
            : k::gmodel_smdisk_evaluate<T, 0, 0, 0, A>;
}


template<typename T> void
GModelWrapper<T>::gmodel_smdisk_evaluate(
        const DiskArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    // (the voxels need not be in memory, so there can be more than an int
    // holds, e.g. 1300 x 1300 x 1300)
    const long long n = 1LL * g.spat_size[0] * g.spat_size[1]
            * gbkfit::gmodel_smdisk_size_z(a, g);
    // (see gmodel_mcdisk_evaluate)
    launch("gmodel_smdisk_evaluate", n,
            gbkfit::gmodel_disk_full_kinematics(a)
            ? smdisk_kernel<T, true>(a, g, f)
            : smdisk_kernel<T, false>(a, g, f),
            a, g, f);
}

// (only this member here, see gmodel_mcdisk.cu)
#define INSTANTIATE(T)\
    template void GModelWrapper<T>::gmodel_smdisk_evaluate(\
            const DiskArgs<T>&, const GridArgs<T>&, const FieldsArgs<T>&);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace gbkfit::cuda
