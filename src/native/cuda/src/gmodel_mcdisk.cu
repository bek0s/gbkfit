#include <algorithm>

#include "gbkfit/cuda/kernels_gmodel.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_gmodel.hpp"

namespace gbkfit::cuda {

// The kernel of a Monte Carlo disk: of the fields of disks with line
// opacity and without, of disks dimmed by opacity, or of the others, with
// the velocity as a vector or without it (see
// gmodel_mcdisk_evaluate_cloud)
template<typename T, bool FullKinematics, bool HasVectors> auto
mcdisk_kernel(const DiskArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    namespace k = kernels;
    constexpr bool A = FullKinematics;
    constexpr bool V = HasVectors;
    return f.fields && a.opt.uids ? k::gmodel_mcdisk_evaluate<T, 0, 1, 1, A, V>
            : f.fields ? k::gmodel_mcdisk_evaluate<T, 0, 1, 0, A, V>
            : g.opacity ? k::gmodel_mcdisk_evaluate<T, 1, 0, 0, A, V>
            : k::gmodel_mcdisk_evaluate<T, 0, 0, 0, A, V>;
}

template<typename T, bool FullKinematics> auto
mcdisk_kernel(const DiskArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    return g.vvec_cmp ? mcdisk_kernel<T, FullKinematics, true>(a, g, f)
            : mcdisk_kernel<T, FullKinematics, false>(a, g, f);
}


template<typename T> void
GModelWrapper<T>::gmodel_mcdisk_evaluate(
        const DiskArgs<T>& a, const GridArgs<T>& g, const MCDiskArgs<T>& mc,
        const FieldsArgs<T>& f)
{
    // (of the full kinematics or not: see gmodel_disk_full_kinematics)
    launch("gmodel_mcdisk_evaluate", mc.nclouds,
            gbkfit::gmodel_disk_full_kinematics(a)
            ? mcdisk_kernel<T, true>(a, g, f)
            : mcdisk_kernel<T, false>(a, g, f),
            a, g, mc, f);
    // The fields of each voxel, from the moments of its clouds
    if (f.fields) {
        const long long n = 1LL * g.spat_size[0] * g.spat_size[1]
                * g.spat_size[2];
        launch("gmodel_mcdisk_fields_moments", n,
                kernels::gmodel_mcdisk_fields_moments<T>, a, g, f);
    }
}

// (only this member here: the disk kernels compile in files of their own,
// at once)
#define INSTANTIATE(T)\
    template void GModelWrapper<T>::gmodel_mcdisk_evaluate(\
            const DiskArgs<T>&, const GridArgs<T>&, const MCDiskArgs<T>&,\
            const FieldsArgs<T>&);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace gbkfit::cuda
