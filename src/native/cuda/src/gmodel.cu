#include <algorithm>

#include "gbkfit/cuda/kernels_gmodel.hpp"
#include "launch.cuh"
#include "gbkfit/cuda/wrapper_gmodel.hpp"

namespace gbkfit::cuda {

template<typename T> void
GModelWrapper<T>::gmodel_smcone_evaluate(
        const ConeArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    // (see gmodel_smdisk_evaluate)
    const long long n = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    // (the kernels of the fields, of cones dimmed by opacity, and of the
    // others)
    namespace k = kernels;
    launch("gmodel_smcone_evaluate", n,
            f.fields ? k::gmodel_smcone_evaluate<T, 0, 1>
            : g.opacity ? k::gmodel_smcone_evaluate<T, 1, 0>
            : k::gmodel_smcone_evaluate<T, 0, 0>,
            a, g, f);
}

template<typename T> void
GModelWrapper<T>::gmodel_smspheroid_evaluate(
        const SpheroidArgs<T>& a, const GridArgs<T>& g, const FieldsArgs<T>& f)
{
    namespace k = kernels;
    if (a.projected) {
        launch("gmodel_smspheroid_evaluate_projected",
                g.spat_size[0] * g.spat_size[1],
                k::gmodel_smspheroid_evaluate_projected<T>, a, g);
        return;
    }
    // (see gmodel_smdisk_evaluate)
    const long long n = 1LL * g.spat_size[0] * g.spat_size[1]
            * g.spat_size[2];
    // (the kernels of the fields, of spheroids dimmed by opacity, and of
    // the others)
    launch("gmodel_smspheroid_evaluate", n,
            f.fields ? k::gmodel_smspheroid_evaluate<T, 0, 1>
            : g.opacity ? k::gmodel_smspheroid_evaluate<T, 1, 0>
            : k::gmodel_smspheroid_evaluate<T, 0, 0>,
            a, g, f);
}

template<typename T> void
GModelWrapper<T>::gmodel_opacity_accumulate(
        int size_x, int size_y, int size_z, T* opacity)
{
    const int n = size_x * size_y;
    launch("gmodel_opacity_accumulate", n,
            kernels::gmodel_opacity_accumulate<T>,
            size_x, size_y, size_z, opacity);
}

template<typename T> void
GModelWrapper<T>::gmodel_transfer_march(const TransferArgs<T>& a)
{
    const long long n = 1LL * a.spat_size[0] * a.spat_size[1]
            * kernels::transfer_chunks(a);
    launch("gmodel_transfer_march", n,
            a.tau ? kernels::gmodel_transfer_march<T, true>
                  : kernels::gmodel_transfer_march<T, false>,
            a);
    if (a.opacity) {
        launch("gmodel_transfer_opacity_front",
                1LL * a.spat_size[0] * a.spat_size[1],
                kernels::gmodel_transfer_opacity_front<T>, a);
    }
}

template<typename T> void
GModelWrapper<T>::gmodel_disk_traits_evaluate(const DiskTraitsArgs<T>& a)
{
    launch("gmodel_disk_traits_evaluate", a.npoints,
            kernels::gmodel_disk_traits_evaluate<T>, a);
}

// (the members of the disks are in gmodel_mcdisk.cu and gmodel_smdisk.cu)
#define INSTANTIATE(T)\
    template void GModelWrapper<T>::gmodel_smcone_evaluate(\
            const ConeArgs<T>&, const GridArgs<T>&, const FieldsArgs<T>&);\
    template void GModelWrapper<T>::gmodel_smspheroid_evaluate(\
            const SpheroidArgs<T>&, const GridArgs<T>&,\
            const FieldsArgs<T>&);\
    template void GModelWrapper<T>::gmodel_opacity_accumulate(\
            int, int, int, T*);\
    template void GModelWrapper<T>::gmodel_transfer_march(\
            const TransferArgs<T>&);\
    template void GModelWrapper<T>::gmodel_disk_traits_evaluate(\
            const DiskTraitsArgs<T>&);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace gbkfit::cuda
