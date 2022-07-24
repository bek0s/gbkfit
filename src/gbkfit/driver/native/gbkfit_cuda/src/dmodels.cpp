
#include "gbkfit/cuda/dmodels.hpp"
#include "gbkfit/cuda/wrapper.hpp"

namespace gbkfit::cuda {

template<typename T> void
_complex_multiply(std::array<int, 3> size, T* cube_c, T* psf_c)
{
    int n0 = size[2], n1 = size[1], n2 = size[0];
    const int n = n0 * n1 * (n2 / 2 + 1);
    const T nfactor = T{1} / (n0 * n1 * n2);
    Wrapper<T>::dmodel_dcube_complex_multiply_and_scale(
            reinterpret_cast<typename cufft<T>::complex*>(cube_c),
            reinterpret_cast<typename cufft<T>::complex*>(psf_c),
            n, nfactor);
}

template<typename T> void
DModel<T>::dcube_downscale(
        std::array<int, 3> scale,
        std::array<int, 3> offset,
        std::array<int, 3> src_size,
        std::array<int, 3> dst_size,
        Ptr src_dcube, Ptr dst_dcube) const
{
    Wrapper<T>::dmodel_dcube_downscale(
            scale[0], scale[1], scale[2],
            offset[0], offset[1], offset[2],
            src_size[0], src_size[1], src_size[2],
            dst_size[0], dst_size[1], dst_size[2],
            reinterpret_cast<T*>(src_dcube),
            reinterpret_cast<T*>(dst_dcube));
}

template<typename T> void
DModel<T>::dcube_mask(
        T cutoff, bool apply,
        std::array<int, 3> size,
        Ptr dcube_d, Ptr dcube_m, Ptr dcube_w) const
{
    Wrapper<T>::dmodel_dcube_make_mask(
            cutoff, apply,
            size[0], size[1], size[2],
            reinterpret_cast<T*>(dcube_d),
            reinterpret_cast<T*>(dcube_m),
            reinterpret_cast<T*>(dcube_w));
}

template<typename T> void
DModel<T>::mmaps_moments(
        std::array<int, 3> size,
        std::array<T, 3> step,
        std::array<T, 3> zero,
        Ptr dcube_d, Ptr dcube_w,
        T cutoff, int norders, Ptr orders,
        Ptr mmaps_d, Ptr mmaps_m, Ptr mmaps_w) const
{
//    Wrapper<T>::dmodel_mmaps_moments(
//            size[0], size[1], size[2],
//            step[0], step[1], step[2],
//            zero[0], zero[1], zero[2],
//            reinterpret_cast<T*>(scube),
//            reinterpret_cast<T*>(mmaps),
//            reinterpret_cast<T*>(masks),
//            reinterpret_cast<int*>(orders),
//            norders);
}

#define INSTANTIATE(T)\
    template struct DModel<T>;
INSTANTIATE(float)
#undef INSTANTIATE

} // namespace gbkfit::cuda
