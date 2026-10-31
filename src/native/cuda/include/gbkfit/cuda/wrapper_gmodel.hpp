#pragma once

#include <gbkfit/gmodel/cone.hpp>
#include <gbkfit/gmodel/disk.hpp>
#include <gbkfit/gmodel/spheroid.hpp>
#include <gbkfit/gmodel/transfer.hpp>

namespace gbkfit::cuda {

// The kernels of the models of galaxies (see gbkfit/gmodel)
template<typename T>
struct GModelWrapper
{
    static void
    gmodel_mcdisk_evaluate(
            const DiskArgs<T>& a, const GridArgs<T>& g,
            const MCDiskArgs<T>& mc, const FieldsArgs<T>& f);

    static void
    gmodel_smdisk_evaluate(
            const DiskArgs<T>& a, const GridArgs<T>& g,
            const FieldsArgs<T>& f);

    static void
    gmodel_smcone_evaluate(
            const ConeArgs<T>& a, const GridArgs<T>& g,
            const FieldsArgs<T>& f);

    static void
    gmodel_smspheroid_evaluate(
            const SpheroidArgs<T>& a, const GridArgs<T>& g,
            const FieldsArgs<T>& f);

    static void
    gmodel_opacity_accumulate(int size_x, int size_y, int size_z, T* opacity);

    static void
    gmodel_transfer_march(const TransferArgs<T>& a);

    static void
    gmodel_disk_traits_evaluate(const DiskTraitsArgs<T>& a);
};

} // namespace gbkfit::cuda
