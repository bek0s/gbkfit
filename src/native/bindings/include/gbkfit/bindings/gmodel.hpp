#pragma once

#include <optional>
#include <string>

#include <nanobind/stl/array.h>

#include <gbkfit/gmodel/disk.hpp>

#include "gbkfit/bindings/arrays.hpp"

namespace gbkfit::bindings {

// The arrays of a set of traits (see gbkfit::TraitSet)
template<typename T, typename Device>
struct TraitSetArrays
{
    using Ints = Array<Device, const int, nb::ndim<1>>;
    using Values = Array<Device, const T, nb::ndim<1>>;

    Ints uids;
    Values cvalues;
    Ints ccounts;
    Values pvalues;
    Ints pcounts;

    TraitSet<T>
    raw() const
    {
        const auto n = uids.shape(0);
        require(ccounts.shape(0) == n && pcounts.shape(0) == n,
                "a trait set needs a constant and a parameter count "
                "for each trait");
        return {int(n), uids.data(), cvalues.data(), ccounts.data(),
                pvalues.data(), pcounts.data()};
    }
};

// The arrays that describe a disk: its radial nodes, its geometry and
// its traits. They are created once, and their values are updated in
// place before each evaluation.
template<typename T, typename Device>
struct DiskArrays
{
    using Values = Array<Device, const T, nb::ndim<1>>;
    using Traits = std::optional<TraitSetArrays<T, Device>>;

    bool loose;
    bool tilted;
    Values rnodes;
    Values vsys;
    Values xpos;
    Values ypos;
    Values posa;
    Values incl;
    Traits rpt, rht, vpt, vht, dpt, dht, zpt, spt, wpt;
};

// The galaxy model operations of a native module. Kernels launches
// the kernels of the module, on the memory of Device.
template<typename T, typename Device, typename Kernels>
struct GModel
{
    using Disk = DiskArrays<T, Device>;
    using Traits = TraitSetArrays<T, Device>;
    using Values = typename Disk::Values;
    using Ints = typename Traits::Ints;
    using Bools = Array<Device, const bool, nb::ndim<1>>;
    using Data = Array<Device, T>;
    using ConstData = Array<Device, const T>;
    using Cube = Array<Device, T, nb::ndim<3>>;
    using ConstCube = Array<Device, const T, nb::ndim<3>>;

    // Assign the same (mean over z) weight of the 3d spatial weight cube
    // to the whole spectrum of each spatial position
    static void
    wcube_evaluate(ConstCube spat_wcube, Cube spec_wcube)
    {
        const auto spat = size_xyz(spat_wcube);
        const auto spec = size_xyz(spec_wcube);
        require(spat[0] == spec[0] && spat[1] == spec[1],
                "spat_wcube and spec_wcube must have the same x and y size");
        Kernels::gmodel_wcube_evaluate(
                spat[0], spat[1], spat[2], spec[2],
                spat_wcube.data(), spec_wcube.data());
    }

    static void
    smdisk_evaluate(
            const Disk& disk,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota,
            int spec_size, T spec_step, T spec_zero,
            ConstData opacity,
            Data image, Data scube,
            Data wdata, Data wdata_cmp,
            Data rdata, Data rdata_cmp,
            Data ordata, Data ordata_cmp,
            Data vdata_cmp, Data ddata_cmp)
    {
        Kernels::gmodel_smdisk_evaluate(make_args(
                disk, spat_size, spat_step, spat_zero, spat_rota,
                spec_size, spec_step, spec_zero, opacity,
                image, scube, wdata, wdata_cmp, rdata, rdata_cmp,
                ordata, ordata_cmp, vdata_cmp, ddata_cmp));
    }

    static void
    mcdisk_evaluate(
            const Disk& disk,
            T cflux, unsigned int seed,
            int nclouds, Ints ncloudscsum, Bools has_analytical_integral,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota,
            int spec_size, T spec_step, T spec_zero,
            ConstData opacity,
            Data image, Data scube,
            Data wdata, Data wdata_cmp,
            Data rdata, Data rdata_cmp,
            Data ordata, Data ordata_cmp,
            Data vdata_cmp, Data ddata_cmp)
    {
        const auto args = make_args(
                disk, spat_size, spat_step, spat_zero, spat_rota,
                spec_size, spec_step, spec_zero, opacity,
                image, scube, wdata, wdata_cmp, rdata, rdata_cmp,
                ordata, ordata_cmp, vdata_cmp, ddata_cmp);
        require(has_analytical_integral.shape(0) == size_t(args.rpt.n),
                "has_analytical_integral needs one value for each density "
                "trait");
        MCDiskArgs<T> mc;
        mc.cflux = cflux;
        mc.seed = seed;
        mc.nclouds = nclouds;
        mc.ncloudscsum = ncloudscsum.data();
        mc.ncloudscsum_len = int(ncloudscsum.shape(0));
        mc.has_analytical_integral = has_analytical_integral.data();
        Kernels::gmodel_mcdisk_evaluate(args, mc);
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<Traits>(m, ("TraitSet" + suffix).c_str())
                .def("__init__", [](
                        Traits* self, Ints uids, Values cvalues, Ints ccounts,
                        Values pvalues, Ints pcounts) {
                    new (self) Traits{
                            uids, cvalues, ccounts, pvalues, pcounts};
                },
                nb::arg("uids").noconvert(),
                nb::arg("cvalues").noconvert(),
                nb::arg("ccounts").noconvert(),
                nb::arg("pvalues").noconvert(),
                nb::arg("pcounts").noconvert());

        nb::class_<Disk>(m, ("Disk" + suffix).c_str())
                .def("__init__", [](
                        Disk* self, bool loose, bool tilted, Values rnodes,
                        Values vsys, Values xpos, Values ypos,
                        Values posa, Values incl,
                        const Traits* rpt, const Traits* rht,
                        const Traits* vpt, const Traits* vht,
                        const Traits* dpt, const Traits* dht,
                        const Traits* zpt, const Traits* spt,
                        const Traits* wpt) {
                    auto opt = [](const Traits* traits) {
                        return traits ? std::optional(*traits) : std::nullopt;
                    };
                    new (self) Disk{
                            loose, tilted, rnodes, vsys, xpos, ypos, posa,
                            incl, opt(rpt), opt(rht), opt(vpt), opt(vht),
                            opt(dpt), opt(dht), opt(zpt), opt(spt), opt(wpt)};
                },
                nb::arg("loose"), nb::arg("tilted"),
                nb::arg("rnodes").noconvert(),
                nb::arg("vsys").noconvert().none() = nb::none(),
                nb::arg("xpos").noconvert(),
                nb::arg("ypos").noconvert(),
                nb::arg("posa").noconvert(),
                nb::arg("incl").noconvert(),
                nb::arg("rpt").none() = nb::none(),
                nb::arg("rht").none() = nb::none(),
                nb::arg("vpt").none() = nb::none(),
                nb::arg("vht").none() = nb::none(),
                nb::arg("dpt").none() = nb::none(),
                nb::arg("dht").none() = nb::none(),
                nb::arg("zpt").none() = nb::none(),
                nb::arg("spt").none() = nb::none(),
                nb::arg("wpt").none() = nb::none());

        auto evaluate_args = [](auto&&... first) {
            return std::make_tuple(
                    first...,
                    nb::arg("spat_size"), nb::arg("spat_step"),
                    nb::arg("spat_zero"), nb::arg("spat_rota"),
                    nb::arg("spec_size"),
                    nb::arg("spec_step"), nb::arg("spec_zero"),
                    nb::arg("opacity").noconvert().none() = nb::none(),
                    nb::arg("image").noconvert().none() = nb::none(),
                    nb::arg("scube").noconvert().none() = nb::none(),
                    nb::arg("wdata").noconvert().none() = nb::none(),
                    nb::arg("wdata_cmp").noconvert().none() = nb::none(),
                    nb::arg("rdata").noconvert().none() = nb::none(),
                    nb::arg("rdata_cmp").noconvert().none() = nb::none(),
                    nb::arg("ordata").noconvert().none() = nb::none(),
                    nb::arg("ordata_cmp").noconvert().none() = nb::none(),
                    nb::arg("vdata_cmp").noconvert().none() = nb::none(),
                    nb::arg("ddata_cmp").noconvert().none() = nb::none());
        };

        auto cls = nb::class_<GModel>(m, ("GModel" + suffix).c_str())
                .def(nb::init<>())
                .def_static("wcube_evaluate", &wcube_evaluate,
                        nb::arg("spat_wcube").noconvert(),
                        nb::arg("spec_wcube").noconvert());
        std::apply([&](auto&&... args) {
            cls.def_static("smdisk_evaluate", &smdisk_evaluate, args...);
        }, evaluate_args(nb::arg("disk")));
        std::apply([&](auto&&... args) {
            cls.def_static("mcdisk_evaluate", &mcdisk_evaluate, args...);
        }, evaluate_args(
                nb::arg("disk"), nb::arg("cflux"), nb::arg("seed"),
                nb::arg("nclouds"),
                nb::arg("ncloudscsum").noconvert(),
                nb::arg("has_analytical_integral").noconvert()));
    }

private:

    static DiskArgs<T>
    make_args(
            const Disk& disk,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota,
            int spec_size, T spec_step, T spec_zero,
            const ConstData& opacity,
            const Data& image, const Data& scube,
            const Data& wdata, const Data& wdata_cmp,
            const Data& rdata, const Data& rdata_cmp,
            const Data& ordata, const Data& ordata_cmp,
            const Data& vdata_cmp, const Data& ddata_cmp)
    {
        DiskArgs<T> a;
        a.loose = disk.loose;
        a.tilted = disk.tilted;
        a.nrnodes = int(disk.rnodes.shape(0));
        a.rnodes = disk.rnodes.data();
        a.vsys = data(disk.vsys);
        a.xpos = disk.xpos.data();
        a.ypos = disk.ypos.data();
        a.posa = disk.posa.data();
        a.incl = disk.incl.data();
        const size_t nloose = disk.loose ? a.nrnodes : 1;
        const size_t ntilted = disk.tilted ? a.nrnodes : 1;
        require((!disk.vsys.is_valid() || disk.vsys.shape(0) == nloose)
                && disk.xpos.shape(0) == nloose
                && disk.ypos.shape(0) == nloose
                && disk.posa.shape(0) == ntilted
                && disk.incl.shape(0) == ntilted,
                "the geometry needs a value for each node of a loose or "
                "tilted disk, and one value otherwise");

        require(disk.rpt.has_value(), "a disk needs density traits");
        auto traits = [](const auto& set) {
            return set ? set->raw() : TraitSet<T>{};
        };
        a.rpt = traits(disk.rpt);
        a.rht = traits(disk.rht);
        a.vpt = traits(disk.vpt);
        a.vht = traits(disk.vht);
        a.dpt = traits(disk.dpt);
        a.dht = traits(disk.dht);
        a.zpt = traits(disk.zpt);
        a.spt = traits(disk.spt);
        a.wpt = traits(disk.wpt);
        require((!disk.rht || a.rht.n == a.rpt.n)
                && (!disk.vht || a.vht.n == a.vpt.n)
                && (!disk.dht || a.dht.n == a.dpt.n),
                "height traits must have as many traits as polar traits");

        for (int i = 0; i < 3; ++i) {
            a.spat_size[i] = spat_size[i];
            a.spat_step[i] = spat_step[i];
            a.spat_zero[i] = spat_zero[i];
        }
        a.spat_rota = spat_rota;
        a.spec_size = spec_size;
        a.spec_step = spec_step;
        a.spec_zero = spec_zero;

        const size_t nimage = size_t(spat_size[0]) * spat_size[1];
        const size_t nspat = nimage * spat_size[2];
        const size_t nscube = nimage * spec_size;
        auto output = [](const auto& array, size_t size, const char* name) {
            require(!array.is_valid() || array.size() == size,
                    std::string(name) + " has the wrong number of elements");
            return data(array);
        };
        a.opacity = output(opacity, nspat, "opacity");
        a.image = output(image, nimage, "image");
        a.scube = output(scube, nscube, "scube");
        a.wdata = output(wdata, nspat, "wdata");
        a.wdata_cmp = output(wdata_cmp, nspat, "wdata_cmp");
        a.rdata = output(rdata, nspat, "rdata");
        a.rdata_cmp = output(rdata_cmp, nspat, "rdata_cmp");
        a.ordata = output(ordata, nspat, "ordata");
        a.ordata_cmp = output(ordata_cmp, nspat, "ordata_cmp");
        a.vdata_cmp = output(vdata_cmp, nspat, "vdata_cmp");
        a.ddata_cmp = output(ddata_cmp, nspat, "ddata_cmp");
        return a;
    }
};

} // namespace gbkfit::bindings
