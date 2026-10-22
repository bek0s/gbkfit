#pragma once

#include <algorithm>
#include <optional>
#include <string>

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <gbkfit/constants.hpp>
#include <gbkfit/gmodel/cone.hpp>
#include <gbkfit/gmodel/disk.hpp>
#include <gbkfit/gmodel/transfer.hpp>

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/bindings/gil.hpp"

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
        // (the kernels take at most TRAIT_NUM_MAX traits of a set, e.g.
        // the per-line coefficients of line opacity are in arrays of it)
        require(n <= size_t(TRAIT_NUM_MAX),
                "a trait set has at most " + std::to_string(TRAIT_NUM_MAX)
                + " traits");
        return {int(n), uids.data(), cvalues.data(), ccounts.data(),
                pvalues.data(), pcounts.data()};
    }
};

// The arrays that describe a disk: its radial nodes, its geometry (with
// the sines and cosines of posa and incl of a loose or tilted disk, see
// gbkfit::DiskArgs), its traits (with the direction of each velocity
// polar trait, and of each dispersion polar trait if any is not
// isotropic) and the dispersion of its vertical equilibrium, if any (with
// whether it is along z only).
// They are created once, and their values are updated in place before
// each evaluation.
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
    Traits rpt, rht, vpt, vht, dpt, dht, zpt, spt, opt;
    Array<Device, const int, nb::ndim<1>> vpt_directions;
    Array<Device, const int, nb::ndim<1>> dpt_directions;
    Array<Device, const int, nb::ndim<1>> opt_rows;
    Values posa_incl_sincos;
    Values deq;
    bool deq_vertical;
};

// The arrays that describe a cone: its sides, its radial nodes (shells),
// its geometry and its traits (with the direction of each velocity polar
// trait; see gbkfit::ConeArgs). They are created once, and their values
// are updated in place before each evaluation.
template<typename T, typename Device>
struct ConeArrays
{
    using Values = Array<Device, const T, nb::ndim<1>>;
    using Traits = std::optional<TraitSetArrays<T, Device>>;

    bool positive;
    bool negative;
    Values rnodes;
    Values vsys;
    Values xpos;
    Values ypos;
    Values posa;
    Values incl;
    Values wall;
    Traits rpt, rht, vpt, vht, dpt, dht, spt;
    Array<Device, const int, nb::ndim<1>> vpt_directions;
};

// The arrays of an emitter of the march of radiative transfer (see
// gbkfit::TransferEmitter): its fields, those of a volume (of shape
// (nslots, nz, ny, nx), with j, v and s in the first three slots, and
// with line opacity, a slot for each line after them) or of a layer (of
// shape (4, ny, nx)); its lines (of shape (nlines, 3)); and with opacity,
// a row of the ratios of the cubes of opacity for each line (of shape
// (nlines, nopacity)). They are created once, and their values are
// updated in place before each march.
template<typename T, typename Device>
struct TransferEmitterArrays
{
    using Fields = Array<Device, const T>;
    using Rows = Array<Device, const T, nb::ndim<2>>;

    bool layer;
    bool line_opacity;
    Fields fields;
    Rows lines;
    Rows opacity_ratios;
};

// The galaxy model operations of a native module. Kernels launches
// the kernels of the module, on the memory of Device: each function on
// the device of the array it writes, or of the disk or cone (see
// DeviceScope of the Kernels).
template<typename T, typename Device, typename Kernels>
struct GModel
{
    using Disk = DiskArrays<T, Device>;
    using Cone = ConeArrays<T, Device>;
    using Traits = TraitSetArrays<T, Device>;
    using Values = typename Disk::Values;
    using Ints = typename Traits::Ints;
    using Bools = Array<Device, const bool, nb::ndim<1>>;
    using Data = Array<Device, T>;
    using ConstData = Array<Device, const T>;
    using Lines = Array<Device, const T, nb::ndim<2>>;
    using Ratios = Array<Device, const T, nb::ndim<2>>;
    using TraitValues = Array<Device, T, nb::ndim<2>>;
    using Emitter = TransferEmitterArrays<T, Device>;

    static void
    smdisk_evaluate(
            const Disk& disk,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota, int slab_zero,
            int spec_size, T spec_step, T spec_zero,
            Lines lines,
            ConstData opacity, Ratios opacity_ratios,
            Data image, Data scube,
            Data rdata, Data rdata_cmp,
            Data ordata, Data ordata_cmp,
            Data vdata_cmp, Data ddata_cmp, Data vdweight_cmp,
            Data fields,
            T rdata_scale, T image_scale)
    {
        const typename Kernels::DeviceScope scope(disk.rnodes.device_id());
        const auto a = make_disk_args(disk);
        const bool thick = disk.rht.has_value();
        const auto g = make_grid_args(
                thick, spat_size, spat_step, spat_zero, spat_rota, slab_zero,
                spec_size, spec_step, spec_zero, lines,
                opacity, opacity_ratios,
                image, scube, rdata, rdata_cmp,
                ordata, ordata_cmp, vdata_cmp, ddata_cmp, vdweight_cmp,
                rdata_scale, image_scale);
        // (a thick disk writes three fields: j, v and s)
        Kernels::gmodel_smdisk_evaluate(
                a, g, make_fields_args(
                        thick, fields_lines(a, g), g, fields, 3));
    }

    static void
    smcone_evaluate(
            const Cone& cone,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota, int slab_zero,
            int spec_size, T spec_step, T spec_zero,
            Lines lines,
            ConstData opacity, Ratios opacity_ratios,
            Data image, Data scube,
            Data rdata, Data rdata_cmp,
            Data ordata, Data ordata_cmp,
            Data vdata_cmp, Data ddata_cmp, Data vdweight_cmp,
            Data fields,
            T rdata_scale, T image_scale)
    {
        const typename Kernels::DeviceScope scope(cone.rnodes.device_id());
        const auto a = make_cone_args(cone);
        const auto g = make_grid_args(
                true, spat_size, spat_step, spat_zero, spat_rota, slab_zero,
                spec_size, spec_step, spec_zero, lines,
                opacity, opacity_ratios,
                image, scube, rdata, rdata_cmp,
                ordata, ordata_cmp, vdata_cmp, ddata_cmp, vdweight_cmp,
                rdata_scale, image_scale);
        // (a cone writes three fields: j, v and s)
        Kernels::gmodel_smcone_evaluate(
                a, g, make_fields_args(true, 0, g, fields, 3));
    }

    static void
    mcdisk_evaluate(
            const Disk& disk,
            ConstData cloud_flux, unsigned int seed,
            int nclouds, Ints ncloudscsum, Bools has_analytical_integral,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota, int slab_zero,
            int spec_size, T spec_step, T spec_zero,
            Lines lines,
            ConstData opacity, Ratios opacity_ratios,
            Data image, Data scube,
            Data rdata, Data rdata_cmp,
            Data ordata, Data ordata_cmp,
            Data vdata_cmp, Data ddata_cmp, Data vdweight_cmp,
            Data fields,
            T rdata_scale, T image_scale)
    {
        const typename Kernels::DeviceScope scope(disk.rnodes.device_id());
        const auto a = make_disk_args(disk);
        const auto g = make_grid_args(
                disk.rht.has_value(), spat_size, spat_step, spat_zero,
                spat_rota, slab_zero, spec_size, spec_step, spec_zero, lines,
                opacity, opacity_ratios,
                image, scube, rdata, rdata_cmp,
                ordata, ordata_cmp, vdata_cmp, ddata_cmp, vdweight_cmp,
                rdata_scale, image_scale);
        require(disk.rht.has_value(),
                "a Monte Carlo disk needs density height traits");
        // (with the scratch of the moments of the clouds, a Monte Carlo
        // disk writes four fields)
        const auto f = make_fields_args(
                true, fields_lines(a, g), g, fields, 4);
        require(has_analytical_integral.shape(0) == size_t(a.rpt.n),
                "has_analytical_integral needs one value for each density "
                "trait");
        require(cloud_flux.shape(0) == ncloudscsum.shape(0),
                "cloud_flux needs one value for each pool of clouds");
        MCDiskArgs<T> mc;
        mc.cloud_flux = cloud_flux.data();
        mc.seed = seed;
        mc.nclouds = nclouds;
        mc.ncloudscsum = ncloudscsum.data();
        mc.ncloudscsum_len = int(ncloudscsum.shape(0));
        mc.has_analytical_integral = has_analytical_integral.data();
        Kernels::gmodel_mcdisk_evaluate(a, g, mc, f);
    }

    // Turn the optical depth of each voxel of an opacity cube (in (x, y, z)
    // order, of size spat_size) into the optical depth from the viewer to
    // the far side of the voxel, in place, as the disk kernels read it (see
    // gbkfit::gmodel_opacity_accumulate)
    static void
    opacity_accumulate(std::array<int, 3> spat_size, Data opacity)
    {
        const typename Kernels::DeviceScope scope(opacity.device_id());
        require(opacity.size()
                == size_t(spat_size[0]) * spat_size[1] * spat_size[2],
                "opacity has the wrong number of elements");
        Kernels::gmodel_opacity_accumulate(
                spat_size[0], spat_size[1], spat_size[2], opacity.data());
    }

    // March along each line of sight of the grid through the planes of a
    // slab, adding the light of the emitters that reaches the viewer to
    // out, the image or the spectral cube (see gbkfit::TransferArgs and
    // gbkfit::gmodel_transfer_march), at the given number of sub-channels
    // of each of its channels (spec_size, spec_step and spec_zero are those
    // of out)
    static void
    transfer_march(
            const std::vector<Emitter>& emitters,
            std::array<int, 3> spat_size, int slab_zero, int depth_size,
            bool image, int spec_size, T spec_step, T spec_zero,
            int subchannels,
            ConstData opacity, Data opacity_front, Data tau, Data out)
    {
        const typename Kernels::DeviceScope scope(out.device_id());
        TransferArgs<T> a;
        for (int i = 0; i < 3; ++i)
            a.spat_size[i] = spat_size[i];
        a.slab_zero = slab_zero;
        a.depth_size = depth_size;
        require(slab_zero >= 0 && slab_zero + spat_size[2] <= depth_size,
                "the slab must be within the depth axis");
        require(!image || spec_size == 1, "an image has one channel");
        require(subchannels >= 1
                && subchannels <= TRANSFER_SUBCHANNEL_NUM_MAX
                && (!image || subchannels == 1),
                "subchannels must be from 1 to "
                + std::to_string(TRANSFER_SUBCHANNEL_NUM_MAX)
                + ", and 1 in an image");
        // (the centre of the first sub-channel is half a sub-channel above
        // the lower edge of the first channel)
        a.image = image;
        a.subchannels = subchannels;
        a.spec_size = spec_size * subchannels;
        a.spec_step = spec_step / subchannels;
        a.spec_zero = spec_zero
                + (T{1} / (2 * subchannels) - T{0.5}) * spec_step;
        const size_t nimage = size_t(spat_size[0]) * spat_size[1];
        const size_t nslab = nimage * spat_size[2];
        require(out.size() == nimage * spec_size,
                "out has the wrong number of elements");
        a.out = out.data();

        // The dust: images in front of the slab, and cubes of the slab if
        // it has dust of its own
        require(!opacity.is_valid() || opacity_front.is_valid(),
                "opacity needs opacity_front");
        if (opacity_front.is_valid()) {
            const size_t nopacity = opacity_front.size() / nimage;
            require(nopacity >= 1 && nopacity <= size_t(OPACITY_NUM_MAX)
                    && opacity_front.size() == nopacity * nimage,
                    "opacity_front needs from 1 to "
                    + std::to_string(OPACITY_NUM_MAX) + " images");
            require(!opacity.is_valid()
                    || opacity.size() == nopacity * nslab,
                    "opacity needs a cube of the slab for each image of "
                    "opacity_front");
            a.nopacity = int(nopacity);
            a.opacity = data(opacity);
            a.opacity_front = opacity_front.data();
        }

        // The optical depth of the lines in front of the slab, with line
        // opacity
        require(!tau.is_valid() || tau.size() == nimage * a.spec_size,
                "tau needs a value for each sub-channel of each spaxel");
        a.tau = data(tau);

        require(emitters.size() <= size_t(TRANSFER_EMITTER_NUM_MAX),
                "the march takes at most "
                + std::to_string(TRANSFER_EMITTER_NUM_MAX) + " emitters");
        a.nemitters = int(emitters.size());
        for (size_t e = 0; e < emitters.size(); ++e)
        {
            const Emitter& em = emitters[e];
            const auto& f = em.fields;
            require(em.layer
                    ? f.ndim() == 3 && f.shape(0) == 4
                      && f.shape(1) == size_t(spat_size[1])
                      && f.shape(2) == size_t(spat_size[0])
                    : f.ndim() == 4 && f.shape(0) >= 3
                      && f.shape(1) == size_t(spat_size[2])
                      && f.shape(2) == size_t(spat_size[1])
                      && f.shape(3) == size_t(spat_size[0]),
                    "the fields of a layer have the shape (4, ny, nx), and "
                    "those of a volume (nslots >= 3, nz, ny, nx) on the "
                    "planes of the slab");
            const size_t nlines = em.lines.shape(0);
            require(em.lines.shape(1) == 3 && nlines >= 1
                    && (!image || nlines == 1),
                    "an emitter needs lines of shape (nlines, 3), and one "
                    "row in an image");
            require(!em.line_opacity
                    || (!em.layer && tau.is_valid()
                        && f.shape(0) >= 3 + nlines),
                    "an emitter with line opacity needs to be a volume with "
                    "a field of each line, and the march tau");
            require(em.opacity_ratios.is_valid() == opacity_front.is_valid()
                    && (!opacity_front.is_valid()
                        || (em.opacity_ratios.shape(0) == nlines
                            && em.opacity_ratios.shape(1)
                               == size_t(a.nopacity))),
                    "with dust (opacity_front), an emitter needs a row of "
                    "opacity_ratios for each line, and a column for each "
                    "image of opacity_front");
            a.emitters[e] = {
                    em.layer, em.line_opacity, f.data(), int(nlines),
                    em.lines.data(), data(em.opacity_ratios)};
        }
        Kernels::gmodel_transfer_march(a);
    }

    // Evaluate each polar trait of one kind of a disk ('rpt', 'vpt',
    // 'dpt', 'zpt' or 'spt') at the points (r, theta) of its plane,
    // into out (a row for each trait), as the disk kernels evaluate them
    // (see gbkfit::gmodel_disk_traits_point). incl projects the velocity
    // traits on the line of sight (without it, their speeds), and is for
    // them only.
    static void
    disk_ptraits_evaluate(
            const Disk& disk, const std::string& kind,
            Values r, Values theta, TraitValues out, std::optional<T> incl)
    {
        const typename Kernels::DeviceScope scope(out.device_id());
        auto a = make_traits_args(disk, kind, true, r, out);
        require(theta.shape(0) == r.shape(0),
                "theta needs one value for each point");
        require(!incl.has_value() || a.kind == TraitKind::VPT,
                "incl is for the velocity traits ('vpt') only");
        a.theta = theta.data();
        a.project = incl.has_value();
        a.incl = incl.value_or(0);
        Kernels::gmodel_disk_traits_evaluate(a);
    }

    // Evaluate each height trait of one kind of a disk ('rht', 'vht' or
    // 'dht') at the points (r, z) of its plane (z: the height above the
    // midplane), into out (a row for each trait), as the disk kernels
    // evaluate them (see gbkfit::gmodel_disk_traits_point)
    static void
    disk_htraits_evaluate(
            const Disk& disk, const std::string& kind,
            Values r, Values z, TraitValues out)
    {
        const typename Kernels::DeviceScope scope(out.device_id());
        auto a = make_traits_args(disk, kind, false, r, out);
        require(z.shape(0) == r.shape(0), "z needs one value for each point");
        a.z = z.data();
        Kernels::gmodel_disk_traits_evaluate(a);
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
                        const Traits* opt,
                        Array<Device, const int, nb::ndim<1>> vpt_directions,
                        Array<Device, const int, nb::ndim<1>> dpt_directions,
                        Array<Device, const int, nb::ndim<1>> opt_rows,
                        Values posa_incl_sincos, Values deq,
                        bool deq_vertical) {
                    auto optional = [](const Traits* traits) {
                        return traits ? std::optional(*traits) : std::nullopt;
                    };
                    new (self) Disk{
                            loose, tilted, rnodes, vsys, xpos, ypos, posa,
                            incl, optional(rpt), optional(rht),
                            optional(vpt), optional(vht), optional(dpt),
                            optional(dht), optional(zpt), optional(spt),
                            optional(opt), vpt_directions, dpt_directions,
                            opt_rows, posa_incl_sincos, deq, deq_vertical};
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
                nb::arg("opt").none() = nb::none(),
                nb::arg("vpt_directions").noconvert().none() = nb::none(),
                nb::arg("dpt_directions").noconvert().none() = nb::none(),
                nb::arg("opt_rows").noconvert().none() = nb::none(),
                nb::arg("posa_incl_sincos").noconvert().none()
                        = nb::none(),
                nb::arg("deq").noconvert().none() = nb::none(),
                nb::arg("deq_vertical") = false);

        nb::class_<Cone>(m, ("Cone" + suffix).c_str())
                .def("__init__", [](
                        Cone* self, bool positive, bool negative,
                        Values rnodes, Values vsys, Values xpos, Values ypos,
                        Values posa, Values incl, Values wall,
                        const Traits* rpt, const Traits* rht,
                        const Traits* vpt, const Traits* vht,
                        const Traits* dpt, const Traits* dht,
                        const Traits* spt,
                        Array<Device, const int, nb::ndim<1>>
                                vpt_directions) {
                    auto optional = [](const Traits* traits) {
                        return traits ? std::optional(*traits) : std::nullopt;
                    };
                    new (self) Cone{
                            positive, negative, rnodes, vsys, xpos, ypos,
                            posa, incl, wall, optional(rpt), optional(rht),
                            optional(vpt), optional(vht), optional(dpt),
                            optional(dht), optional(spt), vpt_directions};
                },
                nb::arg("positive"), nb::arg("negative"),
                nb::arg("rnodes").noconvert(),
                nb::arg("vsys").noconvert().none() = nb::none(),
                nb::arg("xpos").noconvert(),
                nb::arg("ypos").noconvert(),
                nb::arg("posa").noconvert(),
                nb::arg("incl").noconvert(),
                nb::arg("wall").noconvert(),
                nb::arg("rpt").none() = nb::none(),
                nb::arg("rht").none() = nb::none(),
                nb::arg("vpt").none() = nb::none(),
                nb::arg("vht").none() = nb::none(),
                nb::arg("dpt").none() = nb::none(),
                nb::arg("dht").none() = nb::none(),
                nb::arg("spt").none() = nb::none(),
                nb::arg("vpt_directions").noconvert().none() = nb::none());

        nb::class_<Emitter>(m, ("TransferEmitter" + suffix).c_str())
                .def("__init__", [](
                        Emitter* self, bool layer, bool line_opacity,
                        typename Emitter::Fields fields,
                        typename Emitter::Rows lines,
                        typename Emitter::Rows opacity_ratios) {
                    new (self) Emitter{
                            layer, line_opacity, fields, lines,
                            opacity_ratios};
                },
                nb::arg("layer"), nb::arg("line_opacity"),
                nb::arg("fields").noconvert(),
                nb::arg("lines").noconvert(),
                nb::arg("opacity_ratios").noconvert().none() = nb::none());

        auto evaluate_args = [](auto&&... first) {
            return std::make_tuple(
                    first...,
                    nb::arg("spat_size"), nb::arg("spat_step"),
                    nb::arg("spat_zero"), nb::arg("spat_rota"),
                    nb::arg("slab_zero") = 0,
                    nb::arg("spec_size"),
                    nb::arg("spec_step"), nb::arg("spec_zero"),
                    nb::arg("lines").noconvert().none() = nb::none(),
                    nb::arg("opacity").noconvert().none() = nb::none(),
                    nb::arg("opacity_ratios").noconvert().none() = nb::none(),
                    nb::arg("image").noconvert().none() = nb::none(),
                    nb::arg("scube").noconvert().none() = nb::none(),
                    nb::arg("rdata").noconvert().none() = nb::none(),
                    nb::arg("rdata_cmp").noconvert().none() = nb::none(),
                    nb::arg("ordata").noconvert().none() = nb::none(),
                    nb::arg("ordata_cmp").noconvert().none() = nb::none(),
                    nb::arg("vdata_cmp").noconvert().none() = nb::none(),
                    nb::arg("ddata_cmp").noconvert().none() = nb::none(),
                    nb::arg("vdweight_cmp").noconvert().none() = nb::none(),
                    nb::arg("fields").noconvert().none() = nb::none(),
                    nb::arg("rdata_scale") = T{1},
                    nb::arg("image_scale") = T{1});
        };

        auto cls = nb::class_<GModel>(m, ("GModel" + suffix).c_str())
                .def(nb::init<>());
        std::apply([&](auto&&... args) {
            cls.def_static(
                    "smdisk_evaluate", &smdisk_evaluate, ReleaseGil(), args...);
        }, evaluate_args(nb::arg("disk")));
        std::apply([&](auto&&... args) {
            cls.def_static(
                    "smcone_evaluate", &smcone_evaluate, ReleaseGil(), args...);
        }, evaluate_args(nb::arg("cone")));
        std::apply([&](auto&&... args) {
            cls.def_static(
                    "mcdisk_evaluate", &mcdisk_evaluate, ReleaseGil(), args...);
        }, evaluate_args(
                nb::arg("disk"), nb::arg("cloud_flux").noconvert(), nb::arg("seed"),
                nb::arg("nclouds"),
                nb::arg("ncloudscsum").noconvert(),
                nb::arg("has_analytical_integral").noconvert()));
        cls.def_static("opacity_accumulate", &opacity_accumulate,
                ReleaseGil(),
                nb::arg("spat_size"), nb::arg("opacity").noconvert());
        cls.def_static("transfer_march", &transfer_march,
                ReleaseGil(),
                nb::arg("emitters"), nb::arg("spat_size"),
                nb::arg("slab_zero"), nb::arg("depth_size"),
                nb::arg("image"), nb::arg("spec_size"),
                nb::arg("spec_step"), nb::arg("spec_zero"),
                nb::arg("subchannels"),
                nb::arg("opacity").noconvert().none() = nb::none(),
                nb::arg("opacity_front").noconvert().none() = nb::none(),
                nb::arg("tau").noconvert().none() = nb::none(),
                nb::arg("out").noconvert());
        cls.def_static("disk_ptraits_evaluate", &disk_ptraits_evaluate,
                ReleaseGil(),
                nb::arg("disk"), nb::arg("kind"),
                nb::arg("r").noconvert(), nb::arg("theta").noconvert(),
                nb::arg("out").noconvert(),
                nb::arg("incl").none() = nb::none());
        cls.def_static("disk_htraits_evaluate", &disk_htraits_evaluate,
                ReleaseGil(),
                nb::arg("disk"), nb::arg("kind"),
                nb::arg("r").noconvert(), nb::arg("z").noconvert(),
                nb::arg("out").noconvert());
    }

private:

    // The kinds of traits of a disk: their names (those of the Python
    // disks), whether they are polar or height traits, and their sets
    struct Kind
    {
        const char* name;
        TraitKind kind;
        bool polar;
        typename Disk::Traits Disk::* set;
    };

    static constexpr std::array<Kind, 9> KINDS = {{
            {"rpt", TraitKind::RPT, true, &Disk::rpt},
            {"rht", TraitKind::RHT, false, &Disk::rht},
            {"vpt", TraitKind::VPT, true, &Disk::vpt},
            {"vht", TraitKind::VHT, false, &Disk::vht},
            {"dpt", TraitKind::DPT, true, &Disk::dpt},
            {"dht", TraitKind::DHT, false, &Disk::dht},
            {"zpt", TraitKind::ZPT, true, &Disk::zpt},
            {"spt", TraitKind::SPT, true, &Disk::spt},
            {"opt", TraitKind::OPT, true, &Disk::opt}}};

    // The arguments of an evaluation of the traits of one kind of a disk,
    // polar traits (polar) or height traits, at the points of radii r,
    // into out (see gbkfit::DiskTraitsArgs)
    static DiskTraitsArgs<T>
    make_traits_args(
            const Disk& disk, const std::string& kind, bool polar,
            const Values& r, const TraitValues& out)
    {
        std::string names;
        for (const Kind& k : KINDS) {
            if (k.polar == polar) {
                names += (names.empty() ? "'" : ", '") + std::string(k.name)
                        + "'";
            }
        }
        const auto it = std::find_if(
                KINDS.begin(), KINDS.end(), [&](const Kind& k) {
                    return k.polar == polar && kind == k.name;
                });
        require(it != KINDS.end(),
                "kind must be one of " + names + "; it is '" + kind + "'");
        const auto& set = disk.*(it->set);
        require(set.has_value(),
                "the disk has no traits of kind '" + kind + "'");

        DiskTraitsArgs<T> a;
        a.kind = it->kind;
        a.nrnodes = int(disk.rnodes.shape(0));
        a.rnodes = disk.rnodes.data();
        require(a.nrnodes >= 2, "a disk needs at least two radial nodes");
        a.traits = set->raw();
        if (a.kind == TraitKind::VPT)
            a.directions = vpt_directions(disk);
        a.npoints = int(r.shape(0));
        a.r = r.data();
        require(out.shape(0) == size_t(a.traits.n)
                && out.shape(1) == r.shape(0),
                "out needs a row for each trait, and a value for each point");
        a.out = out.data();
        return a;
    }

    // The direction of each velocity polar trait of a disk or a cone (see
    // gbkfit::DiskArgs and gbkfit::ConeArgs): given with those traits, and
    // only with them
    template<typename Arrays>
    static const int*
    vpt_directions(const Arrays& arrays)
    {
        require(arrays.vpt_directions.is_valid() == arrays.vpt.has_value()
                && (!arrays.vpt || arrays.vpt_directions.shape(0)
                                   == arrays.vpt->uids.shape(0)),
                "vpt_directions needs a direction for each velocity polar "
                "trait, and goes with them only");
        return data(arrays.vpt_directions);
    }

    // The trait set of the arrays of a set, if any (none: no traits)
    static TraitSet<T>
    trait_set(const typename Disk::Traits& set)
    {
        return set ? set->raw() : TraitSet<T>{};
    }

    // The arguments of a disk (see gbkfit::DiskArgs)
    static DiskArgs<T>
    make_disk_args(const Disk& disk)
    {
        DiskArgs<T> a;
        a.loose = disk.loose;
        a.tilted = disk.tilted;
        a.nrnodes = int(disk.rnodes.shape(0));
        a.rnodes = disk.rnodes.data();
        require(a.nrnodes >= 2, "a disk needs at least two radial nodes");
        a.vsys = data(disk.vsys);
        a.xpos = disk.xpos.data();
        a.ypos = disk.ypos.data();
        a.posa = disk.posa.data();
        a.incl = disk.incl.data();
        a.posa_incl_sincos = data(disk.posa_incl_sincos);
        const size_t nloose = disk.loose ? a.nrnodes : 1;
        const size_t ntilted = disk.tilted ? a.nrnodes : 1;
        require((!disk.vsys.is_valid() || disk.vsys.shape(0) == nloose)
                && disk.xpos.shape(0) == nloose
                && disk.ypos.shape(0) == nloose
                && disk.posa.shape(0) == ntilted
                && disk.incl.shape(0) == ntilted,
                "the geometry needs a value for each node of a loose or "
                "tilted disk, and one value otherwise");
        require(!(disk.loose || disk.tilted)
                || (disk.posa_incl_sincos.is_valid()
                    && disk.posa_incl_sincos.shape(0) == 4 * ntilted),
                "a loose or tilted disk needs posa_incl_sincos: sin posa, "
                "cos posa, sin incl and cos incl of each value of posa");

        require(disk.rpt.has_value(), "a disk needs density traits");
        a.rpt = trait_set(disk.rpt);
        a.rht = trait_set(disk.rht);
        a.vpt = trait_set(disk.vpt);
        a.vpt_directions = vpt_directions(disk);
        a.vht = trait_set(disk.vht);
        a.dpt = trait_set(disk.dpt);
        require(!disk.dpt_directions.is_valid()
                || (disk.dpt && disk.dpt_directions.shape(0)
                                == disk.dpt->uids.shape(0)),
                "dpt_directions needs a direction for each dispersion polar "
                "trait, and goes with them only");
        a.dpt_directions = data(disk.dpt_directions);
        a.dht = trait_set(disk.dht);
        a.zpt = trait_set(disk.zpt);
        a.spt = trait_set(disk.spt);
        require((!disk.rht || a.rht.n == a.rpt.n)
                && (!disk.vht || a.vht.n == a.vpt.n)
                && (!disk.dht || a.dht.n == a.dpt.n),
                "height traits must have as many traits as polar traits");
        // (a thick disk multiplies each polar trait by its height trait)
        require(!disk.rht || ((!disk.vpt || disk.vht)
                              && (!disk.dpt || disk.dht)),
                "a thick disk (with density height traits) needs the height "
                "traits of its velocity and dispersion traits");
        a.opt = trait_set(disk.opt);
        require(!disk.opt || (disk.opt_rows.is_valid()
                              && disk.opt_rows.shape(0) == size_t(a.opt.n)),
                "the traits of line opacity (opt) need the row of the line "
                "of each (opt_rows)");
        a.opt_rows = data(disk.opt_rows);
        require(!disk.deq.is_valid()
                || disk.deq.shape(0) == size_t(a.nrnodes),
                "deq needs a value for each node");
        a.deq = data(disk.deq);
        require(!disk.deq_vertical || disk.deq.is_valid(),
                "deq_vertical needs deq");
        a.deq_vertical = disk.deq_vertical;
        return a;
    }

    // The number of planes of line opacity in the fields of a thick disk:
    // one for each line with line opacity, none without (see
    // gbkfit::gmodel_fields_lines)
    static int
    fields_lines(const DiskArgs<T>& a, const GridArgs<T>& g)
    {
        return a.opt.uids ? g.nlines : 0;
    }

    // The arguments of a cone (see gbkfit::ConeArgs)
    static ConeArgs<T>
    make_cone_args(const Cone& cone)
    {
        ConeArgs<T> a;
        a.positive = cone.positive;
        a.negative = cone.negative;
        a.nrnodes = int(cone.rnodes.shape(0));
        a.rnodes = cone.rnodes.data();
        require(a.nrnodes >= 2, "a cone needs at least two radial nodes");
        require(!cone.vsys.is_valid() || cone.vsys.shape(0) == 1,
                "vsys needs one value");
        require(cone.xpos.shape(0) == 1 && cone.ypos.shape(0) == 1
                && cone.posa.shape(0) == 1 && cone.incl.shape(0) == 1
                && cone.wall.shape(0) == 1,
                "the geometry of a cone needs one value of each parameter");
        a.vsys = data(cone.vsys);
        a.xpos = cone.xpos.data();
        a.ypos = cone.ypos.data();
        a.posa = cone.posa.data();
        a.incl = cone.incl.data();
        a.wall = cone.wall.data();

        require(cone.rpt.has_value() && cone.rht.has_value(),
                "a cone needs density traits and density height traits");
        a.rpt = trait_set(cone.rpt);
        a.rht = trait_set(cone.rht);
        a.vpt = trait_set(cone.vpt);
        a.vpt_directions = vpt_directions(cone);
        a.vht = trait_set(cone.vht);
        a.dpt = trait_set(cone.dpt);
        a.dht = trait_set(cone.dht);
        a.spt = trait_set(cone.spt);
        require(a.rht.n == a.rpt.n
                && cone.vpt.has_value() == cone.vht.has_value()
                && a.vht.n == a.vpt.n
                && cone.dpt.has_value() == cone.dht.has_value()
                && a.dht.n == a.dpt.n,
                "a cone needs a height trait for each polar trait");
        return a;
    }

    // The grid that a disk or a cone (thick, or a thin disk) is evaluated
    // on, its lines, the opacity that dims it and the outputs (see
    // gbkfit::GridArgs)
    static GridArgs<T>
    make_grid_args(
            bool thick,
            std::array<int, 3> spat_size,
            std::array<T, 3> spat_step,
            std::array<T, 3> spat_zero,
            T spat_rota, int slab_zero,
            int spec_size, T spec_step, T spec_zero,
            const Lines& lines,
            const ConstData& opacity, const Ratios& opacity_ratios,
            const Data& image, const Data& scube,
            const Data& rdata, const Data& rdata_cmp,
            const Data& ordata, const Data& ordata_cmp,
            const Data& vdata_cmp, const Data& ddata_cmp,
            const Data& vdweight_cmp,
            T rdata_scale, T image_scale)
    {
        GridArgs<T> g;
        for (int i = 0; i < 3; ++i) {
            g.spat_size[i] = spat_size[i];
            g.spat_step[i] = spat_step[i];
            g.spat_zero[i] = spat_zero[i];
        }
        g.spat_rota = spat_rota;
        g.slab_zero = slab_zero;
        require(slab_zero >= 0 && (thick || slab_zero == 0),
                "slab_zero must be at least 0, and 0 for a thin disk, which "
                "is evaluated on the whole depth axis");
        g.spec_size = spec_size;
        g.spec_step = spec_step;
        g.spec_zero = spec_zero;
        require(!lines.is_valid() || lines.shape(1) == 3,
                "lines must have shape (nlines, 3)");
        require(!scube.is_valid() || (lines.is_valid() && lines.shape(0) > 0),
                "a spectral cube needs at least one line");
        g.nlines = lines.is_valid() ? int(lines.shape(0)) : 0;
        g.lines = data(lines);

        const size_t nimage = size_t(spat_size[0]) * spat_size[1];
        const size_t nspat = nimage * spat_size[2];
        const size_t nscube = nimage * spec_size;
        auto output = [](const auto& array, size_t size, const char* name) {
            require(!array.is_valid() || array.size() == size,
                    std::string(name) + " has the wrong number of elements");
            return data(array);
        };
        // The cubes of opacity, and a row of their ratios for each line of
        // the spectral cube, or one for the image
        require(opacity.is_valid() == opacity_ratios.is_valid(),
                "opacity and opacity_ratios go together");
        if (opacity.is_valid()) {
            const size_t nopacity = opacity_ratios.shape(1);
            const size_t nrows = scube.is_valid() ? size_t(g.nlines) : 1;
            require(nopacity >= 1 && nopacity <= size_t(OPACITY_NUM_MAX),
                    "opacity_ratios needs a column for each cube of "
                    "opacity, of which there are at most "
                    + std::to_string(OPACITY_NUM_MAX));
            require(opacity.size() == nopacity * nspat,
                    "opacity needs a cube for each column of "
                    "opacity_ratios");
            require(opacity_ratios.shape(0) == nrows,
                    "opacity_ratios needs a row for each line of the "
                    "spectral cube, or one row for the image");
            g.nopacity = int(nopacity);
            g.opacity = opacity.data();
            g.opacity_ratios = opacity_ratios.data();
        }
        g.image = output(image, nimage, "image");
        g.scube = output(scube, nscube, "scube");
        g.rdata = output(rdata, nspat, "rdata");
        g.rdata_cmp = output(rdata_cmp, nspat, "rdata_cmp");
        g.ordata = output(ordata, nspat, "ordata");
        g.ordata_cmp = output(ordata_cmp, nspat, "ordata_cmp");
        g.vdata_cmp = output(vdata_cmp, nspat, "vdata_cmp");
        g.ddata_cmp = output(ddata_cmp, nspat, "ddata_cmp");
        g.vdweight_cmp = output(vdweight_cmp, nspat, "vdweight_cmp");
        g.rdata_scale = rdata_scale;
        g.image_scale = image_scale;
        return g;
    }

    // The fields of the march that a component on the grid g writes
    // instead of its light and its opacity (see gbkfit::FieldsArgs):
    // volume_fields planes of the 3d grid for a thick one (and nk planes
    // of line opacity), four of the x and y axes for a thin one
    static FieldsArgs<T>
    make_fields_args(
            bool thick, int nk, const GridArgs<T>& g, const Data& fields,
            int volume_fields)
    {
        FieldsArgs<T> f;
        require(!fields.is_valid()
                || !(g.image || g.scube || g.opacity),
                "fields replace the image, the spectral cube and the "
                "opacity");
        // (there are at most as many lines with line opacity as traits)
        require(!fields.is_valid() || nk <= TRAIT_NUM_MAX,
                "line opacity takes at most " + std::to_string(TRAIT_NUM_MAX)
                + " lines");
        const size_t nimage = size_t(g.spat_size[0]) * g.spat_size[1];
        const size_t nspat = nimage * g.spat_size[2];
        const size_t size = thick ? (volume_fields + nk) * nspat
                                  : 4 * nimage;
        require(!fields.is_valid() || fields.size() == size,
                "fields has the wrong number of elements");
        f.fields = data(fields);
        return f;
    }
};

} // namespace gbkfit::bindings
