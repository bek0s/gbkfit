#pragma once

#include "gbkfit/bindings/arrays.hpp"
#include "gbkfit/bindings/gil.hpp"
#include "gbkfit/orbits/orbit.hpp"

namespace gbkfit::bindings {

// The orbit operations of a native module: the potential of a galaxy and
// the orbits in it (see gbkfit::PotentialArgs and gbkfit::OrbitArgs).
// Kernels launches the kernels of the module, on the memory of Device:
// each function on the device of the arrays it writes.
template<typename T, typename Device, typename Kernels>
struct Orbits
{
    using Table = Array<Device, const T, nb::ndim<3>>;
    using Range = Array<Device, const T, nb::shape<2>>;
    using Rows = Array<Device, const T, nb::ndim<2>>;
    using Values = Array<Device, const T, nb::ndim<1>>;
    using Counts = Array<Device, const int, nb::ndim<1>>;
    using Offsets = Array<Device, const long long, nb::ndim<1>>;
    using Out1 = Array<Device, T, nb::ndim<1>>;
    using Out2 = Array<Device, T, nb::ndim<2>>;
    using Out3 = Array<Device, T, nb::ndim<3>>;
    using Steps = Array<Device, int, nb::ndim<1>>;

    // The potential of a table of shape (nr, nb, 4) and range (see
    // gbkfit::PotentialArgs)
    static PotentialArgs<T>
    potential_args(const Table& table, const Range& range)
    {
        require(table.shape(0) >= 2 && table.shape(1) >= 2
                && table.shape(2) == 4,
                "table must have the shape (nr, nb, 4), nr and nb at "
                "least 2");
        PotentialArgs<T> p;
        p.nr = int(table.shape(0));
        p.nb = int(table.shape(1));
        p.range = range.data();
        p.values = table.data();
        return p;
    }

    // The potential of a table and range at points (rows of x, y, z) into
    // potential (one value per point), and the force there into forces
    // (a row per point; see gbkfit::orbits_potential_point)
    static void
    potential(
            Table table, Range range, Rows points, Out1 potential,
            Out2 forces)
    {
        const typename Kernels::DeviceScope scope(potential.device_id());
        require(points.shape(1) == 3, "points must have rows of x, y, z");
        require(potential.shape(0) == points.shape(0),
                "potential needs one value per point");
        require(forces.shape(0) == points.shape(0) && forces.shape(1) == 3,
                "forces needs a row of 3 values per point");
        Kernels::orbits_potential(
                potential_args(table, range), int(points.shape(0)),
                points.data(), potential.data(), forces.data());
    }

    // Integrate orbits in the potential of a table and range and record
    // them (see gbkfit::OrbitArgs): from starts (rows of x, y, z, vx, vy,
    // vz), each for turns circular periods; their cells (between the
    // edges cell_r and cell_b) into cells (of shape (norbits, ncell_r,
    // ncell_b), norbits twice the orbits with twins); on the sky (the
    // arguments from grid_sizes to shares; grid_sizes and grids, the 6
    // values of the map of each, a row per grid), their moments in the
    // series of the targets (target_v and target_s) into moments (of
    // shape (norbits, ntargets, order + 2)), and the spectrum of their
    // light (light) on the channels of the map of their velocities
    // (channel_scale and channel_offset) into spectra (of shape
    // (nchannels, ntargets), added to); the moments of their light in the
    // cells into cell_moments (of shape (ncell_r, ncell_b, 9), added to;
    // with the cells); and their steps and energies (a row per orbit, at
    // its start and its end). The cells, the sky, the moments, the
    // spectra, the cell moments and the steps and energies are optional
    // (None). Their samples are kept in samples (of shape (nchunk,
    // nsamples, 6)), nchunk orbits at a time.
    static void
    integrate(
            Table table, Range range,
            Rows starts, T turns, T accuracy, int max_steps,
            int nsamples, int nphi, bool twins, T posa, T incl,
            Values cell_r, Values cell_b, Out3 cells,
            Array<Device, const int, nb::ndim<2>> grid_sizes, Rows grids,
            Offsets grid_cells, Offsets offsets, Counts targets,
            Values shares,
            Values target_v, Values target_s, int order, Out3 moments,
            Values light, T channel_scale, T channel_offset, Out2 spectra,
            Out3 cell_moments, Steps steps, Out2 energies, Out3 samples)
    {
        const typename Kernels::DeviceScope scope(starts.device_id());
        OrbitArgs<T> a;
        a.potential = potential_args(table, range);

        a.norbits = int(starts.shape(0));
        require(starts.shape(1) == 6,
                "starts must have rows of x, y, z, vx, vy, vz");
        require(turns > 0 && accuracy > 0 && max_steps > 0 && nsamples > 0
                && nphi > 0,
                "turns, accuracy, max_steps, nsamples and nphi must be "
                "positive");
        a.starts = starts.data();
        a.turns = turns;
        a.accuracy = accuracy;
        a.max_steps = max_steps;
        a.nsamples = nsamples;
        a.nphi = nphi;
        a.twins = twins;
        const size_t norbits = starts.shape(0) * (twins ? 2 : 1);

        a.posa = posa;
        a.incl = incl;

        require(cells.is_valid() == cell_r.is_valid()
                && cells.is_valid() == cell_b.is_valid(),
                "cells, cell_r and cell_b go together");
        if (cells.is_valid())
        {
            require(cell_r.shape(0) >= 2 && cell_b.shape(0) >= 2,
                    "cell_r and cell_b need two edges at least");
            a.ncell_r = int(cell_r.shape(0)) - 1;
            a.ncell_b = int(cell_b.shape(0)) - 1;
            require(cells.shape(0) == norbits
                    && cells.shape(1) == size_t(a.ncell_r)
                    && cells.shape(2) == size_t(a.ncell_b),
                    "cells must have the shape (norbits, ncell_r, ncell_b)");
            a.cell_r = cell_r.data();
            a.cell_b = cell_b.data();
            a.cells = cells.data();
        }

        const bool sky = moments.is_valid() || spectra.is_valid();
        require(grid_sizes.is_valid() == sky && grids.is_valid() == sky
                && grid_cells.is_valid() == sky && offsets.is_valid() == sky
                && targets.is_valid() == sky && shares.is_valid() == sky,
                "the arrays of the sky go with moments or spectra");
        if (sky)
        {
            a.nsky = int(grid_sizes.shape(0));
            require(grid_sizes.shape(1) == 2
                    && grids.shape(0) == size_t(a.nsky)
                    && grids.shape(1) == 6
                    && grid_cells.shape(0) == size_t(a.nsky),
                    "grid_sizes, grids and grid_cells need a row per grid");
            require(targets.shape(0) == shares.shape(0),
                    "targets and shares must have the same length");
            a.grid_sizes = grid_sizes.data();
            a.grids = grids.data();
            a.grid_cells = grid_cells.data();
            a.offsets = offsets.data();
            a.targets = targets.data();
            a.shares = shares.data();
        }

        require(moments.is_valid() == target_v.is_valid()
                && moments.is_valid() == target_s.is_valid(),
                "moments, target_v and target_s go together");
        if (moments.is_valid())
        {
            a.ntargets = int(target_v.shape(0));
            require(target_s.shape(0) == target_v.shape(0),
                    "target_v and target_s need one value per target");
            require(order >= 0, "order must not be negative");
            require(moments.shape(0) == norbits
                    && moments.shape(1) == size_t(a.ntargets)
                    && moments.shape(2) == size_t(order + 2),
                    "moments must have the shape (norbits, ntargets, "
                    "order + 2)");
            a.target_v = target_v.data();
            a.target_s = target_s.data();
            a.order = order;
            a.moments = moments.data();
        }

        require(light.is_valid()
                == (spectra.is_valid() || cell_moments.is_valid()),
                "light goes with spectra or cell moments");
        if (light.is_valid())
        {
            require(light.shape(0) == norbits,
                    "light needs one value per orbit");
            a.light = light.data();
        }
        if (cell_moments.is_valid())
        {
            require(cells.is_valid()
                    && cell_moments.shape(0) == size_t(a.ncell_r)
                    && cell_moments.shape(1) == size_t(a.ncell_b)
                    && cell_moments.shape(2) == ORBITS_CELL_MOMENTS,
                    "cell_moments go with the cells, of the shape "
                    "(ncell_r, ncell_b, 9)");
            a.cell_moments = cell_moments.data();
        }
        if (spectra.is_valid())
        {
            require(!moments.is_valid()
                    || spectra.shape(1) == size_t(a.ntargets),
                    "spectra and moments need the same targets");
            a.ntargets = int(spectra.shape(1));
            a.nchannels = int(spectra.shape(0));
            a.channel_scale = channel_scale;
            a.channel_offset = channel_offset;
            a.spectra = spectra.data();
        }

        require(!steps.is_valid() || steps.shape(0) == starts.shape(0),
                "steps needs one value per orbit");
        require(!energies.is_valid()
                || (energies.shape(0) == starts.shape(0)
                    && energies.shape(1) == 2),
                "energies needs a row of 2 values per orbit");
        a.steps = data(steps);
        a.energies = data(energies);

        require(samples.shape(0) >= 1
                && samples.shape(1) == size_t(nsamples)
                && samples.shape(2) == 6,
                "samples must have the shape (nchunk, nsamples, 6)");
        a.nchunk = int(samples.shape(0));
        a.samples = samples.data();

        Kernels::orbits_integrate(a);
    }

    static void
    bind(nb::module_& m, const std::string& suffix)
    {
        nb::class_<Orbits>(m, ("Orbits" + suffix).c_str())
                .def(nb::init<>())
                .def_static("potential", &potential,
                        ReleaseGil(),
                        nb::arg("table").noconvert(),
                        nb::arg("range").noconvert(),
                        nb::arg("points").noconvert(),
                        nb::arg("potential").noconvert(),
                        nb::arg("forces").noconvert())
                .def_static("integrate", &integrate,
                        ReleaseGil(),
                        nb::arg("table").noconvert(),
                        nb::arg("range").noconvert(),
                        nb::arg("starts").noconvert(),
                        nb::arg("turns"),
                        nb::arg("accuracy"),
                        nb::arg("max_steps"),
                        nb::arg("nsamples"),
                        nb::arg("nphi"),
                        nb::arg("twins"),
                        nb::arg("posa"),
                        nb::arg("incl"),
                        nb::arg("cell_r").noconvert().none(),
                        nb::arg("cell_b").noconvert().none(),
                        nb::arg("cells").noconvert().none(),
                        nb::arg("grid_sizes").noconvert().none(),
                        nb::arg("grids").noconvert().none(),
                        nb::arg("grid_cells").noconvert().none(),
                        nb::arg("offsets").noconvert().none(),
                        nb::arg("targets").noconvert().none(),
                        nb::arg("shares").noconvert().none(),
                        nb::arg("target_v").noconvert().none(),
                        nb::arg("target_s").noconvert().none(),
                        nb::arg("order"),
                        nb::arg("moments").noconvert().none(),
                        nb::arg("light").noconvert().none(),
                        nb::arg("channel_scale"),
                        nb::arg("channel_offset"),
                        nb::arg("spectra").noconvert().none(),
                        nb::arg("cell_moments").noconvert().none(),
                        nb::arg("steps").noconvert().none(),
                        nb::arg("energies").noconvert().none(),
                        nb::arg("samples").noconvert());
    }
};

} // namespace gbkfit::bindings
