"""
The evaluation of the components of a galaxy by radiative transfer along
each line of sight (TransferPlan; see the line of sight Transfer).
"""

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from ._component_plans import ComponentPlans
from ._units import DataUnit
from .base import DepthAxis, NativeGrid, OpacityCubes
from .lines import Band

if TYPE_CHECKING:
    from ._component_set import ComponentSet
    from .base import Component, ComponentPlan


__all__ = [
    'LineResolution',
    'TransferMember',
    'TransferPlan',
    'warn_resolution'
]


_log = logging.getLogger(__name__)


# The most emitters that the march takes (TRANSFER_EMITTER_NUM_MAX in
# constants.hpp)
_MAX_EMITTERS = 16

# The most sub-channels of a channel that the march takes
# (TRANSFER_SUBCHANNEL_NUM_MAX in constants.hpp)
MAX_SUBCHANNELS = 64

# The resolution of line opacity (see LineResolution): checked where the
# optical depth of the lines at a channel reaches _THIN_TAU, with a
# warning above the given ratios (see warn_resolution). In an edge-on
# disk of HI of optically thick lines (a peak of 180 K without opacity,
# 83 K with), the error of the model is about 1% of its peak at these
# ratios or less: it grows as the square of the spectral ratio (1.1% at
# 0.75, 0.3% at 0.5), and steeply with the depth ratio (0.5% at 2, 3% at
# 2.7, 10% at 4). The light without opacity takes the depth ratio too
# (its ratio of the light): 0.3% at 2, 6% at 3.4.
_THIN_TAU = 0.1
_MAX_SPECTRAL_RATIO = 0.75
_MAX_DEPTH_RATIO = 2.0


class LineResolution(NamedTuple):
    """
    How finely the evaluation of a galaxy on the grid of an observation
    resolves its lines (see ComponentSetPlan.evaluate and
    TransferPlan.evaluate). With line opacity, where the lines are not
    optically thin, the largest ratio of the width of the channels of the
    march (the sub-channels) to the dispersion of a line (spectral), and
    of the change of the velocity between adjacent voxels along a line of
    sight to their dispersion (depth): the march averages the optical
    depth of a voxel over a channel, and over a voxel (0 without line
    opacity). And for its light, the ratio of that change of the velocity
    to the width of the lines with the smoothing of the instrument, where
    the light is bright (light; see ComponentPlans.light_resolution): the
    lines of the voxels add up as a comb that the instrument smooths. All
    are discretisations of the model; lines of zero width, which no
    resolution resolves, are left out.
    """
    spectral: float
    depth: float
    light: float


def warn_resolution(
        resolution: LineResolution,
        channel: float,
        subchannels: int,
        depth_step: float
) -> None:
    """
    Log a warning for each way in which an evaluation, of channels of the
    given width and number of sub-channels and voxels of the given depth
    (arcsec), resolves the lines too coarsely (see LineResolution), with
    the setting that resolves it: for the light, unless the depth of line
    opacity has its warning (which is stricter).
    """
    if resolution.spectral > _MAX_SPECTRAL_RATIO:
        needed = math.ceil(
            subchannels * resolution.spectral / _MAX_SPECTRAL_RATIO)
        _log.warning(
            f"radiative transfer does not resolve line opacity in velocity: "
            f"the channels of the march (of {channel:g} / {subchannels} "
            f"sub-channel{'s' if subchannels > 1 else ''}) are up to "
            f"{resolution.spectral:.3g} times as wide as the dispersion of "
            f"the lines where they are not thin; raise the option "
            f"subchannels of the line of sight to "
            f"{min(needed, MAX_SUBCHANNELS)}")
    if resolution.depth > _MAX_DEPTH_RATIO:
        _log.warning(
            f"radiative transfer does not resolve line opacity along the "
            f"line of sight: the velocity changes by up to "
            f"{resolution.depth:.3g} dispersions between adjacent voxels "
            f"where the lines are not thin; lower the step of the depth "
            f"axis (the option depth: {{step: ...}} of the line of sight) "
            f"from {depth_step:g} to "
            f"{depth_step * _MAX_DEPTH_RATIO / resolution.depth:.3g} arcsec")
    elif resolution.light > _MAX_DEPTH_RATIO:
        _log.warning(
            f"the depth axis does not resolve the velocity of the light "
            f"along the line of sight: it changes by up to "
            f"{resolution.light:.3g} widths of the lines (with the smoothing "
            f"of the instrument) between adjacent voxels where the light is "
            f"bright; lower the step of the depth axis (the option depth: "
            f"{{step: ...}} of the line of sight) from {depth_step:g} to "
            f"{depth_step * _MAX_DEPTH_RATIO / resolution.light:.3g} arcsec")


class TransferMember(NamedTuple):
    """
    A galaxy that a march evaluates (see TransferPlan): its components,
    the indices of those whose light is evaluated (its emitters), how the
    cubes of opacity hold its dust (see opacity_cubes), its redshift, and
    its own depth axis, which holds it, centred on its plane (see
    Depth.axis).
    """
    component_set: 'ComponentSet'
    emitters: tuple[int, ...]
    cubes: OpacityCubes
    redshift: float
    depth: DepthAxis


class TransferPlan:
    """
    The evaluation of the components of one galaxy or more on the grid of
    an observation by radiative transfer along each line of sight: the
    absorbers write the optical depth of each voxel to the cubes of
    opacity, the emitters write the fields of their light (see
    ComponentPlan.evaluate_fields), and a march along each line of sight,
    front to back, adds the light of each channel that reaches the viewer
    to the image or the spectral cube. The march starts from the opacity
    in front of the galaxies, that of the galaxies in front of them, and
    adds their own, for the galaxies behind them (see evaluate).

    One galaxy is evaluated on its own depth axis. Several (that can pass
    through each other, e.g. a merger) are evaluated together, each at its
    position along the line of sight (see evaluate), on one depth axis as
    long as theirs together, so that the opacity of each dims the light of
    the others voxel by voxel. The stretch between the depth axes of
    galaxies that do not overlap is left out: it is empty, so the march
    gives the same light without it. It is left out in whole voxels, and
    the voxels keep the depths they have on the axis of the first galaxy:
    they sample each galaxy as they do apart.

    It evaluates the depth axis in slabs of planes, one after the other,
    to bound its memory: the absorbers and the volumes (e.g. thick disks)
    on the planes of each slab (the clouds of a Monte Carlo disk are drawn
    again for each, the same ones), the layers (e.g. thin disks, points)
    once. The parameters are those of ComponentPlans (but those of each
    galaxy: members; data_unit is that of all), the number of planes of a slab (slab; all if None),
    and the number of sub-channels of each channel that the march
    transfers the light of the channel at, if it carries line opacity
    (subchannels; the light of the channel is their mean), and the
    spectral smoothing of the instrument (km/s; see
    ComponentPlans.light_resolution; None for an image).

    Raises
    ------
    ConfigError
        If there are more emitters than the march takes.
    """

    def __init__(
            self,
            members: Sequence[TransferMember],
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            data_unit: DataUnit | None,
            slab: int | None,
            subchannels: int,
            smoothing: float | None
    ):
        nemitters = sum(len(member.emitters) for member in members)
        if nemitters > _MAX_EMITTERS:
            raise ConfigError(
                f"radiative transfer takes at most {_MAX_EMITTERS} components "
                f"that emit; there are {nemitters}")
        # The depth axis: that of the galaxy, or of several, as long as
        # theirs together and a margin of 2 voxels each (see _placement)
        depth = members[0].depth
        if len(members) > 1:
            depth = depth._replace(size=sum(
                member.depth.size + 2 for member in members))
        self._members = tuple(members)
        self._depth = depth
        self._plans = tuple(
            ComponentPlans(
                member.component_set, driver, grid, spectral, depth, dtype,
                member.emitters, member.cubes, lines, band, data_unit,
                member.redshift)
            for member in members)
        self._driver = driver
        self._dtype = dtype
        self._subchannels = subchannels
        self._smoothing = smoothing
        # (that of every galaxy: the cubes hold the dust of the scene)
        self._ncubes = self._plans[0].ncubes
        native_grid = self._plans[0].native_grid
        # The slabs: their first plane and their number of planes
        nz = native_grid['spat_size'][2]
        size = nz if slab is None else min(slab, nz)
        self._slabs = [(z0, min(size, nz - z0)) for z0 in range(0, nz, size)]
        slab_grid = _slab_grid(native_grid, 0, size)
        # The emitters of the galaxies, by the index of their galaxy and
        # their own
        self._keys = tuple(
            (m, i) for m, plans in enumerate(self._plans)
            for i in plans.emitters)
        # The lines of each emitter, in one buffer (which is copied to the
        # device once per evaluation): one row for each of its wavelengths
        # (those of its lines, or its band, at the redshift of its galaxy)
        nrows = [
            len(self._component(key).emission().wavelengths(
                lines, band, members[key[0]].redshift))
            for key in self._keys]
        ends = np.cumsum(nrows)
        self._rows = {
            key: slice(end - n, end)
            for key, n, end in zip(self._keys, nrows, ends)}
        self._lines = driver.mem_alloc_s((sum(nrows), 3), dtype)
        # The fields of each emitter: of the whole grid for a layer, of the
        # planes of a slab for a volume (flat, so that the fields of the
        # last slab, which can have fewer planes, are its first values)
        self._layouts = {
            key: self._plan(key).fields_layout() for key in self._keys}
        self._fields = {
            key: driver.mem_alloc_d(math.prod(layout.shape(
                native_grid if layout.layer else slab_grid)), dtype)
            for key, layout in self._layouts.items()}
        self._ratios = {}
        for m, i in self._keys:
            ratios = self._plans[m].ratios[i]
            self._ratios[m, i] = None if ratios is None \
                else driver.mem_copy_h2d(ratios)
        # The cubes of opacity of a slab, with absorbers (flat, as the
        # fields)
        self._odata = None
        if self.has_dust():
            self._odata = driver.mem_alloc_d(
                self._ncubes * math.prod(slab_grid['spat_size']), dtype)
        # The emitters of the march of slabs of each size: with the line
        # opacity of their fields, without dust, and with dust if the
        # observation has dust (see evaluate); and without opacity, for
        # the light that no opacity dims (an extra output)
        kinds = [(False, True), (False, False)] \
            + ([(True, True)] if self._ncubes else [])
        self._emitters = {
            (slab_size, dust, line_opacity):
                self._native_emitters(slab_size, dust, line_opacity)
            for slab_size in {slab_size for _, slab_size in self._slabs}
            for dust, line_opacity in kinds}
        self._gmodel = driver.native_class('GModel', dtype)()

    def has_dust(self) -> bool:
        """Return whether it has components with dust."""
        return any(plans.absorbers for plans in self._plans)

    def has_line_opacity(self) -> bool:
        """Return whether it has emitters with line opacity."""
        return any(layout.line_opacity for layout in self._layouts.values())

    def evaluate(
            self,
            params: Sequence[ParamValues],
            gravity: Sequence[dict[int, ParamValues]],
            outputs: dict[str, DeviceArray | None],
            out_extra: Sequence[dict[str, Any]] | None,
            march_extra: dict[str, Any] | None,
            opacity_front: DeviceArray | None,
            tau: DeviceArray | None,
            positions: Sequence[float] | None = None
    ) -> LineResolution | None:
        """
        Add the light of the emitters that reaches the viewer to the
        output array, the 'image' or the 'scube' in outputs, with the
        parameters of each galaxy (params) and those that its gravity gives
        its components (gravity, see GravityPlan.evaluate), at the given
        positions along the line of sight (arcsec, away from the viewer,
        from any origin; all at one if None), dimmed by the opacity in
        front of the galaxies and their own, and add their own to that in
        front: the optical depth of the dust in front at each spaxel, in
        each cube of opacity (opacity_front, of shape (ncubes, ny, nx);
        required with dust of their own, None without dust in front or in
        them), and that of the lines at each sub-channel of each spaxel
        (tau, of shape (nchannels * subchannels, ny, nx); required with
        line opacity, None without line opacity in front or in them).

        The extra outputs of the components of each galaxy go to its item
        of out_extra, named after their names, or their positions (e.g.
        'disk_bdata' or 'component0_bdata'), with the totals of each
        galaxy, 'total_*' (see ComponentPlans.add_extras; along the depth
        axis in the frame of the galaxy, centred on its plane). Those of
        the march go to march_extra (given with out_extra), of the shape
        of the output: 'total_unabsorbed', the light of the emitters
        without opacity (neither their own nor that in front), and with
        line opacity, 'total_line_tau', the optical depth of their lines
        at each channel (the mean of its sub-channels).

        With extras, a warning says where zmax shapes a disk (see
        ComponentPlans.warn_truncation), and it returns how finely it
        resolves the lines (see LineResolution); None otherwise.
        """
        driver = self._driver
        grids = self._grids(positions)
        values = [
            plans.params(galaxy_params, galaxy_gravity)
            for plans, galaxy_params, galaxy_gravity
            in zip(self._plans, params, gravity)]
        extras = None
        if out_extra is not None:
            extras = [{} for _ in self._plans]
            for plans, galaxy_values in zip(self._plans, values):
                plans.warn_truncation(galaxy_values)
        # (those of the volumes and the absorbers of each slab, by galaxy)
        slab_extras = []

        # The lines of the emitters, and the fields of the layers
        lines_h, lines_d = self._lines
        for key in self._keys:
            m, i = key
            lines_h[self._rows[key]] = self._plan(key).line_values(
                values[m][i])
            if self._layouts[key].layer:
                self._evaluate_fields(values, key, grids[m], extras)
        driver.mem_copy_h2d(lines_h, lines_d)

        image = outputs.get('image') is not None
        out = outputs['image'] if image else outputs['scube']
        dust = opacity_front is not None
        subchannels = self._subchannels if tau is not None else 1
        # With extras, the light without opacity, and the optical depth of
        # the lines in front of the galaxies
        unabsorbed = None
        tau_front = None
        resolution = None
        grid = grids[0]
        if out_extra is not None:
            unabsorbed = driver.mem_alloc_d(out.shape, self._dtype)
            driver.mem_fill(unabsorbed, 0)
            if self.has_line_opacity():
                tau_front = driver.mem_copy_d2h(tau)
                resolution = _ResolutionTracker(
                    grid['spec_step'] / subchannels,
                    grid['spat_size'][1::-1])
        for z0, size in self._slabs:
            slab_grids = [_slab_grid(g, z0, size) for g in grids]
            extra = None if extras is None else [{} for _ in self._plans]
            odata = None
            if self._odata is not None:
                odata = _view(
                    self._odata,
                    (self._ncubes,) + slab_grids[0]['spat_size'][::-1])
                driver.mem_fill(odata, 0)
                for m, plans in enumerate(self._plans):
                    plans.evaluate_opacity(
                        values[m], slab_grids[m], odata,
                        None if extra is None else extra[m])
            for key in self._keys:
                m = key[0]
                if not self._layouts[key].layer:
                    self._evaluate_fields(values, key, slab_grids[m], extra)
                if resolution is not None and self._layouts[key].line_opacity:
                    resolution.add(key, driver.mem_copy_d2h(_view(
                        self._fields[key],
                        self._layouts[key].shape(slab_grids[m]))),
                        lines_h[self._rows[key], 1])
            self._march(
                slab_grids[0], image, self._emitters[size, dust, True],
                subchannels, odata, opacity_front, tau, out)
            if unabsorbed is not None:
                self._march(
                    slab_grids[0], image, self._emitters[size, False, False],
                    1, None, None, None, unabsorbed)
            if extra is not None:
                slab_extras.append(extra)

        if out_extra is None:
            return None
        light = 0.0
        for m, plans in enumerate(self._plans):
            # (the extras of the slabs, along the depth axis)
            for i in slab_extras[0][m]:
                extras[m].setdefault(i, {}).update({
                    key: np.concatenate(
                        [extra[m][i][key] for extra in slab_extras])
                    for key in slab_extras[0][m][i]})
            bdata = [
                extras[m][i]['bdata'] for i in plans.emitters
                if 'bdata' in extras[m].get(i, {})]
            plans.add_extras(
                out_extra[m], extras[m],
                dict(bdata=sum(bdata)) if bdata else {},
                grids[m]['spat_zero'][2])
            if self._smoothing is not None:
                light = max(light, plans.light_resolution(
                    values[m], extras[m], self._smoothing))
        plans = self._plans[0]
        march_extra['total_unabsorbed'] = plans.output_extra(
            driver.mem_copy_d2h(unabsorbed))
        spectral = depth = 0.0
        if tau_front is not None:
            line_tau = driver.mem_copy_d2h(tau) - tau_front
            march_extra['total_line_tau'] = plans.output_extra(
                line_tau.reshape(
                    (-1, subchannels) + line_tau.shape[1:]).mean(axis=1))
            spectral, depth = resolution.resolution(
                line_tau.max(axis=0) >= _THIN_TAU)
        return LineResolution(spectral, depth, light)

    def _grids(self, positions: Sequence[float] | None) -> list[NativeGrid]:
        """
        Return the native grid of each galaxy at the given positions (see
        evaluate): the grid of its depth axis, in the frame of the galaxy
        (that is, centred on its plane; see _placement).
        """
        grid = self._plans[0].native_grid
        if len(self._members) == 1:
            return [grid]
        if positions is None:
            positions = [0.0] * len(self._members)
        zero, placed = _placement(
            positions, [member.depth for member in self._members],
            self._depth.size)
        x0, y0, _ = grid['spat_zero']
        return [
            grid | dict(spat_zero=(x0, y0, zero - position))
            for position in placed]

    def _march(
            self,
            slab_grid: NativeGrid,
            image: bool,
            emitters: list[Any],
            subchannels: int,
            opacity: DeviceArray | None,
            opacity_front: DeviceArray | None,
            tau: DeviceArray | None,
            out: DeviceArray
    ) -> None:
        """
        March through the planes of a slab, adding the light of the given
        native emitters to the image or the spectral cube out (see the
        native GModel.transfer_march).
        """
        grid = self._plans[0].native_grid
        self._gmodel.transfer_march(
            emitters=emitters, spat_size=slab_grid['spat_size'],
            slab_zero=slab_grid['slab_zero'],
            depth_size=grid['spat_size'][2], image=image,
            spec_size=grid['spec_size'], spec_step=grid['spec_step'],
            spec_zero=grid['spec_zero'], subchannels=subchannels,
            opacity=opacity, opacity_front=opacity_front, tau=tau, out=out)

    def _evaluate_fields(
            self,
            values: list[dict[int, ParamValues]],
            key: tuple[int, int],
            grid: NativeGrid,
            extras: list[dict[int, dict[str, Any]]] | None
    ) -> None:
        """
        Write the fields of the emitter of the given key (the index of its
        galaxy and its own) on the grid (the whole grid of a layer, or a
        slab of a volume); with extras, its extra outputs go to the item
        of its galaxy, by its index.
        """
        m, i = key
        extra = {} if extras is not None else None
        self._plan(key).evaluate_fields(
            values[m][i], grid,
            _view(self._fields[key], self._layouts[key].shape(grid)), extra)
        if extras is not None:
            extras[m].setdefault(i, {}).update(extra)

    def _native_emitters(
            self, size: int, dust: bool, line_opacity: bool
    ) -> list[Any]:
        """
        Return the native emitters of the march of a slab of the given
        number of planes: their fields, lines and, with dust, ratios; with
        line_opacity, those with line opacity have it.
        """
        native_grid = self._plans[0].native_grid
        slab_grid = _slab_grid(native_grid, 0, size)
        emitter_class = self._driver.native_class(
            'TransferEmitter', self._dtype)
        emitters = []
        for key in self._keys:
            layout = self._layouts[key]
            emitters.append(emitter_class(
                layer=layout.layer,
                line_opacity=line_opacity and layout.line_opacity,
                fields=_view(self._fields[key], layout.shape(
                    native_grid if layout.layer else slab_grid)),
                lines=self._lines[1][self._rows[key]],
                opacity_ratios=self._ratios[key] if dust else None))
        return emitters

    def _plan(self, key: tuple[int, int]) -> 'ComponentPlan':
        """Return the plan of the component of the given key."""
        m, i = key
        return self._plans[m].plans[i]

    def _component(self, key: tuple[int, int]) -> 'Component':
        """Return the component of the given key."""
        m, i = key
        return self._members[m].component_set.components()[i]


def _placement(
        positions: Sequence[float], axes: Sequence[DepthAxis], size: int
) -> tuple[float, list[float]]:
    """
    Return where galaxies at the given positions along the line of sight
    (arcsec), each held by its own depth axis (centred on its plane), go
    on one depth axis of the given number of voxels (of their step): the
    position of its first voxel, and that of each galaxy. The stretches
    between the axes of galaxies that do not overlap are left out, in
    whole voxels, and the voxels have the depths that they have on the
    axis of the first galaxy (so they sample each galaxy as they do on its
    own axis, up to the separations of the galaxies).
    """
    step = axes[0].step
    placed = [float(position) for position in positions]
    # (the depths of the first and the last voxel of each axis, in the
    # frame of its galaxy)
    lows = [axis.zero for axis in axes]
    highs = [axis.zero + (axis.size - 1) * step for axis in axes]
    # Front to back, each galaxy moves forward by the stretches left out
    # in front of it
    cut = 0.0
    end = -math.inf
    for g in sorted(range(len(placed)), key=placed.__getitem__):
        placed[g] -= cut
        gap = placed[g] + lows[g] - (end + step)
        if math.isfinite(gap) and gap >= step:
            stretch = math.floor(gap / step) * step
            placed[g] -= stretch
            cut += stretch
        end = max(end, placed[g] + highs[g])
    start = min(p + low for p, low in zip(placed, lows))
    first = placed[0] + lows[0]
    zero = first - math.ceil((first - start) / step - 1e-9) * step
    if zero + (size - 1) * step < end - 1e-9 * step:
        raise RuntimeError(
            f"the galaxies take {(end - zero) / step + 1:g} voxels of a "
            f"depth axis of {size}")
    return zero, placed


class _ResolutionTracker:
    """
    The resolution of the line opacity of a march at each spaxel (see
    LineResolution), of the voxels of the volumes with line opacity, slab
    by slab, for a width of the channels of the march and the shape (ny,
    nx) of the spaxels.
    """

    def __init__(self, channel: float, shape: tuple[int, int]):
        self._channel = channel
        self._spectral = np.zeros(shape)
        self._depth = np.zeros(shape)
        # (the last plane of the previous slab of each volume)
        self._last = {}

    def add(self, index: int, fields: np.ndarray, scales: np.ndarray) -> None:
        """
        Add the voxels of a slab of the volume of the given index: its
        fields (see FieldsLayout; with line opacity) and the scale of each
        of its lines (see ComponentPlan.line_values).
        """
        j, v, s = fields[:3]
        on = (j != 0) & (s > 0) & (fields[3:3 + len(scales)] != 0).any(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            # The narrowest line of each voxel, in the spectral units
            sigma = np.abs(scales).min() * s
            spectral = np.where(on, self._channel / sigma, 0)
            if index in self._last:
                v, s, on = (
                    np.concatenate([last[None], now])
                    for last, now in zip(self._last[index], (v, s, on)))
            pair = on[1:] & on[:-1]
            depth = np.where(
                pair,
                np.abs(v[1:] - v[:-1]) / np.minimum(s[1:], s[:-1]), 0)
        self._spectral = np.maximum(self._spectral, spectral.max(axis=0))
        self._depth = np.maximum(
            self._depth, depth.max(axis=0, initial=0))
        self._last[index] = (v[-1], s[-1], on[-1])

    def resolution(self, where: np.ndarray) -> tuple[float, float]:
        """
        Return the worst spectral and depth ratios of the march at the
        given spaxels (a mask of shape (ny, nx); see LineResolution).
        """
        def worst(ratio):
            return float(np.max(ratio, where=where, initial=0))
        return worst(self._spectral), worst(self._depth)


def _slab_grid(grid: NativeGrid, z0: int, size: int) -> NativeGrid:
    """
    Return the grid of a slab of the given number of planes of a grid,
    from its plane z0 (see NativeGrid: every slab places a point, e.g. a
    cloud of a Monte Carlo disk, in the same plane).
    """
    return grid | dict(
        spat_size=grid['spat_size'][:2] + (size,), slab_zero=z0)


def _view(data: DeviceArray, shape: tuple[int, ...]) -> DeviceArray:
    """Return the first values of a flat array, as an array of a shape."""
    return data[:math.prod(shape)].reshape(shape)
