"""
A gravitational lens in front of a model (see Lens), and its evaluation
(see LensPlan).
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import numpy as np

from gbkfit.config import (
    BasicParser, Serializable, Unit, check_options, item_prefixes)
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Coords, Grid, make_grid, sky_positions, sky_to_pixel
from gbkfit.params import ParamScalarDesc, ParamValues, StageCache
from gbkfit.utils import iterutils

from .masses import LensMass, lens_mass_parser


__all__ = [
    'Lens',
    'LensPlan',
    'lens_parser'
]


class Lens(Serializable):
    """
    A gravitational lens in front of a model: masses between it and the
    telescope (see LensMass), whose deflections add up.

    The light of each point of the image plane comes from the point of
    the source plane, where the model is, at its position less the
    deflection there (surface brightness is conserved). The model is
    evaluated on a grid of the source plane, centred on the origin of the
    frame of the model and aligned with the sky (see
    sky_positions); each pixel of the image takes the bilinear
    interpolation of the source at its position on the source plane.

    Its parameters are those of its masses (see pdescs), of the
    observation it is in front of (see Foreground.pdescs): its deflection
    is then evaluated for each evaluation of the model.

    Parameters
    ----------
    source_size : Sequence of int
        The number of pixels of the grid of the source plane (nx, ny).
    source_step : Sequence of float
        The size of its pixels (arcsec).
    masses : LensMass or Sequence of LensMass
        Its masses.

    Raises
    ------
    ValueError
        If it has no masses, or only some of its masses have names (see
        pdescs).
    """

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        count = len(self._masses)
        masses = lens_mass_parser.dump(
            list(self._masses),
            prefix=[f'{prefix}lens_{item}' for item in self._prefixes],
            dump_path=[dump_path] * count, overwrite=[overwrite] * count)
        return dict(
            source_size=self._source_grid.size,
            source_step=self._source_grid.coords.step, masses=masses)

    def __init__(
            self,
            source_size: Sequence[int],
            source_step: Annotated[Sequence[float], Unit('arcsec')],
            masses: LensMass | Sequence[LensMass]
    ):
        check_options(Lens.__init__, locals())
        masses = iterutils.tuplify(masses)
        if not masses:
            raise ValueError("a lens needs at least one mass")
        self._source_grid = make_grid(
            tuple(source_size), tuple(source_step))
        self._masses = masses
        self._prefixes = item_prefixes(
            [mass.name() for mass in masses], 'masses', 'mass', False)
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [mass.pdescs() for mass in masses], self._prefixes)

    def source_grid(self) -> Grid:
        """Return the grid of the source plane (x and y)."""
        return self._source_grid

    def masses(self) -> tuple[LensMass, ...]:
        """Return its masses."""
        return self._masses

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """
        Return its parameters, by name: those of its masses, with the
        name of each mass and '_' before them when it has several (e.g.
        'shear_gamma'), or 'mass' and its index for masses without names
        ('mass1_gamma'; none for the first).
        """
        return dict(self._pdescs)

    def centre_params(self) -> tuple[tuple[str, str], ...]:
        """
        Return the names of the parameters of the centres (xpos and ypos)
        of its masses that are placed by them (see LensMass.centred).
        """
        return tuple(
            (names['xpos'], names['ypos'])
            for mass, names in zip(self._masses, self._mappings)
            if mass.centred())

    def deflection(
            self, grid: Grid, params: ParamValues
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the deflection at the pixels of a grid of the image plane:
        the sum of those of its masses.

        Parameters
        ----------
        grid : Grid
            The grid; its x and y axes.
        params : ParamValues
            The values of its parameters, by their names (see pdescs).

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y of the frame of the
            model, each of shape (ny, nx).
        """
        alpha_x = np.zeros(grid.size[1::-1])
        alpha_y = np.zeros(grid.size[1::-1])
        for mass, names in zip(self._masses, self._mappings):
            mass_x, mass_y = mass.deflection(grid, {
                name: params[full] for name, full in names.items()})
            alpha_x += mass_x
            alpha_y += mass_y
        return alpha_x, alpha_y

    def plan(
            self,
            driver: Driver,
            grid: Grid,
            dtype: np.dtype,
            names: Mapping[str, str] | None = None
    ) -> 'LensPlan':
        """
        Plan the lensing of cubes on a grid of the image plane.

        Parameters
        ----------
        driver : Driver
            The driver the cubes are on.
        grid : Grid
            The grid of the image plane.
        dtype : np.dtype
            The floating type of the cubes.
        names : Mapping of str to str, optional
            The names of its parameters among those of the evaluations, by
            their own names (see pdescs); required if it has parameters.

        Returns
        -------
        LensPlan
            The plan.
        """
        return LensPlan(self, driver, grid, dtype, names or {})


class LensPlan:
    """
    The lensing of the cubes of the source plane of a lens to a grid of
    the image plane, on a driver: the position on the source plane of
    each pixel of the image (in pixels of the source), worked out once,
    or if the lens has parameters, again for each evaluation at which
    their values change (see Lens.plan, which makes it).

    Parameters
    ----------
    lens : Lens
        The lens.
    driver : Driver
        The driver the cubes are on.
    grid : Grid
        The grid of the image plane.
    dtype : np.dtype
        The floating type of the cubes.
    names : Mapping of str to str
        The names of the parameters of the lens among those of the
        evaluations, by their own names (see Lens.pdescs).

    Raises
    ------
    ValueError
        If the names of parameters of the lens are missing.
    """

    def __init__(
            self,
            lens: Lens,
            driver: Driver,
            grid: Grid,
            dtype: np.dtype,
            names: Mapping[str, str]
    ):
        if missing := sorted(set(lens.pdescs()) - set(names)):
            raise ValueError(
                f"the names of these parameters of the lens are missing: "
                f"{missing}")
        self._lens = lens
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        self._names = dict(names)
        self._sky = sky_positions(grid)
        self._matrix, self._offset = sky_to_pixel(
            lens.source_grid())
        self._source_x = driver.mem_alloc_d(self._sky[0].shape, dtype)
        self._source_y = driver.mem_alloc_d(self._sky[0].shape, dtype)
        # (the deflection is costly, e.g. seconds for an elliptical NFW halo
        # on 400x400 pixels: it is worked out again only when the values of
        # the parameters of the lens change, e.g. not in a fit of the
        # galaxies alone; without parameters, here)
        self._placement = StageCache(self._names.values())
        if not self._names:
            self._placement.evaluate({}, self._place)
        self._backend = driver.native_class('DModel', dtype)()

    def _place(self, params: ParamValues) -> None:
        """
        Work out the position on the source plane of each pixel of the
        image, with the values of the parameters of the lens (by the names
        of the plan), and copy it to the device.
        """
        x, y = self._sky
        deflection_x, deflection_y = self._lens.deflection(self._grid, {
            name: params[full] for name, full in self._names.items()})
        source_x = x - deflection_x
        source_y = y - deflection_y
        matrix, offset = self._matrix, self._offset
        pixel_x = matrix[0, 0] * source_x + matrix[0, 1] * source_y + offset[0]
        pixel_y = matrix[1, 0] * source_x + matrix[1, 1] * source_y + offset[1]
        self._driver.mem_copy_h2d(pixel_x.astype(self._dtype), self._source_x)
        self._driver.mem_copy_h2d(pixel_y.astype(self._dtype), self._source_y)

    def source_grid(self, grid: Grid) -> Grid:
        """
        Return the grid of the source plane for a model evaluated on the
        given grid of the image plane: the x and y of the source plane,
        whose centre, the origin of the frame of the model, is at the RA
        and Dec of the reference pixel of the grid (see sky_positions),
        and the other axes of the grid (e.g. the spectral axis).
        """
        source = self._lens.source_grid()
        return Grid(
            source.size + grid.size[2:],
            Coords(
                source.coords.step + grid.coords.step[2:],
                source.coords.rpix + grid.coords.rpix[2:],
                grid.coords.rval, source.coords.rota, grid.coords.rest,
                grid.coords.medium, grid.coords.convention),
            grid.spectral_axis)

    def evaluate(
            self,
            params: ParamValues,
            source: DeviceArray,
            image: DeviceArray
    ) -> None:
        """
        Lens the source cube (nz, sy, sx) into the image cube, with the
        values of the parameters of the evaluation (those of the lens, if
        any, by the names of the plan).

        Parameters
        ----------
        params : ParamValues
            The values of the parameters of the evaluation.
        source : DeviceArray
            The cube of the source plane (see source_grid).
        image : DeviceArray
            The cube of the image plane, which it writes.
        """
        self._placement.evaluate(params, self._place)
        self._backend.lens_resample(
            self._source_x, self._source_y, source, image)


lens_parser = BasicParser(Lens)
