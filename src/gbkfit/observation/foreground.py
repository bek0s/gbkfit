import copy
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Self

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamScalarDesc, ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Unit

from .lens_masses import LensMass, lens_mass_parser


__all__ = [
    'Foreground',
    'Lens',
    'LensPlan',
    'foreground_parser',
    'lens_parser'
]


class Lens(parseutils.Serializable):
    """
    A gravitational lens in front of a model: masses between it and the
    telescope (see LensMass), whose deflections add up.

    The light of each point of the image plane comes from the point of
    the source plane, where the model is, at its position less the
    deflection there (surface brightness is conserved). The model is
    evaluated on a grid of the source plane, centred on the origin of the
    frame of the model and aligned with the sky (see
    gridutils.sky_positions); each pixel of the image takes the bilinear
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
    ConfigError
        If it has no masses, or only some of its masses have names (see
        pdescs).
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Lens':
        parseutils.load_option_and_update_info(
            lens_mass_parser, info, 'masses', required=True)
        return super().load(info)

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
            raise ConfigError("a lens needs at least one mass")
        self._source_grid = gridutils.make_grid(
            tuple(source_size), tuple(source_step))
        self._masses = masses
        self._prefixes = parseutils.item_prefixes(
            [mass.name() for mass in masses], 'masses', 'mass', False)
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [mass.pdescs() for mass in masses], self._prefixes)

    def source_grid(self) -> gridutils.Grid:
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
        Return the names of the parameters of the centres (x and y) of its
        masses that are placed by them (see LensMass.centred).
        """
        return tuple(
            (names['x'], names['y'])
            for mass, names in zip(self._masses, self._mappings)
            if mass.centred())

    def deflection(
            self, grid: gridutils.Grid, params: ParamValues
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
            grid: gridutils.Grid,
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
    or for each evaluation if the lens has parameters.
    """

    def __init__(
            self,
            lens: Lens,
            driver: Driver,
            grid: gridutils.Grid,
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
        self._sky = gridutils.sky_positions(grid)
        self._matrix, self._offset = gridutils.sky_to_pixel(
            lens.source_grid())
        self._source_x = driver.mem_alloc_d(self._sky[0].shape, dtype)
        self._source_y = driver.mem_alloc_d(self._sky[0].shape, dtype)
        if not self._names:
            self._place({})
        self._backend = driver.native_class('DModel', dtype)()

    def _place(self, params: ParamValues) -> None:
        """
        Work out the position on the source plane of each pixel of the
        image, with the parameters of the lens, and copy it to the device.
        """
        x, y = self._sky
        deflection_x, deflection_y = self._lens.deflection(self._grid, params)
        source_x = x - deflection_x
        source_y = y - deflection_y
        matrix, offset = self._matrix, self._offset
        pixel_x = matrix[0, 0] * source_x + matrix[0, 1] * source_y + offset[0]
        pixel_y = matrix[1, 0] * source_x + matrix[1, 1] * source_y + offset[1]
        self._driver.mem_copy_h2d(pixel_x.astype(self._dtype), self._source_x)
        self._driver.mem_copy_h2d(pixel_y.astype(self._dtype), self._source_y)

    def source_grid(self, grid: gridutils.Grid) -> gridutils.Grid:
        """
        Return the grid of the source plane for a model evaluated on the
        given grid of the image plane: the x and y of the source plane, and
        the other axes of the grid (e.g. the spectral axis).
        """
        source = self._lens.source_grid()
        return gridutils.Grid(
            source.size + grid.size[2:],
            gridutils.Coords(
                source.coords.step + grid.coords.step[2:],
                source.coords.rpix + grid.coords.rpix[2:],
                source.coords.rval + grid.coords.rval[2:],
                source.coords.rota, grid.coords.rest, grid.coords.medium,
                grid.coords.convention),
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
        """
        if self._names:
            self._place({
                name: params[full] for name, full in self._names.items()})
        self._backend.lens_resample(
            self._source_x, self._source_y, source, image)




lens_parser = parseutils.BasicParser(Lens)


class Foreground(parseutils.Serializable):
    """
    What happens to the light of a model before it reaches the telescope:
    a gravitational lens. Each has its own slot, in the order the light
    meets them.

    Its parameters (those of its lens) are those of the observation it is
    in front of, named after it (see with_prefix and pdescs).

    Parameters
    ----------
    lens : Lens, optional
        The gravitational lens, if any.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Foreground':
        parseutils.load_option_and_update_info(lens_parser, info, 'lens')
        return super().load(info)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(lens=lens_parser.dump(
            self._lens, prefix=prefix, dump_path=dump_path,
            overwrite=overwrite))

    def __init__(self, lens: Lens | None = None):
        check_options(Foreground.__init__, locals())
        self._lens = lens
        self._prefix = ''

    def lens(self) -> Lens | None:
        """Return the gravitational lens, if any."""
        return self._lens

    def with_prefix(self, prefix: str) -> Self:
        """
        Return a copy of it whose parameters have names that start with a
        prefix (e.g. 'muse_', for those of the lens 'muse_lens_theta_e').
        """
        foreground = copy.copy(self)
        foreground._prefix = prefix
        return foreground

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """
        Return its parameters, by name: '{prefix}lens_{name}' for each of
        those of its lens (see with_prefix).
        """
        if self._lens is None:
            return {}
        return {
            f'{self._prefix}lens_{name}': pdesc
            for name, pdesc in self._lens.pdescs().items()}

    def centre_params(self) -> tuple[tuple[str, str], ...]:
        """
        Return the names of the parameters of the centres (x and y) of the
        masses of its lens (see Lens.centre_params), as pdescs names them.
        """
        if self._lens is None:
            return ()
        return tuple(
            (f'{self._prefix}lens_{x}', f'{self._prefix}lens_{y}')
            for x, y in self._lens.centre_params())

    def plan_lens(
            self, driver: Driver, grid: gridutils.Grid, dtype: np.dtype
    ) -> LensPlan | None:
        """
        Plan the lensing of cubes on a grid of the image plane (see
        Lens.plan), if it has a lens.
        """
        if self._lens is None:
            return None
        return self._lens.plan(driver, grid, dtype, {
            name: f'{self._prefix}lens_{name}'
            for name in self._lens.pdescs()})


foreground_parser = parseutils.BasicParser(Foreground)
