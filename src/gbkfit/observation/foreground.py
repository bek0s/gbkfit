"""
What happens to the light of a model before it reaches the telescope
(see Foreground): a gravitational lens (see gbkfit.observation.lensing).
"""

import copy
from typing import Any, Self

import numpy as np

from gbkfit.config import BasicParser, Serializable, check_options
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamScalarDesc

from .lensing import Lens, LensPlan, lens_parser


__all__ = [
    'Foreground',
    'foreground_parser'
]


class Foreground(Serializable):
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
            self._full_name(name): pdesc
            for name, pdesc in self._lens.pdescs().items()}

    def centre_params(self) -> tuple[tuple[str, str], ...]:
        """
        Return the names of the parameters of the centres (x and y) of the
        masses of its lens (see Lens.centre_params), as pdescs names them.
        """
        if self._lens is None:
            return ()
        return self._named(self._lens.centre_params())

    def shift_params(self) -> tuple[tuple[str, str], ...]:
        """
        Return the names of the parameters of the shifts (x and y) of the
        masses of its lens placed by their RA and Dec (see
        Lens.shift_params), as pdescs names them.
        """
        if self._lens is None:
            return ()
        return self._named(self._lens.shift_params())

    def plan_lens(
            self, driver: Driver, grid: Grid, dtype: np.dtype
    ) -> LensPlan | None:
        """
        Plan the lensing of cubes on a grid of the image plane (see
        Lens.plan), if it has a lens.

        Parameters
        ----------
        driver : Driver
            The driver the cubes are on.
        grid : Grid
            The grid of the image plane.
        dtype : np.dtype
            The floating type of the cubes.

        Returns
        -------
        LensPlan or None
            The plan, or None without a lens.
        """
        if self._lens is None:
            return None
        return self._lens.plan(driver, grid, dtype, {
            name: self._full_name(name) for name in self._lens.pdescs()})

    def _named(
            self, pairs: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        """
        Return pairs of names of parameters of its lens as pdescs names
        them.
        """
        return tuple(
            (self._full_name(x), self._full_name(y)) for x, y in pairs)

    def _full_name(self, name: str) -> str:
        """Return the name of a parameter of its lens as pdescs names it."""
        return f'{self._prefix}lens_{name}'


foreground_parser = BasicParser(Foreground)
