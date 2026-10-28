import os.path
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import astropy.units as u
import numpy as np
from annotated_types import Gt
from astropy.table import Table

from gbkfit.utils import parseutils, tableutils
from gbkfit.utils.parseutils import ConfigError, check_options
from gbkfit.utils.typeutils import Column, FilePath, Unit
from . import _detail, _resampling
from .base import Dataset
from .data import Data, DataColumns


__all__ = [
    'DatasetProfile',
    'check_profile_radii'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('the radii',)


def check_profile_radii(
        radii: u.Quantity, distance: u.Quantity | None
) -> tuple[u.Quantity, u.Quantity | None]:
    """
    Return the radii of a profile (see DatasetProfile) and their distance,
    as 1d arrays of floats and a length; or raise ConfigError if the radii
    are not angles or lengths, finite and at least 0, or the distance is
    not a positive length, or is given for radii in angles.
    """
    if not isinstance(radii, u.Quantity) or radii.ndim != 1 \
            or not (radii.unit.is_equivalent(u.arcsec)
                    or radii.unit.is_equivalent(u.kpc)):
        raise ConfigError(
            f"the radii are a vector of angles (e.g. arcsec) or lengths "
            f"(e.g. kpc), with their unit; they are {radii!r}")
    radii = radii.astype(float)
    if not np.all(np.isfinite(radii) & (radii >= 0)):
        raise ConfigError(
            f"the radii must be finite and at least 0; they are {radii}")
    if distance is not None:
        if radii.unit.is_equivalent(u.arcsec):
            raise ConfigError(
                "a distance is that of radii in lengths, measured at it; "
                "the radii are angles")
        if not isinstance(distance, u.Quantity) or distance.size != 1 \
                or not distance.unit.is_equivalent(u.Mpc) \
                or not distance > 0:
            raise ConfigError(
                f"the distance must be a positive length; it is "
                f"{distance!r}")
    return radii, distance


class DatasetProfile(Dataset):
    """
    Values of functions of a galaxy at radii (see ObservationProfile; e.g.
    its rotation curve): a table of radii, and of the values of each
    function, its data items, named after the function (e.g. vcirc).

    The radii are angles (e.g. arcsec), or lengths (e.g. kpc): measured at
    a distance, if given (their angles are the lengths over it), or
    physical (e.g. galactocentric radii in the Milky Way).

    Configurations give the file of its table (see from_file).

    Parameters
    ----------
    items : Mapping of str to Data
        The values of each function, a vector of one value per radius; at
        least one.
    radii : Quantity
        The radii.
    distance : float, optional
        The distance of radii in lengths that were measured at one (Mpc).

    Raises
    ------
    ConfigError
        If the items do not have one value per radius, or the radii or the
        distance are invalid (see check_profile_radii).
    """

    ndim = 1

    config_constructors = ('from_file',)

    @staticmethod
    def type() -> str:
        return 'profile'

    @classmethod
    def from_file(
            cls,
            file: FilePath,
            radius: Column,
            format: str | None = None,
            distance: Annotated[float, Gt(0), Unit('Mpc')] | None = None,
            **items: DataColumns
    ) -> 'DatasetProfile':
        """
        Read a profile from a table file (any that astropy reads, see
        Table.read: e.g. ECSV, FITS, CSV or text; the columns of a text
        file without a header are named col1, col2, ...).

        Parameters
        ----------
        file : str
            The file.
        radius : Column
            The column of the radii (by default in the unit of the
            column, which must have one).
        format : str, optional
            The format of the file, if astropy cannot tell it.
        distance : float, optional
            The distance of radii in lengths that were measured at one
            (Mpc).
        **items : DataColumns
            The columns of each data item, by name.

        Returns
        -------
        DatasetProfile
            The dataset.
        """
        check_options(DatasetProfile.from_file, locals())
        table = tableutils.read_table(file, format)
        with parseutils.config_path('radius'):
            radii = tableutils.read_column(table, radius.column, radius.unit)
            if radii.unit is u.dimensionless_unscaled:
                raise ConfigError(
                    "the radii need their unit (e.g. unit: kpc), which their "
                    "column does not have")
        data = {}
        for key, item in items.items():
            with parseutils.config_path(key):
                data[key] = _detail.table_item(table, item)
        return cls(data, radii, distance)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the dataset to its configuration, writing its table to an
        ECSV file ('{prefix}profile.ecsv'): the radii ('radius'), and of
        each data item, its values ('{item}'), errors ('{item}_error') and
        mask ('{item}_mask').
        """
        filename = f'{prefix}profile.ecsv'
        table = Table(dict(radius=self._radii))
        info = dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            radius=dict(column='radius'))
        if self._distance is not None:
            info['distance'] = float(self._distance.to_value(u.Mpc))
        for key, item in self.items():
            unit = item.unit() or u.dimensionless_unscaled
            table[key] = item.data() * unit
            info[key] = dict(column=key, mask=f'{key}_mask')
            if item.error() is not None:
                table[f'{key}_error'] = item.error() * unit
                info[key]['error'] = f'{key}_error'
            table[f'{key}_mask'] = item.mask()
        table.write(filename, format='ascii.ecsv', overwrite=overwrite)
        return info

    def __init__(
            self,
            items: Mapping[str, Data],
            radii: u.Quantity,
            distance: Annotated[float, Gt(0), Unit('Mpc')] | None = None
    ):
        check_options(DatasetProfile.__init__, locals())
        super().__init__(items)
        radii, distance = check_profile_radii(
            radii, None if distance is None else distance * u.Mpc)
        if self.shape() != radii.shape:
            raise ConfigError(
                f"the data items have {self.shape()[0]} values, but there "
                f"are {radii.size} radii")
        self._radii = radii
        self._distance = distance

    def radii(self) -> u.Quantity:
        """Return the radii."""
        return self._radii

    def distance(self) -> u.Quantity | None:
        """
        Return the distance of radii in lengths that were measured at one,
        if given.
        """
        return self._distance

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of radii: a positive
        integer for each of the radii (see resample_blocks).
        """
        _resampling.block_size(size, _BLOCK_AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its data redrawn in blocks of radii (see
        Dataset.resample_blocks), of the size (radii): runs of consecutive
        radii, in their order (e.g. of 3, for the points of a rotation
        curve closer than the beam that measured them).
        """
        return _resampling.resample_blocks(
            residuals, size, _BLOCK_AXES, rng)

