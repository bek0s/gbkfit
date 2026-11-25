"""
Data items: measured values, with their mask and error.
"""

import copy
import math
import os.path
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Annotated, Any, NamedTuple, TypeAlias

import astropy.units
import numpy as np
from annotated_types import Gt

import gbkfit.units
from gbkfit.config import DataFile, parse_options, split_data_file, warn
from gbkfit.grid import Coords, Rest
from gbkfit.instrument import LSF, PSF, lsf_parser, psf_parser
from gbkfit.io import (
    Beam, ErrorKind, MaskKind, OpenedArrays, coords_from_header, open_arrays)
from .deferred import Deferred


__all__ = [
    'Data',
    'DataColumns',
    'DataFiles',
    'Error',
    'dump_data'
]


# One error for all the values of a data item, in place of a file or a
# column of them: a standard deviation, above 0
Error: TypeAlias = Annotated[float, Gt(0)]


class DataFiles(NamedTuple):
    """
    The files of a data item, and their options (see Data.from_files); in
    a configuration, a dict of them.

    Parameters
    ----------
    data : str or tuple
        The file of the values: a filename, or a filename and the HDU.
    mask : str or tuple, optional
        The file of the mask.
    error : str or tuple or float, optional
        The file of the errors, or one error (a standard deviation) for
        all the values.
    unit : str, optional
        The unit of the values; by default, that of the header.
    beam : Mapping, optional
        The options of their beam (see Beam); by default, that of the
        header.
    spatial_correlation : PSF, optional
        The correlation of the noise on the sky.
    spectral_correlation : LSF, optional
        The correlation of the noise along the spectral axis.
    error_kind : str, optional
        What the file of the errors holds: 'sigma' (standard
        deviations), 'variance' or 'inverse_variance' (see
        read_arrays).
    mask_kind : str, optional
        What the file of the mask marks with values other than 0:
        'valid' (by default) or 'invalid' values (see read_arrays).
    """

    config_titles = dict(
        data='Data', mask='Mask', error='Errors', unit='Unit', beam='Beam',
        spatial_correlation='Spatial correlation',
        spectral_correlation='Spectral correlation', error_kind='Error kind',
        mask_kind='Mask kind')

    data: DataFile
    mask: DataFile | None = None
    error: DataFile | Error | None = None
    unit: str | None = None
    beam: Mapping[str, Any] | None = None
    spatial_correlation: PSF | None = None
    spectral_correlation: LSF | None = None
    error_kind: ErrorKind = 'sigma'
    mask_kind: MaskKind = 'valid'


class DataColumns(NamedTuple):
    """
    The columns of a data item in a table file; in a configuration, a dict
    of them.

    Parameters
    ----------
    column : str
        The column of the values.
    error : str or float, optional
        The column of the errors, or one error for all the values.
    mask : str, optional
        The column of the mask (0 for the values left out).
    unit : str, optional
        The unit of the values; by default, that of their column.
    """

    config_titles = dict(
        column='Column', error='Errors', mask='Mask', unit='Unit')

    column: str
    error: str | Error | None = None
    mask: str | None = None
    unit: str | None = None


def _as_float(x: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Return a copy of an array of floats, in native byte order (the copy
    leaves the caller's array unchanged): of the given dtype, or else
    floats keep their type and other types become float64.
    """
    x = np.asarray(x)
    if dtype is None:
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) \
            else np.dtype(np.float64)
    return np.array(x, dtype=np.dtype(dtype).newbyteorder('='))


class _Arrays(NamedTuple):
    """The arrays of a data item, as Data keeps them (see checked)."""
    data: np.ndarray
    mask: np.ndarray
    error: np.ndarray | None

    @classmethod
    def checked(
            cls,
            data: np.ndarray,
            mask: np.ndarray | None,
            error: np.ndarray | None
    ) -> '_Arrays':
        """
        Return the arrays of values, their mask and errors as Data keeps
        them (see Data), or raise ValueError if they are not valid.
        """
        data = _as_float(data)
        mask = np.ones_like(data) if mask is None \
            else _as_float(mask, data.dtype)
        if error is not None:
            error = _as_float(error, data.dtype)
        if np.any(~np.isfinite(mask)):
            raise ValueError("the mask must be finite")
        if data.shape != mask.shape:
            raise ValueError(
                f"the data and the mask must have one shape; their shapes "
                f"are {data.shape} and {mask.shape}")
        if error is not None and data.shape != error.shape:
            raise ValueError(
                f"the data and the error must have one shape; their shapes "
                f"are {data.shape} and {error.shape}")
        # The total mask: the values that are finite, not masked, and
        # with a finite, positive error
        total_mask = np.isfinite(data) & (mask != 0)
        if error is not None:
            # (negative errors are refused only where the values count: an
            # error map can mark the others with one, e.g. -1)
            negative = total_mask & (error < 0)
            if np.any(negative):
                raise ValueError(
                    f"errors cannot be negative; "
                    f"{np.count_nonzero(negative)} are, down to "
                    f"{error[negative].min():g}")
            total_mask &= np.isfinite(error) & (error > 0)
        data[~total_mask] = np.nan
        if error is not None:
            error[~total_mask] = np.nan
        return cls(data, total_mask.astype(data.dtype), error)


# (of Data.then: the unit of what a change makes, if it does not change it)
_SAME: Any = object()


class Data:
    """
    Measured values with their mask and error: arrays of one shape, as
    copies (float arrays keep their type, and the others become float64;
    the mask and the error take the type of the values), their unit and
    beam, if known, and the correlation of their noise, if any.

    Where the values were measured (e.g. on the pixels of a grid) is
    described by their dataset, as where the instrument looked (see
    DatasetGrid). The values that are masked (mask 0), not finite, or
    have an error that is not finite or is 0 (e.g. blank pixels of an
    error map) are NaN, with mask 0.

    The noise of values on the sky can be correlated (e.g. by the beam of
    a radio image, or by the resampling of the spaxels of an IFU cube),
    and that of values along a spectral axis too (e.g. by Hanning
    smoothing): the correlation of the noise of two values at the
    separation (dx, dy) on the sky and dz along the spectral axis is then
    C_xy(dx, dy) C_z(dz), the shapes of the PSF spatial_correlation and
    of the LSF spectral_correlation divided by their values at their
    centres (as the noise of mocks; see NoiseGauss). The widths of
    spectral_correlation are in km/s of the spectral axis, not of the
    light as those of the LSF of an instrument (see lsfs): the noise is
    correlated over the channels of the axis. Their likelihood
    takes it (see LikelihoodGauss); the kinds of data whose noise can
    be correlated say where (see Dataset.noise_correlation).

    A data item read from files (see from_files) knows its shape, unit,
    beam and correlations from their headers, and reads its values the
    first time they are used (data, mask, error or dtype), when the
    values themselves are checked; so does every item made of it before
    (see then). A copy for another process (a pickle) has its values,
    read first.

    Parameters
    ----------
    data : ndarray
        The values.
    mask : ndarray, optional
        The mask: 0 for the values to leave out; by default, none.
    error : ndarray, optional
        The error of each value (its standard deviation); any value
        (e.g. -1) where the values are masked or not finite.
    unit : str or astropy.units.UnitBase, optional
        The unit of the values and the errors (see
        gbkfit.units.parse_unit).
    beam : Beam or Mapping, optional
        Their beam, for units per beam (e.g. Jy/beam): a Beam, or its
        options bmaj, bmin and bpa.
    spatial_correlation : PSF, optional
        The correlation of their noise on the sky (e.g. the beam: see
        PSFGauss.from_beam); it must not vary along the spectral axis.
    spectral_correlation : LSF, optional
        The correlation of their noise along the spectral axis; it must
        not vary.

    Raises
    ------
    ValueError
        If the mask is not finite, an error of a value that is finite and
        not masked is negative, the arrays are not of one shape, the unit
        is not a unit, the widths of the beam are not positive, or a
        correlation varies.
    """

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None,
            unit: str | astropy.units.UnitBase | None = None,
            beam: Beam | Mapping[str, Any] | None = None,
            spatial_correlation: PSF | None = None,
            spectral_correlation: LSF | None = None
    ):
        arrays = _Arrays.checked(data, mask, error)
        self._set_facts(
            arrays.data.shape, error is not None, unit, beam,
            spatial_correlation, spectral_correlation)
        self._arrays: _Arrays | None = arrays
        self._read: Callable[[], Any] | None = None

    @classmethod
    def deferred(
            cls,
            read: Callable[[], 'Data | tuple'],
            shape: Sequence[int],
            has_error: bool,
            unit: str | astropy.units.UnitBase | None = None,
            beam: Beam | Mapping[str, Any] | None = None,
            spatial_correlation: PSF | None = None,
            spectral_correlation: LSF | None = None
    ) -> 'Data':
        """
        Return a data item of a shape whose values are read the first time
        they are used (see Data): read() returns them, as the arrays of
        values, mask and error that Data takes (each of the mask and the
        error can be None), or as a Data (e.g. of a change, see then).

        Parameters
        ----------
        read : Callable
            Reads the values.
        shape : Sequence of int
            The shape of the arrays (numpy order).
        has_error : bool
            Whether the values have errors (whether read() returns them).
        unit, beam, spatial_correlation, spectral_correlation : optional
            As Data takes them.

        Returns
        -------
        Data
            The data item.

        Raises
        ------
        ValueError
            If the unit is not a unit, the widths of the beam are not
            positive, or a correlation varies (the values are checked as
            they are read).
        """
        item = cls.__new__(cls)
        item._set_facts(
            tuple(shape), has_error, unit, beam, spatial_correlation,
            spectral_correlation)
        item._arrays, item._read = None, read
        return item

    def _set_facts(
            self,
            shape: tuple[int, ...],
            has_error: bool,
            unit: str | astropy.units.UnitBase | None,
            beam: Beam | Mapping[str, Any] | None,
            spatial_correlation: PSF | None,
            spectral_correlation: LSF | None
    ) -> None:
        """Set and check what it knows without its values (see Data)."""
        self._shape = tuple(shape)
        self._has_error = has_error
        self._unit = gbkfit.units.parse_unit(unit) if isinstance(unit, str) \
            else unit
        self._beam = _make_beam(beam)
        for kind, correlation in (
                ('spatial', spatial_correlation),
                ('spectral', spectral_correlation)):
            if correlation is not None and correlation.varies():
                raise ValueError(
                    f"the {kind} correlation of the noise must not vary")
        self._spatial_correlation = spatial_correlation
        self._spectral_correlation = spectral_correlation

    @classmethod
    def from_files(
            cls,
            data: DataFile,
            mask: DataFile | None = None,
            error: DataFile | Error | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rest: Rest = None,
            spectral_axis: int | None = None,
            medium: str | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            spatial_correlation: PSF | None = None,
            spectral_correlation: LSF | None = None,
            velocity: str | None = None,
            error_kind: ErrorKind = 'sigma',
            spectral_type: str | None = None,
            spectral_unit: str | None = None,
            mask_kind: MaskKind = 'valid',
            stated: Collection[str] = ()
    ) -> tuple['Data', Coords]:
        """
        Open a data item of files, with the unit and the beam of the
        header of the file of its values unless given (a unit of the
        header that is not a unit is an error), and the correlation of its
        noise, if any; its values are read when first used (see Data).

        The values, their mask and errors are read in the orientation of
        the model (see read_arrays), and the
        world coordinates of the values are those of the model (see
        coords_from_header).

        Parameters
        ----------
        data : str or tuple
            The file of the values: a filename (its first HDU with data),
            or a filename and the HDU.
        mask : str or tuple, optional
            The file of the mask.
        error : str or tuple or float, optional
            The file of the errors, or one error (a standard deviation)
            for all the values.
        rpix, rval : optional
            Passed to coords_from_header.
        spectral_axis : int, optional
            The index (FITS order, from 0) that the spectral axis (if any)
            must have (see coords_from_header): the data have an axis of
            their own before it for each, of length 1 too (see
            read_arrays).
        medium : str, optional
            The medium of the wavelengths of the spectral axis and of the
            rest, if the header does not state it (see read_arrays and
            coords_from_header).
        rest : str or Quantity, optional
            The rest of the spectral axis, which turns one of wavelengths
            or frequencies into velocities (see model_header).
        unit : str, optional
            The unit of the values (see Data).
        beam : Mapping, optional
            The options of their beam (see Beam); by default, that of the
            file of the values, if any (see read_arrays).
        spatial_correlation : PSF, optional
            The correlation of their noise on the sky (see Data).
        spectral_correlation : LSF, optional
            The correlation of their noise along the spectral axis.
        velocity : str, optional
            The convention of the velocities of a spectral axis that the
            headers leave ambiguous (see model_header).
        spectral_type : str, optional
            The type of a spectral axis whose headers give one that FITS
            does not define, or none (see model_header).
        spectral_unit : str, optional
            The unit of a spectral axis whose header gives none or one that
            is not valid (see SpectralAxis, option unit); it is not used for
            others.
        error_kind : str, optional
            What the file of the errors holds (see
            read_arrays).
        mask_kind : str, optional
            What the file of the mask marks with values other than 0 (see
            read_arrays).
        stated : Collection of str, optional
            The options of the grid of the dataset that it states (see
            stated_options), which the header of the values need not have
            (see read_arrays).

        Returns
        -------
        tuple of Data and Coords
            The data item, and the world coordinates of the file of its
            values.

        Raises
        ------
        ValueError
            If the error is not positive, the error is one error and
            error_kind is not 'sigma', or the headers of the files are not
            valid (see open_arrays); of the values, as they are read (see
            read_arrays and Data).
        """
        error_file = None
        if isinstance(error, bool):
            raise ValueError(
                "the error must be a number or a file; it is a bool")
        if isinstance(error, (int, float)):
            # (an error that is not positive would mask every value)
            if not error > 0:
                raise ValueError(f"the error must be positive; it is {error}")
            if error_kind != 'sigma':
                raise ValueError(
                    f"one error for all the values is a standard deviation; "
                    f"the kind of the errors (error_kind: {error_kind!r}) is "
                    f"that of a file of errors")
        elif error is not None:
            error_file = error
        opened = open_arrays(
            data, error_file, mask, error_kind=error_kind,
            mask_kind=mask_kind, unit=unit,
            rest=rest, velocity=velocity,
            spectral_type=spectral_type, spectral_unit=spectral_unit,
            medium=medium,
            spectral_axis=spectral_axis, stated=stated)
        coords = coords_from_header(
            split_data_file(data)[0], opened.header(), rpix=rpix, rval=rval,
            spectral_axis=spectral_axis, medium=medium, rest=rest)
        one_error = error if isinstance(error, (int, float)) else None
        return cls.deferred(
            _FileReader(opened, one_error), opened.shape(),
            error is not None, opened.unit,
            opened.beam if beam is None else beam, spatial_correlation,
            spectral_correlation), coords

    def is_read(self) -> bool:
        """Check whether its values have been read (see Data)."""
        return self._arrays is not None

    def ndim(self) -> int:
        """Return the number of axes of the arrays."""
        return len(self._shape)

    def size(self) -> int:
        """Return the number of values."""
        return math.prod(self._shape)

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the arrays (numpy order)."""
        return self._shape

    def data(self) -> np.ndarray:
        """Return the values (NaN where left out)."""
        return self._values().data

    def mask(self) -> np.ndarray:
        """Return the mask: 1 for the values used, 0 for the others."""
        return self._values().mask

    def error(self) -> np.ndarray | None:
        """Return the errors, if any (NaN where left out)."""
        return self._values().error

    def has_error(self) -> bool:
        """Check whether the values have errors (see error)."""
        return self._has_error

    def dtype(self) -> np.dtype:
        """Return the dtype of the arrays."""
        return self._values().data.dtype

    def unit(self) -> astropy.units.UnitBase | None:
        """Return the unit of the values, if known."""
        return self._unit

    def beam(self) -> Beam | None:
        """Return the beam of the values, if known."""
        return self._beam

    def spatial_correlation(self) -> PSF | None:
        """Return the correlation of the noise on the sky, if any."""
        return self._spatial_correlation

    def spectral_correlation(self) -> LSF | None:
        """
        Return the correlation of the noise along the spectral axis, if
        any.
        """
        return self._spectral_correlation

    def then(
            self,
            change: Callable[['Data'], 'Data'],
            *,
            unit: str | astropy.units.UnitBase | None = _SAME
    ) -> 'Data':
        """
        Return what a change makes of it: at once, if its values have been
        read; else a data item whose values are made as they are first
        used (see Data), of its values then. The change takes it with its
        values and returns another data item of its shape, beam,
        correlations and errors (if it has them), and of its unit unless
        another is given.

        Parameters
        ----------
        change : Callable
            The change.
        unit : str or UnitBase, optional
            The unit of what the change makes, if it changes it.

        Returns
        -------
        Data
            What the change makes.
        """
        if self.is_read():
            return change(self)
        return Data.deferred(
            _Changed(self, change), self._shape, self._has_error,
            self._unit if unit is _SAME else unit, self._beam,
            self._spatial_correlation, self._spectral_correlation)

    def to_unit(self, unit: astropy.units.UnitBase) -> 'Data':
        """
        Return a copy in another unit of the same kind (e.g. km/s of m/s):
        its values and errors converted, its mask, beam and correlations of
        the noise.

        Parameters
        ----------
        unit : UnitBase
            The other unit.

        Returns
        -------
        Data
            The copy.

        Raises
        ------
        ValueError
            If it has no unit, or its unit is not of the kind of the
            other.
        """
        if self._unit is None or not self._unit.is_equivalent(unit):
            raise ValueError(
                f"data in {self._unit} cannot be given in {unit}")
        factor = self._unit.to(unit)

        def converted(item: Data) -> Data:
            return Data(
                item.data() * factor, item.mask(),
                None if item.error() is None else item.error() * factor,
                unit, item.beam(), item.spatial_correlation(),
                item.spectral_correlation())
        return self.then(converted, unit=unit)

    def with_data(self, data: np.ndarray) -> 'Data':
        """
        Return a copy with other values (e.g. those of a bootstrap), of its
        dtype, and its mask, error, unit, beam and correlations of the
        noise: the values it masks stay masked.

        Parameters
        ----------
        data : np.ndarray
            The other values.

        Returns
        -------
        Data
            The copy.

        Raises
        ------
        ValueError
            If the values are not of its shape.
        """
        if np.shape(data) != self._shape:
            raise ValueError(
                f"the values must be of the shape of the data, "
                f"{self._shape}; they are of {np.shape(data)}")

        def replaced(item: Data) -> Data:
            return Data(
                np.asarray(data, item.dtype()), item.mask(), item.error(),
                item.unit(), item.beam(), item.spatial_correlation(),
                item.spectral_correlation())
        return self.then(replaced)

    def masked(self, where: 'np.ndarray | Deferred') -> 'Data':
        """
        Return a copy with its values masked where an array (of its shape,
        or one that broadcasts to it) is 0, and as it is elsewhere (e.g.
        outside where the instrument looked; see DatasetGrid).

        Parameters
        ----------
        where : np.ndarray or Deferred
            Where its values are kept (not 0); one read when first used,
            with its values.

        Returns
        -------
        Data
            The copy.
        """
        def kept(item: Data) -> Data:
            values = where.values() if isinstance(where, Deferred) else where
            keep = np.broadcast_to(values, item.shape()) != 0
            return Data(
                item.data(), item.mask() * keep, item.error(), item.unit(),
                item.beam(), item.spatial_correlation(),
                item.spectral_correlation())
        return self.then(kept)

    def _values(self) -> _Arrays:
        """Return its arrays, read now if not yet (see Data)."""
        if self._arrays is None:
            self._arrays, self._read = self._fresh(), None
        return self._arrays

    def _fresh(self) -> _Arrays:
        """
        Return its arrays as they are read now, without keeping them (see
        _values); or those kept, once read.
        """
        if self._arrays is not None:
            return self._arrays
        read = self._read()
        arrays = read._fresh() if isinstance(read, Data) \
            else _Arrays.checked(*read)
        if arrays.data.shape != self._shape:
            raise ValueError(
                f"the values read are of shape {arrays.data.shape}; their "
                f"header says {self._shape}")
        return arrays

    def _as_read(self, arrays: _Arrays) -> 'Data':
        """
        Return a copy of it with arrays read for it (see _fresh), not
        checked again.
        """
        item = copy.copy(self)
        item._arrays, item._read = arrays, None
        return item

    def __copy__(self) -> 'Data':
        # (its values not read for it: a pickle reads them, see
        # __getstate__)
        item = Data.__new__(Data)
        item.__dict__.update(self.__dict__)
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_arrays'], state['_read'] = self._values(), None
        return state


class _FileReader:
    """
    Reads the values of a data item of files opened (see Data.from_files),
    with one error for all of them, if given.
    """

    def __init__(self, opened: OpenedArrays, error: float | None):
        self._opened = opened
        self._error = error

    def __call__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        arrays = self._opened.read()
        errors = arrays.error if self._error is None \
            else np.full(arrays.data.shape, self._error, dtype=float)
        return arrays.data, arrays.mask, errors


class _Changed:
    """
    Makes the values of a data item that a change makes of another (see
    Data.then), of its values as they are read then: those of the other
    are not kept by it, which is no longer used.
    """

    def __init__(self, item: Data, change: Callable[[Data], Data]):
        self._item = item
        self._change = change

    def __call__(self) -> Data:
        item = self._item
        return self._change(
            item if item.is_read() else item._as_read(item._fresh()))


def _make_beam(beam: Beam | Mapping[str, Any] | None) -> Beam | None:
    """
    Return a beam from itself or its options. Raise ValueError if its
    widths are not positive.
    """
    if beam is None:
        return None
    if not isinstance(beam, Beam):
        beam = Beam(**parse_options(
            beam, required={'bmaj', 'bmin'}, optional={'bpa', 'spectral'}))
    if not (np.all(np.asarray(beam.bmaj) > 0)
            and np.all(np.asarray(beam.bmin) > 0)):
        raise ValueError(
            f"the widths of a beam must be positive; they are {beam.bmaj} "
            f"and {beam.bmin}")
    return beam


def dump_data(
        data: Data,
        filenames: Mapping[str, str],
        write: Callable[[str, np.ndarray], None],
        dump_path: bool = True,
        prefix: str = '',
        overwrite: bool = False
) -> dict[str, Any]:
    """
    Write the arrays of a data item to files, and return its
    configuration: the file of each array, its unit and beam (one beam,
    not those of the channels of a cube; see Data.from_files), and the
    correlations of its noise, if any.

    Parameters
    ----------
    data : Data
        The data item.
    filenames : Mapping
        The file of each array to write ('data', 'mask', 'error').
    write : Callable
        Writes an array to a file, as write(filename, array) (e.g. with
        the world coordinates of the dataset).
    dump_path : bool, optional
        Whether the configuration has the paths of the files, or only
        their names.
    prefix : str, optional
        The start of the names of the files of the correlations (e.g. of
        an image), if any.
    overwrite : bool, optional
        Whether to overwrite the existing files of the correlations.

    Returns
    -------
    dict
        The configuration.
    """
    arrays = dict(data=data.data(), mask=data.mask(), error=data.error())
    info = {}
    for key, filename in filenames.items():
        if arrays[key] is not None:
            write(filename, arrays[key])
            info[key] = filename if dump_path else os.path.basename(filename)
    if data.unit() is not None:
        info['unit'] = gbkfit.units.format_unit(data.unit())
    beam = data.beam()
    if beam is not None and beam.spectral is None:
        info['beam'] = dict(
            bmaj=float(beam.bmaj), bmin=float(beam.bmin), bpa=float(beam.bpa))
    elif beam is not None:
        warn(
            "the beams of the channels of a data item are not dumped; give "
            "its beam in its configuration (beam: {bmaj, bmin, bpa})")
    kwargs = dict(dump_path=dump_path, overwrite=overwrite)
    if (spatial := data.spatial_correlation()) is not None:
        info['spatial_correlation'] = psf_parser.dump(
            spatial, prefix=f'{prefix}spatial_correlation_', **kwargs)
    if (spectral := data.spectral_correlation()) is not None:
        info['spectral_correlation'] = lsf_parser.dump(
            spectral, prefix=f'{prefix}spectral_correlation_', **kwargs)
    return info


def item_filenames(prefix: str, key: str) -> dict[str, str]:
    """Return the files of the arrays of a data item (see dump_data)."""
    return dict(
        data=f'{prefix}{key}_d.fits',
        mask=f'{prefix}{key}_m.fits',
        error=f'{prefix}{key}_e.fits')
