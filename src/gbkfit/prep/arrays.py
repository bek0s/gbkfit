"""
Data read from files and prepared for the datasets of gbkfit (see
FitsArrays): in the orientation of the model, cropped, padded, masked
and clipped.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import astropy.io.fits as fits
import astropy.units
import astropy.wcs
import numpy as np
import numpy.typing as npt

from gbkfit.config import ConfigError, DataFile
from gbkfit.utils import datafiles, fitsutils
from .masks import bounding_box


__all__ = [
    'FitsArrays'
]


def _shift_axes(header: fits.Header, offset: npt.ArrayLike) -> fits.Header:
    """
    Return the header of the data whose first pixel is the pixel at the
    given offset (on each numpy axis) of the original data: positive where
    the data were cropped, negative where they were padded.
    """
    if not fitsutils.has_wcs(header):
        return header
    wcs = astropy.wcs.WCS(header)
    wcs.wcs.crpix -= np.asarray(offset)[::-1]
    return fitsutils.with_wcs(header, wcs)


@dataclasses.dataclass(frozen=True)
class FitsArrays:
    """
    Data read from files, being prepared for a dataset: the data, their
    errors (if any: standard deviations) and mask (1 where valid), with
    the header of the data (their world coordinates, if any), in the
    orientation of the model (see datafiles.read_arrays; e.g. RA before
    Dec, and velocities increasing). The data and errors are NaN where the
    mask is 0.

    The methods return new arrays; those that move the first pixel
    (crop, minified, padded) keep the world coordinates of the pixels.

    Attributes
    ----------
    data : np.ndarray
        The data.
    error : np.ndarray or None
        The errors.
    mask : np.ndarray
        The mask: 1 where valid, 0 elsewhere.
    header : fits.Header
        The header of the data.
    beam : dict or None
        The beam of the data, if any (see datafiles.Arrays): of a cube of
        CASA, that of each of its channels, in their order.
    """
    data: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray
    header: fits.Header
    beam: dict[str, Any] | None = None

    @classmethod
    def read(
            cls,
            data: DataFile,
            error: DataFile | None = None,
            mask: DataFile | None = None,
            velocity: str | None = None,
            *,
            rest: str | astropy.units.Quantity | None = None,
            error_kind: datafiles.ErrorKind = 'sigma'
    ) -> 'FitsArrays':
        """
        Read the data, errors and mask from files in the orientation of
        the model, as the model reads them (see datafiles.read_arrays):
        ranges of pixels (see crop) are those of the data it sees, with
        the spectral axis in velocities of the rest, if given.

        Parameters
        ----------
        data : DataFile
            The file of the data (with the world coordinates): a filename
            (its first HDU with data), or a filename and the HDU.
        error, mask : DataFile, optional
            The files of the errors and the mask.
        velocity : str, optional
            The convention of the velocities of the spectral axis, 'radio'
            or 'optical', where the header does not state it.
        rest : str or Quantity, optional
            The rest that turns a spectral axis of wavelengths or
            frequencies into velocities (see fitsutils.model_header).
        error_kind : str, optional
            What the file of the errors holds (see datafiles.read_arrays).

        Returns
        -------
        FitsArrays
            The arrays.

        Raises
        ------
        ConfigError
            If the files cannot be read as data, errors and mask (see
            datafiles.read_arrays), or the rest is given for data without
            a spectral axis.
        """
        arrays = datafiles.read_arrays(
            data, error, mask, error_kind=error_kind, rest=rest,
            velocity=velocity)
        header = arrays.header
        if rest is not None and not (
                fitsutils.has_wcs(header)
                and astropy.wcs.WCS(header).wcs.spec >= 0):
            raise ConfigError(
                f"the data have no spectral axis to give the rest {rest}")
        return cls(
            arrays.data, arrays.error, arrays.mask, header, arrays.beam)

    def masked(self, mask: npt.ArrayLike) -> 'FitsArrays':
        """
        Return the arrays with the pixels a mask marks invalid (0 or
        False) also invalid.
        """
        valid = (self.mask != 0) & (np.asarray(mask) != 0)
        data = np.where(valid, self.data, np.nan)
        error = None if self.error is None \
            else np.where(valid, self.error, np.nan)
        return dataclasses.replace(
            self, data=data, error=error, mask=valid.astype(float))

    def crop(self, slices: Sequence[slice]) -> 'FitsArrays':
        """
        Return the arrays cropped to ranges of pixels (one slice per numpy
        axis, of step 1).
        """
        slices = tuple(slices)
        offset = [s.indices(n)[0] for s, n in zip(slices, self.data.shape)]
        return FitsArrays(
            self.data[slices],
            None if self.error is None else self.error[slices],
            self.mask[slices], _shift_axes(self.header, offset),
            self._channels(lambda values: values[slices[0]]))

    def minified(self) -> 'FitsArrays':
        """
        Return the arrays cropped to their valid pixels (see
        bounding_box).

        Raises
        ------
        ConfigError
            If no pixel is valid.
        """
        return self.crop(bounding_box(self.mask))

    def padded(self, size: int) -> 'FitsArrays':
        """Return the arrays padded with invalid pixels on every side."""
        def pad(array: np.ndarray, value: float) -> np.ndarray:
            return np.pad(array, size, 'constant', constant_values=value)
        # (the beams of the channels padded with those of the end channels,
        # of the invalid pixels)
        return FitsArrays(
            pad(self.data, np.nan),
            None if self.error is None else pad(self.error, np.nan),
            pad(self.mask, 0),
            _shift_axes(self.header, [-size] * self.data.ndim),
            self._channels(lambda values: np.pad(values, size, 'edge')))

    def _channels(
            self, change: Callable[[np.ndarray], np.ndarray]
    ) -> dict[str, Any] | None:
        """
        Return its beam, its beams of the channels of a cube (if it has
        those) changed as its channels (by change, of an array along the
        channels, its first axis).
        """
        if self.beam is None or 'spectral' not in self.beam \
                or self.data.ndim != 3:
            return self.beam
        return {key: change(value) for key, value in self.beam.items()}

    def write(
            self,
            data: str,
            error: str | None = None,
            mask: str | None = None,
            dtype: npt.DTypeLike = np.float32
    ) -> None:
        """
        Write the data, errors and mask (if given filenames) as FITS
        files, with the header of the data (the mask without its unit).

        Parameters
        ----------
        data : str
            The file of the data.
        error, mask : str, optional
            The files of the errors and the mask.
        dtype : DTypeLike, optional
            The type of the values.
        """
        # (each file is of one array, which an HDU name of the input, e.g.
        # SCI, would misname)
        header = self.header.copy()
        header.remove('EXTNAME', ignore_missing=True)
        mask_header = header.copy()
        mask_header.remove('BUNIT', ignore_missing=True)
        for filename, array, array_header in (
                (data, self.data, header),
                (error, self.error, header),
                (mask, self.mask, mask_header)):
            if filename is not None and array is not None:
                fits.writeto(
                    filename, array.astype(dtype), array_header,
                    overwrite=True)
