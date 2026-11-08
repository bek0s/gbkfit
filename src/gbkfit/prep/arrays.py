"""
Data read from files and prepared for the datasets of gbkfit (see
FitsArrays): in the orientation of the model, cropped, padded, masked
and clipped.
"""

import dataclasses
from collections.abc import Callable, Sequence

import astropy.io.fits as fits
import astropy.units
import astropy.wcs
import numpy as np
import numpy.typing as npt

from gbkfit.config import DataFile
from gbkfit.io import (
    Arrays, Beam, ErrorKind, has_wcs, read_arrays, read_sky_map,
    shifted_header, write_image)
from .masks import bounding_box


__all__ = [
    'FitsArrays'
]


@dataclasses.dataclass(frozen=True)
class FitsArrays(Arrays):
    """
    Data read from files (see Arrays), being prepared for a dataset: in
    the orientation of the model (see read_arrays; e.g. RA before Dec,
    and velocities increasing), the data and errors NaN where the mask is
    0.

    The methods return new arrays; those that move the first pixel
    (crop, minified, padded) keep the world coordinates of the pixels,
    and have no orientation (they are no longer on the pixels of the
    file, see read_sky_map).
    """

    @classmethod
    def read(
            cls,
            data: DataFile,
            error: DataFile | None = None,
            mask: DataFile | None = None,
            velocity: str | None = None,
            *,
            rest: str | astropy.units.Quantity | None = None,
            error_kind: ErrorKind = 'sigma',
            spectral_type: str | None = None
    ) -> 'FitsArrays':
        """
        Read the data, errors and mask from files in the orientation of
        the model, as the model reads them (see read_arrays):
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
            frequencies into velocities (see model_header).
        error_kind : str, optional
            What the file of the errors holds (see read_arrays).
        spectral_type : str, optional
            The type of an axis whose header gives one that FITS does not
            define (see model_header).

        Returns
        -------
        FitsArrays
            The arrays.

        Raises
        ------
        ValueError
            If the files cannot be read as data, errors and mask (see
            read_arrays), or the rest is given for data without
            a spectral axis.
        """
        arrays = read_arrays(
            data, error, mask, error_kind=error_kind, rest=rest,
            velocity=velocity, spectral_type=spectral_type)
        header = arrays.header
        if rest is not None and not (
                has_wcs(header)
                and astropy.wcs.WCS(header).wcs.spec >= 0):
            raise ValueError(
                f"the data have no spectral axis to give the rest {rest}")
        return cls(**vars(arrays))

    def read_sky_map(
            self, file: DataFile, plane: int | None = None
    ) -> tuple[np.ndarray, fits.Header]:
        """
        Read an image of the sky of the data (e.g. the bin of each pixel;
        of a plane of a file of them, if given) on their pixels, as their
        errors and mask are read (see gbkfit.io.read_sky_map), and its
        header.

        Raises
        ------
        ValueError
            If the arrays are no longer on the pixels of the file of the
            data (cropped or padded), or as read_sky_map.
        """
        if self.orientation is None:
            raise ValueError(
                "an image of the sky is read on the pixels of the file of "
                "the data, before they are cropped or padded")
        return read_sky_map(file, self.header, self.orientation, plane)

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
        return dataclasses.replace(
            self, data=self.data[slices],
            error=None if self.error is None else self.error[slices],
            mask=self.mask[slices], header=shifted_header(self.header, offset),
            beam=self._channels(lambda values: values[slices[0]]),
            footprint=None if self.footprint is None
            else self.footprint[slices],
            orientation=None)

    def minified(self) -> 'FitsArrays':
        """
        Return the arrays cropped to their valid pixels (see
        bounding_box).

        Raises
        ------
        ValueError
            If no pixel is valid.
        """
        return self.crop(bounding_box(self.mask))

    def padded(self, size: int) -> 'FitsArrays':
        """Return the arrays padded with invalid pixels on every side."""
        def pad(array: np.ndarray, value: float) -> np.ndarray:
            return np.pad(array, size, 'constant', constant_values=value)
        # (the beams of the channels padded with those of the end channels,
        # of the invalid pixels)
        return dataclasses.replace(
            self, data=pad(self.data, np.nan),
            error=None if self.error is None else pad(self.error, np.nan),
            mask=pad(self.mask, 0),
            header=shifted_header(self.header, [-size] * self.data.ndim),
            beam=self._channels(lambda values: np.pad(values, size, 'edge')),
            footprint=None if self.footprint is None
            else pad(self.footprint, 0),
            orientation=None)

    def _channels(
            self, change: Callable[[np.ndarray], np.ndarray]
    ) -> Beam | None:
        """
        Return its beam, its beams of the channels of a cube (if it has
        those) changed as its channels (by change, of an array along the
        channels, its first axis).
        """
        if self.beam is None or self.beam.spectral is None \
                or self.data.ndim != 3:
            return self.beam
        return Beam(*(change(value) for value in self.beam))

    def write(
            self,
            data: str,
            error: str | None = None,
            mask: str | None = None,
            dtype: npt.DTypeLike = np.float32
    ) -> None:
        """
        Write the data, errors and mask (if given filenames) as FITS
        files, with the header of the data and their unit (the mask
        without one).

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
        # SCI, would misname; its unit is that of the arrays)
        header = self.header.copy()
        for key in ('EXTNAME', 'BUNIT'):
            header.remove(key, ignore_missing=True)
        for filename, array, unit in (
                (data, self.data, self.unit),
                (error, self.error, self.unit),
                (mask, self.mask, None)):
            if filename is not None and array is not None:
                write_image(
                    filename, array.astype(dtype), header, unit=unit,
                    overwrite=True)
