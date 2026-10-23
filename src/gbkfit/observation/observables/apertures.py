"""
The apertures of slitless spectra (see SlitlessSpectra): a mask in the
focal plane in front of the disperser (e.g. the shutters of a
multi-object spectrograph, or a long slit), whose open parts let the
light of the sky through.

An aperture is placed along the axes of the image, x along the
dispersion, in arcsec from its reference pixel (see gridutils.Coords).
"""

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Aperture',
    'ApertureShutters',
    'aperture_parser'
]


class Aperture(parseutils.TypedSerializable, abc.ABC):
    """
    An aperture in front of a disperser: open rectangles along the axes
    of the image (see the module).
    """

    @abc.abstractmethod
    def rectangles(self) -> np.ndarray:
        """
        Return its open rectangles.

        Returns
        -------
        np.ndarray
            The edges of each (x0, x1, y0, y1; arcsec), of shape (n, 4).
        """
        pass

    def clip(
            self,
            centres: np.ndarray,
            sizes: np.ndarray,
            step: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Clip boxes (e.g. the pixels of the sky) to its open rectangles.

        Parameters
        ----------
        centres : np.ndarray
            The centres of the boxes (x, y; arcsec, see the module), of
            shape (n, 2).
        sizes : np.ndarray
            Their widths (arcsec), of shape (n, 2).
        step : Sequence of float
            The size of the pixels of the image (arcsec), to tell the open
            parts of the boxes that are too small to matter (below 1e-9
            of a pixel).

        Returns
        -------
        index : np.ndarray
            The box of each open part (a box has a part for each rectangle
            it overlaps).
        centres, sizes : np.ndarray
            The centres and the widths of the open parts.
        """
        rectangles = self.rectangles()
        lo = np.maximum(
            centres[:, None] - sizes[:, None] / 2, rectangles[None, :, 0::2])
        hi = np.minimum(
            centres[:, None] + sizes[:, None] / 2, rectangles[None, :, 1::2])
        widths = hi - lo
        open_ = np.all(widths > 1e-9 * np.asarray(step, float), axis=-1)
        index, _ = np.nonzero(open_)
        return index, ((lo + hi) / 2)[open_], widths[open_]


class ApertureShutters(Aperture):
    """
    A column of shutters along y (along the slit, across the dispersion),
    each open over a rectangle of a size, one pitch from the next (e.g.
    the slitlets of the micro-shutter array of JWST NIRSpec: shutters
    open over 0.20 x 0.46 arcsec, of a pitch of 0.27 x 0.53 arcsec); the
    bars between them block the light. One shutter is a slit.

    Parameters
    ----------
    size : Sequence of float
        The open width (along x) and length (along y) of a shutter
        (arcsec).
    pitch : float, optional
        The distance between the centres of neighbouring shutters along y
        (arcsec); required for more than one shutter.
    count : int, optional
        The number of shutters.
    centre : Sequence of float, optional
        The centre of the column (x, y; arcsec, see the module).

    Raises
    ------
    ConfigError
        If the size is not positive, the count is less than 1, or the
        pitch is missing or smaller than the length of a shutter.
    """

    @staticmethod
    def type() -> str:
        return 'shutters'

    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        info = dict(size=list(self._size), count=self._count,
                    centre=list(self._centre))
        if self._pitch is not None:
            info['pitch'] = self._pitch
        return info

    def __init__(
            self,
            size: Sequence[float],
            pitch: float | None = None,
            count: int = 1,
            centre: Sequence[float] = (0, 0)
    ):
        size = tuple(float(value) for value in size)
        if len(size) != 2 or not min(size) > 0:
            raise ConfigError(
                f"the size of the shutters must be two positive widths; it "
                f"is {size}")
        if count < 1:
            raise ConfigError(
                f"the count of the shutters must be at least 1; it is "
                f"{count}")
        if count > 1 and (pitch is None or pitch < size[1]):
            raise ConfigError(
                f"several shutters need a pitch of at least their length "
                f"({size[1]}); it is {pitch}")
        self._size = size
        self._pitch = pitch
        self._count = count
        self._centre = tuple(float(value) for value in centre)

    def rectangles(self) -> np.ndarray:
        width, length = self._size
        x, y = self._centre
        offsets = (np.arange(self._count) - (self._count - 1) / 2) \
            * (self._pitch or 0)
        return np.column_stack([
            np.full(self._count, x - width / 2),
            np.full(self._count, x + width / 2),
            y + offsets - length / 2, y + offsets + length / 2])


aperture_parser = parseutils.TypedParser(Aperture, [ApertureShutters])
