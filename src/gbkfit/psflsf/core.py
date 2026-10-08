
import abc

import numpy as np

import gbkfit.math
from gbkfit.utils import parseutils


__all__ = [
    'LSF',
    'PSF',
    'lsf_parser',
    'psf_parser',
    'MIN_EXTENT',
    'WING_FLUX',
    'check_scale',
    'check_ratio'
]


# The analytic PSFs and LSFs are drawn out to at least MIN_EXTENT scale
# lengths (e.g. sigma), and further for profiles with wide wings: until
# the wings beyond hold at most WING_FLUX of their flux. Beyond that
# they are 0, and the arrays are normalised to sum to 1.
MIN_EXTENT = 8
WING_FLUX = 0.01


def check_scale(name: str, value: float) -> None:
    """Raise RuntimeError unless a scale length (e.g. sigma) is > 0."""
    if not value > 0:
        raise RuntimeError(f"{name} must be greater than 0; it is {value}")


def check_ratio(ratio: float) -> None:
    """Raise RuntimeError unless an axis ratio is in (0, 1]."""
    if not 0 < ratio <= 1:
        raise RuntimeError(
            f"ratio must be greater than 0 and at most 1; it is {ratio}")


class LSF(parseutils.TypedSerializable, abc.ABC):

    def size(self, step: float, offset: int = 0) -> int:
        """
        Compute LSF size with offset, ensuring it's odd.
        """
        base_size = self._size_impl(step)
        return int(gbkfit.math.roundu_odd(base_size + offset))

    def asarray(
            self,
            step: float,
            size: int | None = None,
            offset: int = 0
    ) -> np.ndarray:
        """
        Return the LSF as a NumPy array.

        If `size` is None, it is set by `self.size(step, offset)`.
        The value (size + offset) must be odd.
        """
        if size is None:
            size = self.size(step, offset)
        elif gbkfit.math.is_even(size + offset):
            raise RuntimeError(
                f"invalid LSF size: (size + offset) = "
                f"({size} + {offset} = {size + offset}), "
                f"but it must be odd")
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step: float) -> float:
        """Abstract method to compute the base size of the LSF."""
        pass

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: float,
            size: int,
            offset: int
    ) -> np.ndarray:
        """
        Abstract method to generate the LSF as a NumPy array.
        """
        pass


class PSF(parseutils.TypedSerializable, abc.ABC):

    def size(
            self,
            step: tuple[float, float],
            offset: tuple[int, int] = (0, 0)
    ) -> tuple[int, int]:
        """
        Compute PSF size with offset, ensuring it's odd.
        """
        base_size = self._size_impl(step)
        return (int(gbkfit.math.roundu_odd(base_size[0] + offset[0])),
                int(gbkfit.math.roundu_odd(base_size[1] + offset[1])))

    def asarray(
            self,
            step: tuple[float, float],
            size: tuple[int, int] | None = None,
            offset: tuple[int, int] = (0, 0),
            rota: float = 0
    ) -> np.ndarray:
        """
        Return the PSF as a NumPy array, on a grid rotated on the sky by
        rota (degrees, counterclockwise like a position angle): a PSF
        with position angle posa (on the sky) is drawn at posa - rota.

        If `size` is None, it is set by `self.size(step, offset)`.
        Both (size + offset) values must be odd.
        """
        if size is None:
            size = self.size(step, offset)
        elif (gbkfit.math.is_even(size[0] + offset[0]) or
              gbkfit.math.is_even(size[1] + offset[1])):
            raise RuntimeError(
                f"invalid PSF size: (size + offset) = "
                f"({size[0]} + {offset[0]} = {size[0] + offset[0]}, "
                f"{size[1]} + {offset[1]} = {size[1] + offset[1]}), "
                f"but both values must be odd")
        return self._asarray_impl(step, size, offset, rota)

    @abc.abstractmethod
    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        """Abstract method to compute the base size of the PSF."""
        pass

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        """
        Abstract method to generate the PSF as a NumPy array, on a grid
        rotated by rota.
        """
        pass


lsf_parser = parseutils.TypedParser(LSF)
psf_parser = parseutils.TypedParser(PSF)
