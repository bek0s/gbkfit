
import abc

import numpy as np

import gbkfit.math
from gbkfit.utils import parseutils


__all__ = [
    'LSF',
    'PSF',
    'lsf_parser',
    'psf_parser'
]


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
            offset: tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Return the PSF as a NumPy array.

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
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        """Abstract method to compute the base size of the PSF."""
        pass

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int]
    ) -> np.ndarray:
        """
        Abstract method to generate the PSF as a NumPy array.
        """
        pass


lsf_parser = parseutils.TypedParser(LSF)
psf_parser = parseutils.TypedParser(PSF)
