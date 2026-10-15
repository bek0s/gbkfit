"""
The drivers: the memory and the native (C++/CUDA) modules of a device,
the host CPU or a CUDA GPU, on which observations evaluate their models
(and galaxies their gravity); and the FFT of a driver.
"""

import abc
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
import numpy.typing as npt

import gbkfit.math
from gbkfit.utils import parseutils

if TYPE_CHECKING:
    import cupy


__all__ = [
    'DeviceArray',
    'Driver',
    'DriverFFT',
    'FFTConvolution',
    'driver_by_name',
    'driver_parser'
]


# An array in the memory of a driver: a numpy array for the host driver,
# a CuPy array for the cuda driver
DeviceArray: TypeAlias = Union[np.ndarray, 'cupy.ndarray']

# The shape of an array
Shape: TypeAlias = int | tuple[int, ...]

# The floating types of the native modules, and the suffixes of their
# classes (e.g. GModelf32)
_NATIVE_CLASS_SUFFIXES = {
    np.dtype(np.float32): 'f32',
    np.dtype(np.float64): 'f64'}


class FFTConvolution(abc.ABC):
    """
    The convolution of the cubes of a shape by a kernel, in place, by FFTs
    on a driver (see DriverFFT.convolution). It keeps the spectrum of the
    kernel, made once (not the kernel), and a buffer for the spectra of
    the cubes.
    """

    @abc.abstractmethod
    def __call__(self, cube: DeviceArray) -> None:
        """Convolve a cube of the shape of the convolution, in place."""


class DriverFFT(abc.ABC):
    """
    The FFT of a driver, for the convolutions of the cubes of the models
    (see DCube): the shapes of the arrays of the convolutions, and the
    convolutions.

    Parameters
    ----------
    fft_shape_threshold : int
        The size up to which an axis of a convolution is padded to a power
        of 2; larger axes are padded to a multiple of it.

    Attributes
    ----------
    separable : bool
        Whether a PSF and an LSF that do not vary along the spectral axis
        are convolved as those that do, by 2D FFTs of each channel and
        sums along it, rather than by one 3D FFT of their product: the
        same convolution, faster on some drivers (see DCubePlan).
    """

    separable: bool = False

    @staticmethod
    def fft_convolution_shift(
            data: np.ndarray,
            axes: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """
        Roll the centre of a kernel (at shape // 2, or one pixel before for
        an even shape) to the first pixel, along the given axes (all by
        default).
        """
        axes = tuple(range(data.ndim)) if axes is None else tuple(axes)
        shift = [data.shape[axis] // 2 + 1 for axis in axes]
        return np.roll(data, shift, axis=axes)

    def __init__(self, fft_shape_threshold: int):
        self._fft_shape_threshold = fft_shape_threshold

    def fft_optimal_shape(self, shape: tuple[int, ...]) -> np.ndarray:
        """
        Return the shape to pad an array to, for fast transforms: each axis
        rounded up to a power of 2, or to a multiple of the threshold if
        larger.
        """
        threshold = self._fft_shape_threshold
        new_shape = []
        for dim in shape:
            dim_po2 = gbkfit.math.roundu_po2(dim)
            dim_mul = gbkfit.math.roundu_multiple(dim, threshold)
            dim_new = dim_po2 if dim_po2 <= threshold else dim_mul
            new_shape.append(dim_new)
        return np.array(new_shape)

    def fft_convolution_shape(
            self,
            data1_shape: tuple[int, ...],
            data2_shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Return the shape of the arrays of the convolution of two arrays
        (without wrap-around, and fast to transform), and the margin of the
        first one in it.
        """
        shape = np.array(data1_shape) + np.array(data2_shape) - 1
        shape = self.fft_optimal_shape(shape)
        margin = np.array(data2_shape) // 2
        return tuple(shape), tuple(margin)

    def convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...] = (0, 1, 2),
            dtype: npt.DTypeLike | None = None
    ) -> FFTConvolution:
        """
        Return the convolution of the cubes of a shape (z, y, x) by a
        kernel: of the whole cubes, or of each channel (the first axis) by
        its image. The transforms are of the floating type of the FFT
        (dtype()), the cubes of that or a less precise one (e.g. float32
        cubes, with float64 transforms that do not spread the rounding
        of float32 ones over the cubes).

        Parameters
        ----------
        kernel : numpy.ndarray
            The kernel, on the host, of the floating type of the FFT and
            with its centre at the first pixel (see fft_convolution_shift):
            a cube of the shape for axes (0, 1, 2); for axes (1, 2), an
            image for each channel, or one for all (of shape (1, y, x)).
        shape : tuple of int
            The shape of the cubes.
        axes : tuple of int, optional
            The axes of the convolution: (0, 1, 2) (by default), or (1, 2).
        dtype : dtype, optional
            The floating type of the cubes: that of the FFT by default.

        Raises
        ------
        ValueError
            If the axes are not one of those, the kernel does not have the
            shape or the floating type of the FFT, or the cubes are more
            precise than the FFT or not of a floating type of the native
            modules.
        """
        shape = tuple(int(n) for n in shape)
        axes = tuple(axes)
        if axes not in ((0, 1, 2), (1, 2)):
            raise ValueError(
                f"axes must be (0, 1, 2) or (1, 2); they are {axes}")
        one_image = axes == (1, 2) and kernel.shape == (1,) + shape[1:]
        if kernel.shape != shape and not one_image:
            raise ValueError(
                f"the kernel must have the shape of the cubes {shape}, or "
                f"one image for all their channels (axes (1, 2)); its shape "
                f"is {kernel.shape}")
        if kernel.dtype != self.dtype():
            raise ValueError(
                f"the kernel must be of the type of the FFT, "
                f"{self.dtype().name}; it is {kernel.dtype.name}")
        dtype = self.dtype() if dtype is None else np.dtype(dtype)
        if dtype not in _NATIVE_CLASS_SUFFIXES \
                or dtype.itemsize > self.dtype().itemsize:
            raise ValueError(
                f"the cubes must be of a floating type of the native "
                f"modules no more precise than the FFT "
                f"({self.dtype().name}); they are {dtype.name}")
        return self._convolution(
            np.ascontiguousarray(kernel), shape, axes, dtype)

    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Return the floating type of the transforms."""

    @abc.abstractmethod
    def _convolution(
            self,
            kernel: np.ndarray,
            shape: tuple[int, int, int],
            axes: tuple[int, ...],
            dtype: np.dtype
    ) -> FFTConvolution:
        """Return the convolution (see convolution), of checked arguments."""


class Driver(parseutils.TypedSerializable, abc.ABC):
    """
    A device on which models are evaluated: its memory, its native module
    (the C++ or CUDA kernels, in float32 and float64) and its FFT. All its
    work (its memory, transfers and FFTs, and the kernels of its native
    module on its arrays) runs on its device, whatever the thread that
    asks for it, so that the drivers of several devices work in one
    process, and in several threads at once.

    Arrays on the device are DeviceArrays; the host side of a transfer is
    a numpy array (in pinned memory for the cuda driver).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @staticmethod
    def dtypes() -> tuple[np.dtype, ...]:
        """Return the floating types of the native modules."""
        return tuple(_NATIVE_CLASS_SUFFIXES)

    @abc.abstractmethod
    def mem_alloc_s(
            self,
            shape: Shape,
            dtype: npt.DTypeLike
    ) -> tuple[np.ndarray, DeviceArray]:
        """
        Allocate a "shared" array: a host array and a device array of the
        same shape, for data that go between them. With managed memory
        (and on the host driver) they are one array.
        """

    @abc.abstractmethod
    def mem_alloc_h(self, shape: Shape, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate an array on the host, for transfers."""

    @abc.abstractmethod
    def mem_alloc_d(self, shape: Shape, dtype: npt.DTypeLike) -> DeviceArray:
        """Allocate an array on the device."""

    @abc.abstractmethod
    def mem_copy_h2d(
            self,
            h_src: np.ndarray,
            d_dst: DeviceArray | None = None
    ) -> DeviceArray:
        """
        Copy a host array to the device (into d_dst, or a new array), and
        return the device array.
        """

    @abc.abstractmethod
    def mem_copy_d2h(
            self,
            d_src: DeviceArray,
            h_dst: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Copy a device array to the host (into h_dst, or a new array), and
        return the host array.
        """

    @abc.abstractmethod
    def mem_fill(self, x: DeviceArray, value: float) -> None:
        """Set every value of a device array."""

    @abc.abstractmethod
    def math_add(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        """Return x1 + x2 (into out, if given)."""

    @abc.abstractmethod
    def math_mul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        """Return x1 * x2 (into out, if given)."""

    @abc.abstractmethod
    def native_module(self) -> types.ModuleType:
        """Return the native (C++/CUDA) module of the driver."""

    def native_class(self, name: str, dtype: npt.DTypeLike) -> type:
        """
        Return a class of the native module, for a floating type (e.g.
        'GModel' for float32 is the class GModelf32).

        Raises
        ------
        RuntimeError
            If the native module has no classes for the type (see dtypes;
            observations check their types when loaded).
        """
        dtype = np.dtype(dtype)
        if dtype not in _NATIVE_CLASS_SUFFIXES:
            supported = [dt.name for dt in _NATIVE_CLASS_SUFFIXES]
            raise RuntimeError(
                f"the {self.type()} driver does not support dtype "
                f"{dtype.name}; supported dtypes: {supported}")
        return getattr(
            self.native_module(), name + _NATIVE_CLASS_SUFFIXES[dtype])

    @abc.abstractmethod
    def fft(self, dtype: npt.DTypeLike) -> DriverFFT:
        """Return a new FFT of the driver, for a floating type."""


driver_parser = parseutils.TypedParser(Driver)


def driver_by_name(
        name: object,
        drivers: Mapping[str | None, Driver],
        owner: str
) -> Driver | None:
    """
    Return the driver of the given name among the drivers of a
    configuration, which it declares at its top level (the one unnamed
    driver of a configuration has the name None).

    Parameters
    ----------
    name : object
        The name, from the option driver of the owner; None for the only
        driver (none if there are none).
    drivers : Mapping of str to Driver
        The drivers, by name.
    owner : str
        What names the driver, in messages (e.g. 'observation').

    Returns
    -------
    Driver or None
        The driver; None if the name is None and there are no drivers.

    Raises
    ------
    ConfigError
        If the name is not one of the drivers, or is left out while there
        are several drivers.
    """
    names = [n for n in drivers if n is not None]
    if name is None:
        if len(drivers) > 1:
            raise parseutils.ConfigError(
                f"the {owner} must name its driver; the drivers are {names}")
        return next(iter(drivers.values()), None)
    if not isinstance(name, str):
        raise parseutils.ConfigError(
            f"the driver of the {owner} is the name of a driver that the "
            f"configuration declares at its top level (section 'driver' for "
            f"one driver, 'drivers' for several by name); it is {name!r}")
    if name not in names:
        raise parseutils.ConfigError(
            f"unknown driver {name!r}; the named drivers are {names}")
    return drivers[name]
