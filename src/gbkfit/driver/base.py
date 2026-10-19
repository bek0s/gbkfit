"""
The drivers: the memory and the native (C++/CUDA) modules of a device,
the host CPU or a CUDA GPU, on which observations evaluate their models
(and galaxies their gravity); and the FFTs and non-uniform FFTs of a
driver.
"""

import abc
import numbers
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
    'DriverNUFFT',
    'FFTConvolution',
    'NUFFT',
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


class NUFFT(abc.ABC):
    """
    A non-uniform discrete Fourier transform by a NUFFT on a driver (see
    DriverNUFFT), made for its points: applied to any data at them, and
    moved to other points (set_points) faster than another is made.

    Parameters
    ----------
    nufft_type : int
        Its type: 1, 2 or 3 (see DriverNUFFT).
    ndim : int
        The number of dimensions of its points.
    """

    def __init__(self, nufft_type: int, ndim: int):
        self._type = nufft_type
        self._ndim = ndim

    @abc.abstractmethod
    def __call__(self, data: DeviceArray, out: DeviceArray) -> None:
        """
        Transform data into out: contiguous complex128 arrays on the
        driver, of the shapes of the input and the output of the
        transform (see DriverNUFFT).
        """

    def set_points(
            self,
            points: np.ndarray,
            frequencies: np.ndarray | None = None
    ) -> None:
        """
        Move the transform to other points, and other frequencies for type
        3 (see DriverNUFFT), of as many dimensions.

        Parameters
        ----------
        points : numpy.ndarray
            The points, of shape (n, d), on the host.
        frequencies : numpy.ndarray, optional
            The frequencies of type 3, of shape (m, d), on the host.

        Raises
        ------
        ValueError
            If the points or the frequencies are not of the dimensions of
            the transform, or frequencies are given to a transform of type
            1 or 2, or not to one of type 3.
        """
        if (frequencies is None) != (self._type != 3):
            raise ValueError(
                "frequencies are those of a transform of type 3, which needs "
                "them")
        points = _nufft_columns(points, 'points', 2 * np.pi)
        frequencies = () if frequencies is None \
            else _nufft_columns(frequencies, 'frequencies', 1)
        if len(points) != self._ndim \
                or (frequencies and len(frequencies) != self._ndim):
            raise ValueError(
                f"the points and frequencies must be of the {self._ndim} "
                f"dimensions of the transform")
        self._set_points(points, frequencies)

    @abc.abstractmethod
    def _set_points(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...]
    ) -> None:
        """
        Move the transform to other points (see set_points), of checked
        columns: those of the points, times 2 pi, and of the frequencies
        (none for types 1 and 2).
        """


class DriverNUFFT(abc.ABC):
    """
    The non-uniform FFTs of a driver (FINUFFT): sums of complex values c_j
    at points x_j (of 1 to 3 dimensions),

        f_k = sum_j c_j exp(sign 2 pi i x_j . s_k),

    at frequencies s_k (type 3; e.g. the visibilities of a sky model) or
    at the integer frequencies of the modes of an array (type 1), and the
    sums of such modes at points (type 2), to a relative tolerance eps
    (the 2-norm of the error of the result over that of the result).
    Their points are fixed when they are made, after which each transform
    is fast.

    They are of float64, of complex128 arrays: float32 rounds phases of
    hundreds of radians to more than the tolerance (e.g. errors of 4e-5 in
    the visibilities of 256 x 256 pixels sampling 8 km baselines at 21 cm,
    measured 2026-10-18). Not supported yet: float32 NUFFTs, as an opt-in
    approximation (about twice as fast on a GPU; cuFINUFFT 2.5.1 then needs
    upsampfac=2 for type 3 to reach its tolerance).
    """

    # TODO: float32 NUFFTs, as an opt-in approximation (cuFINUFFT 2.5.1
    # needs upsampfac=2 for its type 3 to reach the tolerance)

    def type3(
            self,
            points: np.ndarray,
            frequencies: np.ndarray,
            sign: int,
            eps: float,
            ntransforms: int = 1
    ) -> NUFFT:
        """
        Return the NUFFT from points to frequencies (type 3),

            f_k = sum_j c_j exp(sign 2 pi i x_j . s_k).

        Parameters
        ----------
        points : numpy.ndarray
            The points x_j, of shape (n, d), on the host.
        frequencies : numpy.ndarray
            The frequencies s_k, in cycles per unit of the points, of shape
            (m, d), on the host.
        sign : {-1, 1}
            The sign of the exponent.
        eps : float
            The relative tolerance.
        ntransforms : int, optional
            The number of sets of values of each transform: its input is of
            shape (ntransforms, n), and its output (ntransforms, m).

        Raises
        ------
        ValueError
            If the points or the frequencies are not of 1 to 3 dimensions,
            or not of the same, the sign is not -1 or 1, the tolerance not
            positive, or the number of sets not a positive integer.
        """
        points = _nufft_columns(points, 'points', 2 * np.pi)
        frequencies = _nufft_columns(frequencies, 'frequencies', 1)
        if len(frequencies) != len(points):
            raise ValueError(
                f"the frequencies must be of the dimensions of the points "
                f"({len(points)}); they are of {len(frequencies)}")
        _check_nufft_options(sign, eps, ntransforms)
        return self._type3(points, frequencies, sign, eps, int(ntransforms))

    def type1(
            self,
            points: np.ndarray,
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int = 1
    ) -> NUFFT:
        """
        Return the NUFFT from points to the modes of an array (type 1): of
        the index q of the array f of a shape,

            f[q] = sum_j c_j exp(sign 2 pi i x_j . k),  k = q - shape // 2

        (e.g. the dirty image of visibilities, at x_j their baselines in
        cycles per pixel).

        Parameters
        ----------
        points : numpy.ndarray
            The points x_j, in cycles per unit of k, of shape (n, d), on
            the host: a column for each axis of the array.
        shape : tuple of int
            The shape of the array, of d = 1 to 3 axes.
        sign : {-1, 1}
            The sign of the exponent.
        eps : float
            The relative tolerance.
        ntransforms : int, optional
            The number of sets of values of each transform: its input is of
            shape (ntransforms, n), and its output (ntransforms, *shape).

        Raises
        ------
        ValueError
            If the points are not of 1 to 3 dimensions, or not of those of
            the array, the sign is not -1 or 1, the tolerance not positive,
            or the number of sets not a positive integer.
        """
        points, shape = _array_points(points, shape)
        _check_nufft_options(sign, eps, ntransforms)
        return self._type1(points, shape, sign, eps, int(ntransforms))

    def type2(
            self,
            points: np.ndarray,
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int = 1
    ) -> NUFFT:
        """
        Return the NUFFT from the modes of arrays to points (type 2, the
        adjoint of type 1): of the index q of an array f of a shape,

            c_j = sum_q f[q] exp(sign 2 pi i x_j . k),  k = q - shape // 2

        (e.g. the visibilities of an image whose pixels are uniform in the
        direction cosines, at x_j their baselines in cycles per pixel), of
        several arrays at once.

        Parameters
        ----------
        points : numpy.ndarray
            The points x_j, in cycles per unit of k, of shape (n, d), on
            the host: a column for each axis of the arrays.
        shape : tuple of int
            The shape of the arrays, of d = 1 to 3 axes.
        sign : {-1, 1}
            The sign of the exponent.
        eps : float
            The relative tolerance.
        ntransforms : int, optional
            The number of arrays of each transform: its input is of shape
            (ntransforms, *shape), and its output (ntransforms, n).

        Raises
        ------
        ValueError
            If the points are not of 1 to 3 dimensions, or not of those of
            the arrays, the sign is not -1 or 1, the tolerance not
            positive, or the number of arrays not a positive integer.
        """
        points, shape = _array_points(points, shape)
        _check_nufft_options(sign, eps, ntransforms)
        return self._type2(points, shape, sign, eps, int(ntransforms))

    @abc.abstractmethod
    def _type3(
            self,
            points: tuple[np.ndarray, ...],
            frequencies: tuple[np.ndarray, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        """
        Return the NUFFT of type 3 (see type3) of checked arguments: the
        columns of the points, times 2 pi, and of the frequencies.
        """

    @abc.abstractmethod
    def _type1(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        """
        Return the NUFFT of type 1 (see type1) of checked arguments: the
        columns of the points, times 2 pi.
        """

    @abc.abstractmethod
    def _type2(
            self,
            points: tuple[np.ndarray, ...],
            shape: tuple[int, ...],
            sign: int,
            eps: float,
            ntransforms: int
    ) -> NUFFT:
        """
        Return the NUFFT of type 2 (see type2) of checked arguments: the
        columns of the points, times 2 pi.
        """


def _nufft_columns(
        points: np.ndarray,
        name: str,
        scale: float
) -> tuple[np.ndarray, ...]:
    """
    Return the columns of the points of a NUFFT, of shape (n, d), times a
    scale, as contiguous float64 arrays.
    """
    points = np.asarray(points)
    if points.ndim != 2 or not 1 <= points.shape[1] <= 3:
        raise ValueError(
            f"the {name} must be of shape (n, d), of d = 1 to 3 dimensions; "
            f"their shape is {points.shape}")
    return tuple(
        np.ascontiguousarray(scale * column, np.float64)
        for column in points.T)


def _array_points(
        points: np.ndarray,
        shape: tuple[int, ...]
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """
    Return the columns of the points of a NUFFT of the modes of arrays (see
    _nufft_columns), times 2 pi, and the shape of the arrays.
    """
    points = _nufft_columns(points, 'points', 2 * np.pi)
    shape = tuple(int(n) for n in shape)
    if len(shape) != len(points):
        raise ValueError(
            f"the arrays must have an axis for each dimension of the points "
            f"({len(points)}); their shape is {shape}")
    return points, shape


def _check_nufft_options(sign: int, eps: float, ntransforms: int) -> None:
    """
    Check the sign of the exponent, the tolerance and the number of sets
    of values of a NUFFT.
    """
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or 1; it is {sign}")
    if not eps > 0:
        raise ValueError(f"eps must be positive; it is {eps}")
    if not isinstance(ntransforms, numbers.Integral) or ntransforms < 1:
        raise ValueError(
            f"ntransforms must be a positive integer; it is {ntransforms!r}")


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
    def mem_copy_d2d(self, d_src: DeviceArray, d_dst: DeviceArray) -> None:
        """
        Copy a device array into another of its shape, in the type of the
        other (e.g. the real parts of complex128 values into float32).
        """

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
    def math_matmul(
            self,
            x1: DeviceArray,
            x2: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        """Return the matrix product x1 @ x2 (into out, if given)."""

    @abc.abstractmethod
    def mem_gather(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray | None = None
    ) -> DeviceArray:
        """
        Return the values of a device array at indices along its last axis,
        x[..., indices] (into out, if given).
        """

    @abc.abstractmethod
    def mem_scatter(
            self,
            x: DeviceArray,
            indices: DeviceArray,
            out: DeviceArray
    ) -> None:
        """
        Write the values of a device array to indices along the last axis
        of another, out[..., indices] = x.
        """

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

    @abc.abstractmethod
    def nufft(self) -> DriverNUFFT:
        """
        Return the non-uniform FFTs of the driver.

        Raises
        ------
        ModuleNotFoundError
            If the NUFFT library of the driver is not installed (an
            optional dependency).
        """


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
