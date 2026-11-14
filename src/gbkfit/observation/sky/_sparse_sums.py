"""
The products of a sparse matrix with each channel of cubes: weighted sums
of their pixels (e.g. in regions, or the dispersion of slitless spectra).
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse

from gbkfit.driver import DeviceArray, Driver


# The most pixels a thread sums: larger rows (e.g. a region of the whole
# field) are summed in chunks of this many pixels, in parallel, and then
# the chunks of each row are summed
_CHUNK = 256


class SparseSumsPlan:
    """
    The products of a sparse matrix (nrows, npixels) with each channel of
    cubes, on a driver: the sums of the pixels of each channel with the
    weights of each row (e.g. those of the pixels of regions, see
    Regions.weights).
    """

    def __init__(
            self,
            weights: scipy.sparse.csr_array,
            driver: Driver,
            dtype: np.dtype
    ):
        weights = scipy.sparse.csr_array(weights)
        weights.sum_duplicates()
        indptr = weights.indptr
        self._nrows = weights.shape[0]
        self._npix = weights.shape[1]
        self._driver = driver
        self._dtype = dtype
        self._backend = driver.native_class('DModel', dtype)()

        def to_device(
                values: npt.ArrayLike, dtype_: npt.DTypeLike
        ) -> DeviceArray:
            return driver.mem_copy_h2d(np.asarray(values).astype(dtype_))

        # The chunks of each row: from each start, at most _CHUNK pixels
        # to its end (the rows are contiguous in the matrix, and empty rows
        # have none)
        starts = indptr[:-1].astype(np.int64)
        counts = (indptr[1:] - starts + _CHUNK - 1) // _CHUNK
        self._chunked = bool(np.any(counts > 1))
        if not self._chunked:
            self._stages = [(
                to_device(indptr, np.int32),
                to_device(weights.indices, np.int32),
                to_device(weights.data, dtype))]
            return
        # (the chunks of row i are bounds[i] to bounds[i + 1]; chunk k of a
        # row starts k _CHUNK pixels after it)
        bounds = np.concatenate(([0], np.cumsum(counts)))
        nchunks = int(bounds[-1])
        chunk_starts = np.repeat(starts, counts) + _CHUNK * (
            np.arange(nchunks) - np.repeat(bounds[:-1], counts))
        # Stage 1: the sums of the chunks (the last ends with the matrix);
        # stage 2: the sums of the chunks of each row
        self._stages = [
            (to_device(np.append(chunk_starts, indptr[-1]), np.int32),
             to_device(weights.indices, np.int32),
             to_device(weights.data, dtype)),
            (to_device(bounds, np.int32),
             to_device(np.arange(nchunks), np.int32),
             to_device(np.ones(nchunks), dtype))]
        self._nchunks = nchunks
        self._partial = None

    def nrows(self) -> int:
        """Return the number of rows of the matrix: the sums of a channel."""
        return self._nrows

    def evaluate(self, cube: DeviceArray, out: DeviceArray) -> None:
        """
        Sum the rows in each channel of a cube (nz, ny, nx; ny * nx pixels
        of the weights), into out (nz, nrows).
        """
        if cube.shape[1] * cube.shape[2] != self._npix:
            raise RuntimeError(
                f"the matrix has weights for {self._npix} pixels, but the "
                f"cube has {cube.shape[1] * cube.shape[2]} in each channel")
        if not self._chunked:
            self._backend.regions_sum(*self._stages[0], cube, out)
            return
        if self._partial is None or self._partial.shape[0] != cube.shape[0]:
            self._partial = self._driver.mem_alloc_d(
                (cube.shape[0], 1, self._nchunks), self._dtype)
        self._backend.regions_sum(
            *self._stages[0], cube, self._partial[:, 0, :])
        self._backend.regions_sum(*self._stages[1], self._partial, out)
