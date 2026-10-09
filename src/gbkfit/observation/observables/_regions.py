import numpy as np
import scipy.sparse


__all__ = [
    'RegionSumsPlan',
    'flux_weights'
]


# The most pixels a thread sums: larger regions (e.g. the whole field) are
# summed in chunks of this many pixels, in parallel, and then the chunks of
# each region are summed
_CHUNK = 256


def flux_weights(regions, grid):
    """
    The weights of the pixels of a spatial grid in regions (see
    Regions.weights) times the area of a pixel (arcsec^2): the area of
    each pixel inside each region. The sums of the surface brightness of
    the model with them are the fluxes of the regions, whatever the size
    of the pixels of the grid.
    """
    step_x, step_y = grid.coords.step[:2]
    return regions.weights(grid) * abs(step_x * step_y)


class RegionSumsPlan:
    """
    The weighted sums of the pixels of regions (see Regions.weights) in
    each channel of cubes, on a driver.
    """

    def __init__(self, weights: scipy.sparse.csr_array, driver, dtype):
        weights = scipy.sparse.csr_array(weights)
        weights.sum_duplicates()
        indptr = weights.indptr
        self._nregions = weights.shape[0]
        self._npix = weights.shape[1]
        self._driver = driver
        self._dtype = dtype
        self._backend = driver.native_class('DModel', dtype)()

        def to_device(values, dtype_):
            return driver.mem_copy_h2d(np.asarray(values).astype(dtype_))

        # The chunks of each region: from each start, at most _CHUNK
        # pixels to its end (the regions are contiguous in the matrix, and
        # empty regions have none)
        starts, ends = indptr[:-1], indptr[1:]
        chunk_starts = [np.arange(s, e, _CHUNK) for s, e in zip(starts, ends)]
        counts = np.array([len(c) for c in chunk_starts])
        chunk_starts = np.concatenate(chunk_starts).astype(np.int64)
        self._chunked = bool(np.any(counts > 1))
        if not self._chunked:
            self._stages = [(
                to_device(indptr, np.int32),
                to_device(weights.indices, np.int32),
                to_device(weights.data, dtype))]
            return
        nchunks = len(chunk_starts)
        last = ends[np.repeat(np.arange(self._nregions), counts)][-1]
        # Stage 1: the sums of the chunks; stage 2: the sums of the chunks
        # of each region
        self._stages = [
            (to_device(np.append(chunk_starts, last), np.int32),
             to_device(weights.indices, np.int32),
             to_device(weights.data, dtype)),
            (to_device(np.concatenate(([0], np.cumsum(counts))), np.int32),
             to_device(np.arange(nchunks), np.int32),
             to_device(np.ones(nchunks), dtype))]
        self._nchunks = nchunks
        self._partial = None

    def nregions(self) -> int:
        return self._nregions

    def evaluate(self, cube, out):
        """
        The sums of the regions in each channel of a cube (nz, ny, nx;
        ny * nx pixels of the weights), into out (nz, nregions).
        """
        if cube.shape[1] * cube.shape[2] != self._npix:
            raise RuntimeError(
                f"the regions have weights for {self._npix} pixels, but "
                f"the cube has {cube.shape[1] * cube.shape[2]} in each "
                f"channel")
        if not self._chunked:
            self._backend.regions_sum(*self._stages[0], cube, out)
            return
        if self._partial is None or self._partial.shape[0] != cube.shape[0]:
            self._partial = self._driver.mem_alloc_d(
                (cube.shape[0], 1, self._nchunks), self._dtype)
        self._backend.regions_sum(
            *self._stages[0], cube, self._partial[:, 0, :])
        self._backend.regions_sum(*self._stages[1], self._partial, out)
