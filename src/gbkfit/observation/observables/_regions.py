import numpy as np
import scipy.sparse


__all__ = [
    'RegionSumsPlan'
]


class RegionSumsPlan:
    """
    The weighted sums of the pixels of regions (see Regions.weights) in
    each channel of cubes, on a driver.
    """

    def __init__(self, weights: scipy.sparse.csr_array, driver, dtype):
        weights = scipy.sparse.csr_array(weights)
        weights.sum_duplicates()
        self._nregions = weights.shape[0]
        self._npix = weights.shape[1]
        self._indptr = driver.mem_copy_h2d(weights.indptr.astype(np.int32))
        self._indices = driver.mem_copy_h2d(weights.indices.astype(np.int32))
        self._weights = driver.mem_copy_h2d(weights.data.astype(dtype))
        self._backend = driver.native_class('DModel', dtype)()

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
        self._backend.regions_sum(
            self._indptr, self._indices, self._weights, cube, out)
