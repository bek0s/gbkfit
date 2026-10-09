import numpy as np
import scipy.special

from gbkfit.params.pdescs import ParamScalarDesc
from .core import ComponentPlan


__all__ = [
    'PointPlan',
    'point_pdescs'
]


def point_pdescs(spectral):
    """
    The parameters of a point component: its position (xpos, ypos) and
    flux, and for spectral ones its systemic velocity and dispersion.
    """
    names = ('xpos', 'ypos', 'flux') + (('vsys', 'disp') if spectral else ())
    return {name: ParamScalarDesc(name) for name in names}


def _pixels(params, grid):
    """
    The four pixels (x and y indices) around the position of a point on
    the sky, and the share of its light of each (bilinear); those outside
    the grid are left out.
    """
    rota = np.radians(grid['spat_rota'])
    x, y = params['xpos'], params['ypos']
    # From the sky to the pixels of the grid (rotated by rota)
    gx = x * np.cos(rota) + y * np.sin(rota)
    gy = -x * np.sin(rota) + y * np.cos(rota)
    px = (gx - grid['spat_zero'][0]) / grid['spat_step'][0]
    py = (gy - grid['spat_zero'][1]) / grid['spat_step'][1]
    x0, y0 = int(np.floor(px)), int(np.floor(py))
    fx, fy = px - x0, py - y0
    size_x, size_y = grid['spat_size'][:2]
    pixels = []
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            if 0 <= x0 + dx < size_x and 0 <= y0 + dy < size_y \
                    and wx * wy > 0:
                pixels.append((x0 + dx, y0 + dy, wx * wy))
    return pixels


class PointPlan(ComponentPlan):
    """
    The evaluation of a point component (e.g. an unresolved nucleus): its
    flux at its position on the sky, shared by the four pixels around it
    (bilinearly), as surface brightness (per unit area, as the disks), in
    the image, or in the spectral cube as its emission lines (see Lines;
    offset, scale and flux of each in lines), each a Gaussian of its
    dispersion, of which each channel gets its mean over the channel (as
    the disks do).
    """

    def __init__(self, driver, dtype, lines=None, ratios=()):
        """
        lines has the offset, scale and flux of each line (spectral
        points), and ratios the row and the name of the flux ratio of the
        lines after the first.
        """
        self._driver = driver
        self._dtype = np.dtype(dtype)
        self._lines = None if lines is None else np.array(lines, dtype=float)
        self._ratios = tuple(ratios)

    def _spectrum(self, params, grid):
        """The spectrum of the point, per unit flux."""
        size, step, zero = (
            grid['spec_size'], grid['spec_step'], grid['spec_zero'])
        edges = zero + (np.arange(size + 1) - 0.5) * step
        spectrum = np.zeros(size)
        for row, name in self._ratios:
            self._lines[row, 2] = params[name]
        for offset, scale, flux in self._lines:
            centre = offset + scale * params['vsys']
            sigma = scale * abs(params['disp'])
            if sigma > 0:
                cdf = 0.5 * (1 + scipy.special.erf(
                    (edges - centre) / (np.sqrt(2) * sigma)))
                spectrum += flux * np.diff(cdf) / step
            else:
                channel = int(np.floor((centre - edges[0]) / step))
                if 0 <= channel < size:
                    spectrum[channel] += flux / step
        return spectrum

    def evaluate(self, params, grid, outputs, out_extra):
        driver = self._driver
        image = outputs.get('image')
        scube = outputs.get('scube')
        # The surface brightness of all the flux in one pixel
        brightness = params['flux'] / (
            grid['spat_step'][0] * grid['spat_step'][1])
        values = None
        if scube is not None:
            values = brightness * self._spectrum(params, grid)
        for x, y, weight in _pixels(params, grid):
            if image is not None:
                pixel = image.reshape(-1, *image.shape[-2:])[0, y, x:x + 1]
                driver.math_add(pixel, self._dtype.type(
                    brightness * weight), out=pixel)
            if scube is not None:
                spectrum = scube[:, y, x]
                driver.math_add(spectrum, driver.mem_copy_h2d(
                    (weight * values).astype(self._dtype)), out=spectrum)
