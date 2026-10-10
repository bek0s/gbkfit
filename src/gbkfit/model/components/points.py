from collections.abc import Sequence

import numpy as np
import scipy.special

from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import parseutils
from ._detail import dump_lines, dump_name
from .base import (
    BrightnessComponent2D, BrightnessComponent3D, ComponentPlan,
    SpectralComponent2D, SpectralComponent3D)
from .lines import Line, Lines, line_parser


__all__ = [
    'BrightnessPoint2D',
    'BrightnessPoint3D',
    'SpectralPoint2D',
    'SpectralPoint3D'
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


class BrightnessPoint2D(BrightnessComponent2D):
    """
    An unresolved component (e.g. a nucleus): its flux at a point (see
    PointPlan). Its parameters: xpos, ypos and flux.
    """

    @staticmethod
    def type():
        return 'point'

    def dump(self):
        return dict(type=self.type()) | dump_name(self)

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def pdescs(self):
        return point_pdescs(spectral=False)

    def plan(self, driver, spectral, dtype, lines):
        return PointPlan(driver, dtype)


class BrightnessPoint3D(BrightnessComponent3D):
    """
    An unresolved component (e.g. a nucleus): its flux at a point (see
    PointPlan). Its parameters: xpos, ypos and flux.
    It is not absorbed by the opacity of the model.
    """

    @staticmethod
    def type():
        return 'point'

    def dump(self):
        return dict(type=self.type()) | dump_name(self)

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def pdescs(self):
        return point_pdescs(spectral=False)

    def plan(self, driver, spectral, dtype, lines):
        return PointPlan(driver, dtype)


class SpectralPoint2D(SpectralComponent2D):
    """
    An unresolved component (e.g. a broad central component): its flux at
    a point, as its emission lines (see PointPlan). Its parameters:
    xpos, ypos, flux, vsys and disp (the dispersion of its lines), and
    the flux ratios of its lines (see Lines).
    """

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(line_parser, info, 'lines')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self):
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_lines(self._lines))

    def __init__(
            self,
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        super().__init__(name)
        self._lines = Lines(lines)
        pdescs = point_pdescs(spectral=True)
        if repeated := sorted(set(self._lines.pdescs()) & set(pdescs)):
            raise RuntimeError(
                f"the parameters of the lines have the names of other "
                f"parameters: {repeated}; rename the lines")
        self._pdescs = pdescs | self._lines.pdescs()

    def pdescs(self):
        return self._pdescs

    def line_names(self):
        return self._lines.names()

    def plan(self, driver, spectral, dtype, lines):
        selected = self._lines.select(lines)
        values = self._lines.values(spectral)[list(selected)]
        ratios = [
            (row, self._lines.ratio_name(index))
            for row, index in enumerate(selected)
            if self._lines.ratio_name(index) is not None]
        return PointPlan(driver, dtype, values, ratios)


class SpectralPoint3D(SpectralComponent3D):
    """
    An unresolved component (e.g. a broad central component): its flux at
    a point, as its emission lines (see PointPlan). Its parameters:
    xpos, ypos, flux, vsys and disp (the dispersion of its lines), and
    the flux ratios of its lines (see Lines).
    It is not absorbed by the opacity of the model.
    """

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(line_parser, info, 'lines')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self):
        return (
            dict(type=self.type())
            | dump_name(self)
            | dump_lines(self._lines))

    def __init__(
            self,
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        super().__init__(name)
        self._lines = Lines(lines)
        pdescs = point_pdescs(spectral=True)
        if repeated := sorted(set(self._lines.pdescs()) & set(pdescs)):
            raise RuntimeError(
                f"the parameters of the lines have the names of other "
                f"parameters: {repeated}; rename the lines")
        self._pdescs = pdescs | self._lines.pdescs()

    def pdescs(self):
        return self._pdescs

    def line_names(self):
        return self._lines.names()

    def plan(self, driver, spectral, dtype, lines):
        selected = self._lines.select(lines)
        values = self._lines.values(spectral)[list(selected)]
        ratios = [
            (row, self._lines.ratio_name(index))
            for row, index in enumerate(selected)
            if self._lines.ratio_name(index) is not None]
        return PointPlan(driver, dtype, values, ratios)
