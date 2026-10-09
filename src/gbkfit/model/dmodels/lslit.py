
from collections.abc import Sequence

import numpy as np

from gbkfit.dataset.datasets import DatasetLSlit
from gbkfit.model.core import DModel, GModelSCube
from gbkfit.psflsf import LSF, PSF, lsf_parser, psf_parser
from . import _dcube, _detail


__all__ = [
    'DModelLSlit'
]


class DModelLSlit(DModel):

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    _spectral_axis = 1

    @staticmethod
    def type():
        return 'lslit'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(
            cls, info, 2, True, True, dataset, DatasetLSlit)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            scale=self.scale(),
            slit_width=self.slit_width(),
            psf=psf_parser.dump(self.psf()),
            lsf=lsf_parser.dump(self.lsf()),
            smooth_weights=self._dcube.smooth_weights(),
            mask_cutoff=self._dcube.mask_cutoff(),
            mask_apply=self._dcube.mask_apply(),
            dtype=self.dtype().name)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[int | float] = (1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0),
            rota: int | float = 0,
            scale: Sequence[int] = (1, 1),
            slit_width: int | float | None = None,
            psf: PSF | None = None,
            lsf: LSF | None = None,
            smooth_weights: bool = False,
            mask_cutoff: int | float | None = None,
            mask_apply: bool = False,
            dtype: str = 'float32'
    ):
        """
        A long-slit spectrum: a position-velocity image along a slit.

        The size, step, rpix, rval and scale are given for the position
        along the slit and the spectral axis. The slit runs along the x
        axis of the sky, rotated by rota, through the reference position.
        It is modelled as a cube one pixel wide along the y axis, whose
        step is the slit width (by default, the step along the slit).
        """
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        if slit_width is None:
            slit_width = step[0]
        self._dcube = _dcube.DCube(
            (size[0], 1, size[1]),
            (step[0], slit_width, step[1]),
            (rpix[0], 0, rpix[1]),
            (rval[0], 0, rval[1]),
            rota,
            (scale[0], scale[0], scale[1]),
            psf, lsf, smooth_weights, mask_cutoff, mask_apply,
            np.dtype(dtype))

    def keys(self):
        return ['lslit']

    def size(self):
        return _without_width(self._dcube.size())

    def step(self):
        return _without_width(self._dcube.step())

    def zero(self):
        return _without_width(self._dcube.zero())

    def rpix(self):
        return _without_width(self._dcube.rpix())

    def rval(self):
        return _without_width(self._dcube.rval())

    def rota(self):
        return self._dcube.rota()

    def scale(self):
        return _without_width(self._dcube.scale())

    def slit_width(self):
        return self._dcube.step()[1]

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def _prepare_impl(self, gmodel):
        self._dcube.prepare(self._driver, gmodel.has_weights())

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        has_mcube = dcube.mcube() is not None
        has_wcube = dcube.wcube() is not None
        # The gmodel adds to the data cube, so clear it
        driver.mem_fill(dcube.scratch_dcube(), 0)
        # Evaluate gmodel on DCube's arrays
        gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_dcube(),
            dcube.scratch_wcube(),
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            dcube.rota(),
            dcube.dtype(),
            out_gmodel_extra)
        # Evaluate DCube (perform convolution, supersampling, etc)
        dcube.evaluate(out_dmodel_extra)
        # Model evaluation complete.
        # Return data, mask, and weight arrays (if available),
        # with shape (spectral, position)
        return dict(lslit=dict(
            d=dcube.dcube()[:, 0, :],
            m=dcube.mcube()[:, 0, :] if has_mcube else None,
            w=dcube.wcube()[:, 0, :] if has_wcube else None))


def _without_width(values):
    """The (position, spectral) values of a (position, width, spectral)."""
    return values[0], values[2]
