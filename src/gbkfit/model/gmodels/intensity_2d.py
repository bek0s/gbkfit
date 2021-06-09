
from gbkfit.model.core import GModelImage
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import DensityComponent2D
from .density_smdisk_2d import DensitySMDisk2D


__all__ = ['GModelIntensity2D']


_dcmp_parser = parseutils.TypedParser(DensityComponent2D)


def _register_components():
    _dcmp_parser.register(DensitySMDisk2D)


_register_components()


class GModelIntensity2D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_2d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_dcmp_parser.load(info['components']))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_dcmp_parser.dump(self._cmps))

    def __init__(self, components):
        self._cmps = iterutils.tuplify(components)
        self._driver = None
        (self._params,
         self._mappings) = _detail.make_gmodel_2d_params(self._cmps)

    def params(self):
        return self._params

    def _prepare(self, driver):
        self._driver = driver

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota, dtype,
            out_extra):

        if self._driver is not driver:
            self._prepare(driver)

        spat_size = size
        spat_step = step
        spat_zero = zero
        spat_rota = rota

        # Evaluate components
        for i, (cmp, mapping) in enumerate(zip(self._cmps, self._mappings)):
            cparams = {p: params[mapping[p]] for p in cmp.params()}
            cmp_out_extra = {} if out_extra is not None else None
            cmp.evaluate(
                driver, cparams, image, weights,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, cmp_out_extra)
            if out_extra is not None:
                for k, v in cmp_out_extra.items():
                    out_extra[f'component{i}_{k}'] = v
