
from gbkfit.model.core import GModelSCube
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import SpectralComponent2D
from .spectral_smdisk_2d import SpectralSMDisk2D


__all__ = ['GModelKinematics2D']


_scmp_parser = parseutils.TypedParser(SpectralComponent2D)


def _register_components():
    _scmp_parser.register(SpectralSMDisk2D)


_register_components()


class GModelKinematics2D(GModelSCube):

    @staticmethod
    def type():
        return 'kinematics_2d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_scmp_parser.load(info['components']))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            components=_scmp_parser.dump(self._cmps))

    def __init__(self, components):
        self._cmps = iterutils.tuplify(components)
        self._driver = None
        (self._params,
         self._mappings) = _detail.make_gmodel_2d_params(self._cmps)

    def params(self):
        return self._params

    def _prepare(self, driver):
        self._driver = driver

    def evaluate_scube(
            self, driver, params, scube, weights, size, step, zero, rota, dtype,
            out_extra):

        if self._driver is not driver:
            self._prepare(driver)

        spat_size = size[:2]
        spat_step = step[:2]
        spat_zero = zero[:2]
        spec_size = size[2]
        spec_step = step[2]
        spec_zero = zero[2]
        spat_rota = rota

        # Evaluate components
        for i, (cmp, mapping) in enumerate(zip(self._cmps, self._mappings)):
            cparams = {p: params[mapping[p]] for p in cmp.params()}
            cmp_out_extra = {} if out_extra is not None else None
            cmp.evaluate(
                driver, cparams, scube, weights,
                spat_size, spat_step, spat_zero, spat_rota,
                spec_size, spec_step, spec_zero,
                dtype, cmp_out_extra)
            if out_extra is not None:
                for k, v in cmp_out_extra.items():
                    out_extra[f'component{i}_{k}'] = v
