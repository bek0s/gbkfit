
from gbkfit.model import GModelImage
from gbkfit.utils import iterutils, parseutils
from . import _detail, component_d3d_parser


__all__ = ['GModelIntensity3D']


class GModelIntensity3D(GModelImage):

    @staticmethod
    def type():
        return 'intensity_3d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=component_d3d_parser.load(opts['components']),
            tcomponents=component_d3d_parser.load(opts.get('tcomponents')))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            tauto=self._tauto,
            size_z=self._size_z,
            step_z=self._step_z,
            components=component_d3d_parser.dump(self._cmps),
            tcomponents=component_d3d_parser.dump(self._tcmps))

    def __init__(
            self, components, size_z=None, step_z=None,
            tauto=False, tcomponents=None):
        if tcomponents is None:
            tcomponents = ()
        self._cmps = iterutils.tuplify(components)
        self._tcmps = iterutils.tuplify(tcomponents)
        self._tauto = tauto
        self._tcube = None
        self._dtype = None
        self._driver = None
        self._size_z = size_z
        self._step_z = step_z
        self._spat_size = None
        self._spat_step = None
        self._spat_zero = None
        (self._params,
         self._mappings,
         self._tmappings) = _detail.make_gmodel_3d_params(
            self._cmps, self._tcmps, self._tauto)

    def params(self):
        return self._params

    def _prepare(self, driver, size, step, zero, dtype):
        self._driver = driver
        self._dtype = dtype
        spat_zidx = 0 if size[0] > size[1] else 1
        spat_size_z = self._size_z if self._size_z else size[spat_zidx]
        spat_step_z = self._step_z if self._step_z else step[spat_zidx]
        spat_zero_z = -(spat_size_z / 2 - 0.5) * spat_step_z
        self._spat_size = size + (spat_size_z,)
        self._spat_step = step + (spat_step_z,)
        self._spat_zero = zero + (spat_zero_z,)
        if self._tauto or self._tcmps:
            self._tcube = driver.mem_alloc_d(self._spat_size[::-1], dtype)

    def evaluate_image(
            self, driver, params, image, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._spat_size[:2] != size
                or self._spat_step[:2] != step
                or self._spat_zero[:2] != zero
                or self._dtype != dtype):
            self._prepare(driver, size, step, zero, dtype)

        tcube = self._tcube
        spat_size = self._spat_size
        spat_step = self._spat_step
        spat_zero = self._spat_zero
        spat_rota = rota

        # Evaluate auto transparency (if requested)
        if self._tauto:
            for i, (cmp, mapping) in enumerate(zip(self._cmps, self._mappings)):
                cparams = {p: params[mapping[p]] for p in cmp.params()}
                cmp_out_extra = {} if out_extra is not None else None
                cmp.evaluate(
                    driver, cparams, None, tcube,
                    spat_size, spat_step, spat_zero, spat_rota,
                    dtype, cmp_out_extra)
            driver.math_mul(tcube, params['tauto'], out=tcube)
            if out_extra is not None:
                out_extra['transparency_auto'] = \
                    driver.mem_copy_d2h(tcube).copy()

        # Evaluate transparency components (if provided)
        for i, (cmp, mapping) in enumerate(zip(self._tcmps, self._mappings)):
            cparams = {p: params[mapping[p]] for p in cmp.params()}
            cmp_out_extra = {} if out_extra is not None else None
            cmp.evaluate(
                driver, cparams, None, tcube,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, cmp_out_extra)
            if out_extra is not None:
                for k, v in cmp_out_extra.items():
                    out_extra[f'transparency_component{i}_{k}'] = v

        # Store total transparency (if requested)
        if out_extra is not None and tcube is not None:
            out_extra['transparency_full'] = driver.mem_copy_d2h(tcube).copy()

        # Evaluate components
        for i, (cmp, mapping) in enumerate(zip(self._cmps, self._mappings)):
            cparams = {p: params[mapping[p]] for p in cmp.params()}
            cmp_out_extra = {} if out_extra is not None else None
            cmp.evaluate(
                driver, cparams, image, tcube,
                spat_size, spat_step, spat_zero, spat_rota,
                dtype, cmp_out_extra)
            if out_extra is not None:
                for k, v in cmp_out_extra.items():
                    out_extra[f'component{i}_{k}'] = v
