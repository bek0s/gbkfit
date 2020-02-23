
import gbkfit.gmodel
from gbkfit.utils import iterutils
from . import _common


class GModelIntensity3D(
    gbkfit.gmodel.GModelImageSupport, gbkfit.gmodel.GModel):

    @staticmethod
    def type():
        return 'intensity_3d'

    @classmethod
    def load(cls, info):
        tauto = info.get('tauto')
        size_z = info.get('size_z')
        step_z = info.get('step_z')
        cmp = _common.density_component_3d_parser.load(info['components'])
        tcmp = _common.density_component_3d_parser.load(info.get('tcomponents'))
        return cls(cmp, size_z, step_z, tauto, tcmp)

    def dump(self):
        return {
            'type': self.type(),
            'tauto': self._tauto,
            'size_z': self._size_z,
            'step_z': self._step_z,
            'components': _common.density_component_3d_parser.dump(
                self._components),
            'tcomponents': _common.density_component_3d_parser.dump(
                self._tcomponents)}

    def __init__(
            self, components, size_z=None, step_z=None,
            tauto=None, tcomponents=None):
        if tcomponents is None:
            tcomponents = ()
        self._components = iterutils.tuplify(components)
        self._tcomponents = iterutils.tuplify(tcomponents)
        self._tauto = bool(tauto)
        self._tcube = None
        self._size_z = size_z
        self._step_z = step_z
        self._dtype = None
        self._driver = None
        self._spat_size = None
        self._spat_step = None
        self._spat_zero = None
        (self._params,
         self._mappings,
         self._tmappings) = _common.make_model_3d_params(
            self._components, self._tcomponents, False)

    def params(self):
        return self._params

    def _prepare(self, driver, dtype, size, step, zero):
        self._driver = driver
        self._dtype = dtype
        spat_zidx = 0 if size[0] > size[1] else 1
        spat_size_z = self._size_z if self._size_z else size[spat_zidx]
        spat_step_z = self._step_z if self._step_z else step[spat_zidx]
        spat_zero_z = -(spat_size_z / 2 - 0.5) * spat_step_z
        self._spat_size = size + (spat_size_z,)
        self._spat_step = step + (spat_step_z,)
        self._spat_zero = zero + (spat_zero_z,)
        if self._tauto or self._tcomponents:
            self._tcube = driver.mem_alloc_d(self._spat_size, dtype)

    def evaluate_image(
            self, driver, params, image, dtype, size, step, zero, out_extra):

        # Perform preparations if needed
        if (self._driver is not driver
                or self._dtype != dtype
                or self._spat_size is None
                or self._spat_step is None
                or self._spat_zero is None
                or self._spat_size[:2] != size[:2]
                or self._spat_step[:2] != step[:2]
                or self._spat_zero[:2] != zero[:2]):
            self._prepare(driver, dtype, size, step, zero)

        # Shortcuts for clarity
        spat_size = self._spat_size
        spat_step = self._spat_step
        spat_zero = self._spat_zero

        # Evaluate auto transparency, if requested
        if self._tauto:
            for i, (component, mapping) in enumerate(
                   zip(self._components, self._mappings)):
                cparams = {p: params[mapping[p]] for p in component.params()}
                cmp_out_extra = {} if out_extra is not None else None
                component.evaluate(
                    driver, cparams, None, self._tcube, dtype,
                    spat_size, spat_step, spat_zero,
                    cmp_out_extra)
            driver.array_mul(self._tcube, params['tauto'])
            if out_extra is not None:
                out_extra[f'transparency_auto'] = driver.mem_copy_d2h(
                        self._tcube).copy()

        # Evaluate transparency components
        for i, (component, mapping) in enumerate(
                zip(self._tcomponents, self._tmappings)):
            cparams = {p: params[mapping[p]] for p in component.params()}
            cmp_out_extra = {} if out_extra is not None else None
            component.evaluate(
                driver, cparams, None, self._tcube, dtype,
                spat_size, spat_step, spat_zero,
                cmp_out_extra)
            if out_extra is not None:
                for key, value in cmp_out_extra.items():
                    out_extra[f'transparency_component{i}_{key}'] = value

        # Store total transparency (if requested)
        if out_extra is not None and self._tcube is not None:
            out_extra['transparency'] = driver.mem_copy_d2h(self._tcube).copy()

        # Evaluate components
        for i, (component, mapping) in enumerate(
                zip(self._components, self._mappings)):
            cparams = {p: params[mapping[p]] for p in component.params()}
            cmp_out_extra = {} if out_extra is not None else None
            component.evaluate(
                driver, cparams, image, None, dtype,
                spat_size, spat_step, spat_zero,
                cmp_out_extra)
            if out_extra is not None:
                for key, value in cmp_out_extra.items():
                    out_extra[f'component{i}_{key}'] = value
