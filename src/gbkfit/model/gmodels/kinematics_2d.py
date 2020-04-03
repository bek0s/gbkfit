
import gbkfit.model.gmodel
from gbkfit.utils import iterutils
from . import _common


class GModelKinematics2D(
    gbkfit.model.gmodel.GModelSCubeSupport, gbkfit.model.gmodel.GModel):

    @staticmethod
    def type():
        return 'kinematics_2d'

    @classmethod
    def load(cls, info):
        cmp = _common.spectral_component_2d_parser.load(info['components'])
        return cls(cmp)

    def dump(self):
        return {
            'type': self.type(),
            'components': _common.spectral_component_2d_parser.dump(
                self._components)}

    def __init__(self, components):
        self._components = iterutils.tuplify(components)
        self._driver = None
        (self._params,
         self._mappings) = _common.make_model_2d_params(self._components)

    def params(self):
        return self._params

    def _prepare(self, driver):
        self._driver = driver

    def evaluate_scube(
            self, driver, params, scube, dtype, size, step, zero, out_extra):

        # Perform preparations if needed
        if self._driver is not driver:
            self._prepare(driver)

        # Shortcuts for clarity
        spat_size = size[:2]
        spat_step = step[:2]
        spat_zero = zero[:2]
        spec_size = size[2]
        spec_step = step[2]
        spec_zero = zero[2]

        # Evaluate components
        for i, (component, mapping) in enumerate(
                zip(self._components, self._mappings)):
            cparams = {p: params[mapping[p]] for p in component.params()}
            cmp_out_extra = {} if out_extra is not None else None
            component.evaluate(
                driver, cparams, scube, dtype,
                spat_size, spat_step, spat_zero,
                spec_size, spec_step, spec_zero,
                cmp_out_extra)
            if out_extra is not None:
                for key, value in cmp_out_extra.items():
                    out_extra[f'component{i}_{key}'] = value
