
from gbkfit.model.core import GModelSCube
from gbkfit.utils import iterutils, parseutils
from . import _detail
from .core import DensityComponent3D, SpectralComponent3D
from .density_mcdisk_3d import DensityMCDisk3D
from .density_smdisk_3d import DensitySMDisk3D
from .spectral_mcdisk_3d import SpectralMCDisk3D
from .spectral_smdisk_3d import SpectralSMDisk3D


__all__ = ['GModelKinematics3D']


_dcmp_parser = parseutils.TypedParser(DensityComponent3D)
_scmp_parser = parseutils.TypedParser(SpectralComponent3D)
_dcmp_parser.register(DensityMCDisk3D)
_dcmp_parser.register(DensitySMDisk3D)
_scmp_parser.register(SpectralMCDisk3D)
_scmp_parser.register(SpectralSMDisk3D)


class GModelKinematics3D(GModelSCube):

    @staticmethod
    def type():
        return 'kinematics_3d'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(
            components=_scmp_parser.load(info['components']),
            tcomponents=_dcmp_parser.load(info.get('tcomponents')))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            tauto=self._tauto,
            size_z=self._size[2],
            step_z=self._step[2],
            components=_scmp_parser.dump(self._cmps),
            tcomponents=_dcmp_parser.dump(self._tcmps))

    def __init__(
            self, components, size_z=None, step_z=None,
            tauto=False, tcomponents=None):
        if tcomponents is None:
            tcomponents = ()
        self._cmps = iterutils.tuplify(components)
        self._tcmps = iterutils.tuplify(tcomponents)
        self._tauto = tauto
        self._size = [None, None, size_z]
        self._step = [None, None, step_z]
        self._zero = [None, None, None]
        self._tcube = None
        self._wcube = None
        self._dtype = None
        self._driver = None
        self._backend = None
        (self._params,
         self._mappings,
         self._tmappings) = _detail.make_gmodel_3d_params(
            self._cmps, self._tcmps, tauto)

    def params(self):
        return self._params

    def is_weighted(self):
        return True

    def _prepare(self, driver, wdata, size, step, zero, dtype):
        self._driver = driver
        self._dtype = dtype
        self._size[:2] = size
        self._step[:2] = step
        self._zero[:2] = zero
        spat_zidx = 0 if size[0] > size[1] else 1
        self._size[2] = self._size[2] if self._size[2] else size[spat_zidx]
        self._step[2] = self._step[2] if self._step[2] else step[spat_zidx]
        self._zero[2] = -(self._size[2] / 2 - 0.5) * self._step[2]
        if self._tauto or self._tcmps:
            self._tcube = driver.mem_alloc_d(self._size[::-1], dtype)
        if wdata is not None:
            self._wcube = driver.mem_alloc_d(self._size[::-1], dtype)
        self._backend = driver.backends().gmodel(dtype)

    def evaluate_scube(
            self, driver, params, scube, wdata, size, step, zero, rota, dtype,
            out_extra):

        if (self._driver is not driver
                or self._size[:2] != size[:2]
                or self._step[:2] != step[:2]
                or self._zero[:2] != zero[:2]
                or self._dtype != dtype):
            self._prepare(driver, wdata, size[:2], step[:2], zero[:2], dtype)

        spat_size = self._size
        spat_step = self._step
        spat_zero = self._zero
        spat_rota = rota
        spec_size = size[2]
        spec_step = step[2]
        spec_zero = zero[2]
        tcube = self._tcube
        wcube = self._wcube

        # Evaluate auto transparency (if requested)
        if self._tauto:
            for i, (cmp, mapping) in enumerate(zip(self._cmps, self._mappings)):
                cparams = {p: params[mapping[p]] for p in cmp.params()}
                cmp_out_extra = {} if out_extra is not None else None
                cmp.evaluate(
                    driver, cparams, None, tcube, None,
                    spat_size, spat_step, spat_zero, spat_rota,
                    spec_size, spec_step, spec_zero,
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
                driver, cparams, None, tcube, None,
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
                driver, cparams, scube, tcube, wcube,
                spat_size, spat_step, spat_zero, spat_rota,
                spec_size, spec_step, spec_zero,
                dtype, cmp_out_extra)
            if out_extra is not None:
                for k, v in cmp_out_extra.items():
                    out_extra[f'component{i}_{k}'] = v

        # Generate weight cube
        if wcube is not None:
            self._backend.make_wcube(spat_size, spec_size, wcube, wdata)
