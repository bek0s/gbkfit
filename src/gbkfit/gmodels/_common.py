
import abc

import numpy as np

from gbkfit.params import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, parseutils
from . import traits


class DensityComponent2D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, image, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        pass


class DensityComponent3D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, image, rcube, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        pass


class SpectralComponent2D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, scube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass


class SpectralComponent3D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass


density_component_2d_parser = parseutils.TypedParser(DensityComponent2D)
density_component_3d_parser = parseutils.TypedParser(DensityComponent3D)
spectral_component_2d_parser = parseutils.TypedParser(SpectralComponent2D)
spectral_component_3d_parser = parseutils.TypedParser(SpectralComponent3D)


def _make_component_params(components, prefix, force_prefix):
    params = {}
    mappings = []
    for i, cmp in enumerate(components):
        prefix = f'{prefix}{i}_' * (i > 0 or force_prefix)
        params.update({prefix + k: v for k, v in cmp.params().items()})
        mappings.append(dict(zip(cmp.params().keys(), params.keys())))
    return params, mappings


def make_model_2d_params(components):
    return _make_component_params(components, 'cmp', False)


def make_model_3d_params(components, tcomponents, tauto):
    params, mappings = _make_component_params(components, 'cmp', False)
    tparams, tmappings = _make_component_params(tcomponents, 'tcmp', True)
    return {**params, **tparams}, mappings, tmappings


def trait_info(traitslist, prefix, nnodes=None):
    uids = []
    cvalues = []
    ccounts = []
    pcounts = []
    pdescs = {}
    ponames = []
    pnnames = []
    for i, trait in enumerate(traitslist):
        is_piecewise = isinstance(trait, traits.ParamSupportNW)
        if is_piecewise and not nnodes:
            raise RuntimeError()
        prefix += str(i) * (i > 0)
        consts = trait.consts()
        params = trait.params(nnodes) if is_piecewise else trait.params()
        uids += [trait.uid()]
        cvalues += consts
        ccounts += [len(consts)]
        pcounts += [sum(p.size() for p in params)]
        foo = {f'{prefix}_{p.name()}': p for p in params}
        pdescs.update(foo)
        ponames.append([p.name() for p in params])
        pnnames.append(list(foo.keys()))
    return uids, cvalues, ccounts, pcounts, pdescs, ponames, pnnames


def make_param_descs(key, nnodes, pw):
    return {key: ParamVectorDesc(key, nnodes) if pw else ParamScalarDesc(key)}


def prepare_param_array(backend, values, array, descs):
    start = 0
    for key, desc in descs.items():
        stop = start + desc.size()
        array[0][start:stop] = values[key]
        start = stop
    backend.mem_copy_h2d(array[0], array[1])


def prepare_rnode_array(backend, dtype, array_rnodes, rnodes):
    array_rnodes[:] = backend.mem_alloc(len(rnodes), dtype)
    array_rnodes[0][:] = rnodes
    backend.mem_copy_h2d(array_rnodes[0], array_rnodes[1])


def prepare_trait_arrays(
        backend, dtype,
        array_uids, uids,
        array_ccounts, ccounts,
        array_pcounts, pcounts,
        array_cvalues, cvalues,
        array_pvalues):
    array_uids[:] = backend.mem_alloc(len(uids), np.int32)
    array_ccounts[:] = backend.mem_alloc(len(ccounts), np.int32)
    array_pcounts[:] = backend.mem_alloc(len(pcounts), np.int32)
    array_cvalues[:] = backend.mem_alloc(len(cvalues), dtype)
    array_pvalues[:] = backend.mem_alloc(sum(pcounts), dtype)
    array_uids[0][:] = uids
    array_ccounts[0][:] = ccounts
    array_pcounts[0][:] = pcounts
    array_cvalues[0][:] = cvalues
    backend.mem_copy_h2d(array_uids[0], array_uids[1])
    backend.mem_copy_h2d(array_ccounts[0], array_ccounts[1])
    backend.mem_copy_h2d(array_pcounts[0], array_pcounts[1])
    backend.mem_copy_h2d(array_cvalues[0], array_cvalues[1])


def parse_density_disk_2d_common_args(
        loose, tilted, rnodes,
        rptraits,
        sptraits):

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")

    # For convenience, convert all traits to tuples
    rptraits = iterutils.tuplify(rptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits,
        sptraits=sptraits)


def parse_density_disk_3d_common_args(
        loose, tilted, rnodes,
        rptraits, rhtraits,
        wptraits,
        sptraits):

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not rhtraits:
        raise RuntimeError("at least one rhtrait is required")

    # For convenience, convert all traits to tuples
    rptraits = iterutils.tuplify(rptraits)
    rhtraits = iterutils.tuplify(rhtraits)
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    # The number of polar and height traits must match
    rptraits_len = len(rptraits)
    rhtraits_len = len(rhtraits)
    if rptraits_len != rhtraits_len:
        raise RuntimeError(
            f"the number of rhtraits must be equal to "
            f"the number of rptraits ({rhtraits_len} != {rptraits_len})")

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits, rhtraits=rhtraits,
        wptraits=wptraits,
        sptraits=sptraits)


def parse_spectral_disk_2d_common_args(
        loose, tilted, rnodes,
        rptraits,
        vptraits,
        dptraits,
        sptraits):

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")

    # For convenience, convert all traits to tuples
    rptraits = iterutils.tuplify(rptraits)
    vptraits = iterutils.tuplify(vptraits)
    dptraits = iterutils.tuplify(dptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits,
        vptraits=vptraits,
        dptraits=dptraits,
        sptraits=sptraits)


def parse_spectral_disk_3d_common_args(
        loose, tilted, rnodes,
        rptraits, rhtraits,
        vptraits, vhtraits,
        dptraits, dhtraits,
        wptraits,
        sptraits):

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not rhtraits:
        raise RuntimeError("at least one rhtrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")

    # For convenience, convert all traits to tuples.
    rptraits = iterutils.tuplify(rptraits)
    rhtraits = iterutils.tuplify(rhtraits)
    vptraits = iterutils.tuplify(vptraits)
    vhtraits = iterutils.tuplify(vhtraits)
    dptraits = iterutils.tuplify(dptraits)
    dhtraits = iterutils.tuplify(dhtraits)
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    # ...
    if None in vhtraits:
        vhtraits = iterutils.replace_items_and_copy(
            vhtraits, None, traits.VHTraitOne())
    if None in dhtraits:
        dhtraits = iterutils.replace_items_and_copy(
            dhtraits, None, traits.DHTraitOne())

    # The number of polar and height traits must match
    rptraits_len = len(rptraits)
    rhtraits_len = len(rhtraits)
    vptraits_len = len(vptraits)
    vhtraits_len = len(vhtraits)
    dptraits_len = len(dptraits)
    dhtraits_len = len(dhtraits)
    if rptraits_len != rhtraits_len:
        raise RuntimeError(
            f"the number of rhtraits must be equal to "
            f"the number of rptraits ({rhtraits_len} != {rptraits_len})")
    if vptraits_len != vhtraits_len:
        raise RuntimeError(
            f"the number of vhtraits must be equal to "
            f"the number of vptraits ({vhtraits_len} != {vptraits_len})")
    if len(dptraits) != len(dhtraits):
        raise RuntimeError(
            f"the number of dhtraits must be equal to "
            f"the number of dptraits ({dhtraits_len} != {dptraits_len})")

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits, rhtraits=rhtraits,
        vptraits=vptraits, vhtraits=vhtraits,
        dptraits=dptraits, dhtraits=dhtraits,
        wptraits=wptraits,
        sptraits=sptraits)
