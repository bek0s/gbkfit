
import abc
import itertools
import logging
import typing

import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, miscutils, numutils
from .traits import *


_log = logging.getLogger(__name__)


def _make_param_descs(key, nnodes, nw):
    return {key: ParamVectorDesc(key, nnodes) if nw else ParamScalarDesc(key)}


def _trait_param_info(traits, prefix, nrnodes):
    pdescs_list = []
    for i, trait in enumerate(traits):
        pdescs_sm = trait.params_sm()
        pdescs_nw = tuple([pdesc for pdesc, _ in trait.params_rnw(nrnodes)])
        pdescs = pdescs_sm + pdescs_nw
        pdescs_list.append({pdesc.name(): pdesc for pdesc in pdescs})
    pdescs, mappings = miscutils.merge_dicts_and_make_mappings(
        pdescs_list, prefix, True, False)
    return pdescs, mappings


def _prepare_trait_arrays(
        driver,
        traits, nnodes, nsubnodes,
        arr_uids,
        arr_cvalues, arr_ccounts,
        arr_pvalues, arr_pcounts,
        dtype):
    # Ignore unused traits
    if not traits:
        return
    # Prepare trait data calculate their sizes
    uids = []
    cvalues = []
    ccounts = []
    pcounts = []
    for trait in traits:
        consts = trait.consts()
        params_sm = trait.params_sm()
        params_nw = trait.params_rnw(nnodes)
        pcounts_sm = sum(p.size() for p in params_sm)
        # Each subring will have its own parameter values
        # This is because of the interpolation we perform
        pcounts_nw = len(params_nw) * nsubnodes
        uids.append(trait.uid())
        cvalues += consts
        ccounts += [len(consts)]
        pcounts += [pcounts_sm + pcounts_nw]
    # Allocate memory for trait data and assign constants
    arr_uids[:] = driver.mem_alloc_s(len(uids), np.int32)
    arr_cvalues[:] = driver.mem_alloc_s(len(cvalues), dtype)
    arr_ccounts[:] = driver.mem_alloc_s(len(ccounts), np.int32)
    arr_pvalues[:] = driver.mem_alloc_s(sum(pcounts), dtype)
    arr_pcounts[:] = driver.mem_alloc_s(len(pcounts), np.int32)
    arr_uids[0][:] = uids
    arr_cvalues[0][:] = cvalues
    arr_ccounts[0][:] = ccounts
    arr_pcounts[0][:] = pcounts
    driver.mem_copy_h2d(arr_uids[0], arr_uids[1])
    driver.mem_copy_h2d(arr_cvalues[0], arr_cvalues[1])
    driver.mem_copy_h2d(arr_ccounts[0], arr_ccounts[1])
    driver.mem_copy_h2d(arr_pcounts[0], arr_pcounts[1])


def _nwmode_transform_for_param(param, nwmode):
    match nwmode['type']:
        case 'absolute':
            pass
        case 'relative1':
            origin_idx = nwmode.get('origin', 0)
            origin_val = param[origin_idx]
            param += param[origin_idx]
            param[origin_idx] = origin_val
            pass
        case 'relative2':
            origin_idx = nwmode.get('origin', 0)
            numutils.cumsum(param, origin_idx)
        case _:
            assert False, "impossible"


def _nwmode_transform_for_common_params(params, descs, nwmode):
    for name, desc in descs.items():
        _nwmode_transform_for_param(params[name], nwmode)


def _nwmode_transform_for_trait_params(params, descs, traits, nrnodes):

    for trait in traits:

        # if not isinstance(trait, TraitNWModeSupport):
        #     continue

        print("BEGIN")

        for desc in trait.params_rnw(nrnodes):
            #_nwmode_transform_for_param(params, trait.nwmode())
            print(desc)
        print("END")



def _prepare_common_params_array(
        driver, params, arr, descs, nodes, subnodes, interp, nodewise, nwmode):
    if not descs:
        return
    start = 0
    for name, desc in descs.items():
        if nodewise:
            stop = start + len(subnodes)
            params[name] = interp(nodes, params[name])(subnodes)
            arr[0][start:stop] = params[name]
        else:
            stop = start + desc.size()
            arr[0][start:stop] = params[name]
        start = stop
    driver.mem_copy_h2d(arr[0], arr[1])


def _prepare_traits_params_array(
        driver, params, arr, descs, nodes, subnodes, interp, mappings, traits):
    # print(params)
    if not descs:
        return
    start = 0
    for trait, mapping in zip(traits, mappings):
        print(descs)
        print(mapping)
        for name in trait.params_sm():
            new_name = mapping[name.name()]
            stop = start + descs[new_name].size()
            arr[0][start:stop] = params[new_name]
            start = stop
        for desc, nwmode in trait.params_rnw(len(nodes)):
            new_name = mapping[desc.name()]
            print(desc)
            print(new_name)
            stop = start + len(subnodes)
            params[new_name] = interp(nodes, params[new_name])(subnodes)
            arr[0][start:stop] = params[new_name]
            start = stop
    driver.mem_copy_h2d(arr[0], arr[1])


class Disk(abc.ABC):

    def __init__(
            self,
            loose, tilted, rnodes, rstep, interp,
            vsys_nwmode,
            xpos_nwmode, ypos_nwmode,
            posa_nwmode, incl_nwmode,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            zptraits,
            sptraits,
            wptraits):

        nrnodes = len(rnodes)

        #
        # Calculate radial sub nodes
        # The end node will not be exact, but it will be good enough
        # The smaller the rstep the better
        #
        # Inner radial sub nodes
        subrnodes = np.arange(rnodes[0] + rstep / 2, rnodes[-1], rstep)
        # Outer radial sub nodes
        subrnodes = np.insert(subrnodes, 0, rnodes[0])
        subrnodes = np.append(subrnodes, subrnodes[-1] + rstep / 2)
        # Make some IDEs shut up
        subrnodes = tuple(typing.cast(list, subrnodes.tolist()))

        self._loose = loose
        self._tilted = tilted
        self._rnodes = rnodes
        self._nrnodes = nrnodes
        self._rstep = rstep
        self._subrnodes = subrnodes
        self._nsubrnodes = len(subrnodes)
        self._interp = interp
        self._vsys_nwmode = vsys_nwmode
        self._xpos_nwmode = xpos_nwmode
        self._ypos_nwmode = ypos_nwmode
        self._posa_nwmode = posa_nwmode
        self._incl_nwmode = incl_nwmode
        self._rptraits = rptraits
        self._rhtraits = rhtraits
        self._vptraits = vptraits
        self._vhtraits = vhtraits
        self._dptraits = dptraits
        self._dhtraits = dhtraits
        self._zptraits = zptraits
        self._sptraits = sptraits
        self._wptraits = wptraits

        # Make descs for common disk parameters
        self._vsys_pdescs = _make_param_descs('vsys', nrnodes, loose) \
            if self._vptraits else {}
        self._xpos_pdescs = _make_param_descs('xpos', nrnodes, loose)
        self._ypos_pdescs = _make_param_descs('ypos', nrnodes, loose)
        self._posa_pdescs = _make_param_descs('posa', nrnodes, tilted)
        self._incl_pdescs = _make_param_descs('incl', nrnodes, tilted)

        # Make descs for trait disk parameters
        (self._rpt_pdescs,
         self._rpt_pnames) = _trait_param_info(rptraits, 'rpt', nrnodes)
        (self._rht_pdescs,
         self._rht_pnames) = _trait_param_info(rhtraits, 'rht', nrnodes)
        (self._vpt_pdescs,
         self._vpt_pnames) = _trait_param_info(vptraits, 'vpt', nrnodes)
        (self._vht_pdescs,
         self._vht_pnames) = _trait_param_info(vhtraits, 'vht', nrnodes)
        (self._dpt_pdescs,
         self._dpt_pnames) = _trait_param_info(dptraits, 'dpt', nrnodes)
        (self._dht_pdescs,
         self._dht_pnames) = _trait_param_info(dhtraits, 'dht', nrnodes)
        (self._zpt_pdescs,
         self._zpt_pnames) = _trait_param_info(zptraits, 'zpt', nrnodes)
        (self._spt_pdescs,
         self._spt_pnames) = _trait_param_info(sptraits, 'spt', nrnodes)
        (self._wpt_pdescs,
         self._wpt_pnames) = _trait_param_info(wptraits, 'wpt', nrnodes)

        # Merge all parameter descs into the same dictionary
        self._pdescs = {
            **self._vsys_pdescs,
            **self._xpos_pdescs,
            **self._ypos_pdescs,
            **self._posa_pdescs,
            **self._incl_pdescs,
            **self._rpt_pdescs,
            **self._rht_pdescs,
            **self._vpt_pdescs,
            **self._vht_pdescs,
            **self._dpt_pdescs,
            **self._dht_pdescs,
            **self._zpt_pdescs,
            **self._spt_pdescs,
            **self._wpt_pdescs}

        # This is where we store subrnodes and common parameter values
        # Each variable will contain host and device memory
        self._s_subrnodes = [None, None]
        self._s_vsys_pvalues = [None, None]
        self._s_xpos_pvalues = [None, None]
        self._s_ypos_pvalues = [None, None]
        self._s_posa_pvalues = [None, None]
        self._s_incl_pvalues = [None, None]

        # This is where we store the trait information
        # Each variable will contain host and device memory
        (self._s_rpt_uids,
         self._s_rpt_cvalues,
         self._s_rpt_pvalues,
         self._s_rpt_ccounts,
         self._s_rpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_rht_uids,
         self._s_rht_cvalues,
         self._s_rht_pvalues,
         self._s_rht_ccounts,
         self._s_rht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_vpt_uids,
         self._s_vpt_cvalues,
         self._s_vpt_pvalues,
         self._s_vpt_ccounts,
         self._s_vpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_vht_uids,
         self._s_vht_cvalues,
         self._s_vht_pvalues,
         self._s_vht_ccounts,
         self._s_vht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_dpt_uids,
         self._s_dpt_cvalues,
         self._s_dpt_pvalues,
         self._s_dpt_ccounts,
         self._s_dpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_dht_uids,
         self._s_dht_cvalues,
         self._s_dht_pvalues,
         self._s_dht_ccounts,
         self._s_dht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_zpt_uids,
         self._s_zpt_cvalues,
         self._s_zpt_pvalues,
         self._s_zpt_ccounts,
         self._s_zpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_spt_uids,
         self._s_spt_cvalues,
         self._s_spt_pvalues,
         self._s_spt_ccounts,
         self._s_spt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._s_wpt_uids,
         self._s_wpt_cvalues,
         self._s_wpt_pvalues,
         self._s_wpt_ccounts,
         self._s_wpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)

        self._spat_size = None
        self._spat_step = None
        self._spat_zero = None
        self._spec_size = None
        self._spec_step = None
        self._spec_zero = None

        self._dtype = None
        self._driver = None
        self._backend = None

    def loose(self):
        return self._loose

    def tilted(self):
        return self._tilted

    def rnodes(self):
        return self._rnodes

    def rstep(self):
        return self._rstep

    def interp(self):
        return self._interp

    def rptraits(self):
        return self._rptraits

    def rhtraits(self):
        return self._rhtraits

    def vptraits(self):
        return self._vptraits

    def vhtraits(self):
        return self._vhtraits

    def dptraits(self):
        return self._dptraits

    def dhtraits(self):
        return self._dhtraits

    def zptraits(self):
        return self._zptraits

    def sptraits(self):
        return self._sptraits

    def wptraits(self):
        return self._wptraits

    def params(self):
        return self._pdescs

    def _prepare(self, driver, dtype):
        # Allocate memory for parameter values on host and device
        loose_param_count = self._nsubrnodes if self._loose else 1
        tilted_param_count = self._nsubrnodes if self._tilted else 1
        self._s_vsys_pvalues = driver.mem_alloc_s(loose_param_count, dtype) \
            if self._vptraits else [None, None]
        self._s_xpos_pvalues = driver.mem_alloc_s(loose_param_count, dtype)
        self._s_ypos_pvalues = driver.mem_alloc_s(loose_param_count, dtype)
        self._s_posa_pvalues = driver.mem_alloc_s(tilted_param_count, dtype)
        self._s_incl_pvalues = driver.mem_alloc_s(tilted_param_count, dtype)

        # Allocate memory and copy the sub rnode data into it
        self._s_subrnodes = driver.mem_alloc_s(self._nsubrnodes, dtype)
        self._s_subrnodes[0][:] = self._subrnodes
        driver.mem_copy_h2d(self._s_subrnodes[0], self._s_subrnodes[1])

        # Prepare trait data memory
        # This includes trait memory allocation and initialization
        _prepare_trait_arrays(
            driver,
            self._rptraits, self._nrnodes, self._nsubrnodes,
            self._s_rpt_uids,
            self._s_rpt_cvalues, self._s_rpt_ccounts,
            self._s_rpt_pvalues, self._s_rpt_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._rhtraits, self._nrnodes, self._nsubrnodes,
            self._s_rht_uids,
            self._s_rht_cvalues, self._s_rht_ccounts,
            self._s_rht_pvalues, self._s_rht_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._vptraits, self._nrnodes, self._nsubrnodes,
            self._s_vpt_uids,
            self._s_vpt_cvalues, self._s_vpt_ccounts,
            self._s_vpt_pvalues, self._s_vpt_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._vhtraits, self._nrnodes, self._nsubrnodes,
            self._s_vht_uids,
            self._s_vht_cvalues, self._s_vht_ccounts,
            self._s_vht_pvalues, self._s_vht_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._dptraits, self._nrnodes, self._nsubrnodes,
            self._s_dpt_uids,
            self._s_dpt_cvalues, self._s_dpt_ccounts,
            self._s_dpt_pvalues, self._s_dpt_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._dhtraits, self._nrnodes, self._nsubrnodes,
            self._s_dht_uids,
            self._s_dht_cvalues, self._s_dht_ccounts,
            self._s_dht_pvalues, self._s_dht_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._zptraits, self._nrnodes, self._nsubrnodes,
            self._s_zpt_uids,
            self._s_zpt_cvalues, self._s_zpt_ccounts,
            self._s_zpt_pvalues, self._s_zpt_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._sptraits, self._nrnodes, self._nsubrnodes,
            self._s_spt_uids,
            self._s_spt_cvalues, self._s_spt_ccounts,
            self._s_spt_pvalues, self._s_spt_pcounts,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._wptraits, self._nrnodes, self._nsubrnodes,
            self._s_wpt_uids,
            self._s_wpt_cvalues, self._s_wpt_ccounts,
            self._s_wpt_pvalues, self._s_wpt_pcounts,
            dtype)

        self._dtype = dtype
        self._driver = driver
        self._backend = driver.backends().gmodel(dtype)

        # Perform preparation specific to the derived class
        self._impl_prepare(driver, dtype)

    def evaluate(
            self, driver, params, image, scube, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        if self._driver is not driver or self._dtype is not dtype:
            self._prepare(driver, dtype)

        def prepare_common_params(arr, descs, nodewise, nwmode):
            _prepare_common_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, nodewise, nwmode)

        def prepare_traits_params(arr, descs, mappings, traits):

            _nwmode_transform_for_trait_params(params, descs, traits, self._nrnodes)

            _prepare_traits_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, mappings, traits)

        prepare_common_params(
            self._s_vsys_pvalues, self._vsys_pdescs, self._loose, self._vsys_nwmode)
        prepare_common_params(
            self._s_xpos_pvalues, self._xpos_pdescs, self._loose, self._xpos_nwmode)
        prepare_common_params(
            self._s_ypos_pvalues, self._ypos_pdescs, self._loose, self._ypos_nwmode)
        prepare_common_params(
            self._s_posa_pvalues, self._posa_pdescs, self._tilted, self._posa_nwmode)
        prepare_common_params(
            self._s_incl_pvalues, self._incl_pdescs, self._tilted, self._incl_nwmode)

        prepare_traits_params(
            self._s_rpt_pvalues, self._rpt_pdescs, self._rpt_pnames,
            self._rptraits)
        prepare_traits_params(
            self._s_rht_pvalues, self._rht_pdescs, self._rht_pnames,
            self._rhtraits)
        prepare_traits_params(
            self._s_vpt_pvalues, self._vpt_pdescs, self._vpt_pnames,
            self._vptraits)
        prepare_traits_params(
            self._s_vht_pvalues, self._vht_pdescs, self._vht_pnames,
            self._vhtraits)
        prepare_traits_params(
            self._s_dpt_pvalues, self._dpt_pdescs, self._dpt_pnames,
            self._dptraits)
        prepare_traits_params(
            self._s_dht_pvalues, self._dht_pdescs, self._dht_pnames,
            self._dhtraits)
        prepare_traits_params(
            self._s_zpt_pvalues, self._zpt_pdescs, self._zpt_pnames,
            self._zptraits)
        prepare_traits_params(
            self._s_spt_pvalues, self._spt_pdescs, self._spt_pnames,
            self._sptraits)
        prepare_traits_params(
            self._s_wpt_pvalues, self._wpt_pdescs, self._wpt_pnames,
            self._wptraits)

        rdata = None
        vdata = None
        ddata = None

        if out_extra is not None:
            shape = spat_size[::-1]
            if self._rptraits:
                rdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(rdata, 0)
            if self._vptraits:
                vdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdata, np.nan)
            if self._dptraits:
                ddata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ddata, np.nan)

        self._impl_evaluate(
            driver, params, image, scube, rcube, wcube,
            rdata, vdata, ddata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)

        if out_extra is not None:
            if self._rptraits:
                out_extra['rdata'] = driver.mem_copy_d2h(rdata)
            if self._vptraits:
                out_extra['vdata'] = driver.mem_copy_d2h(vdata)
            if self._dptraits:
                out_extra['ddata'] = driver.mem_copy_d2h(ddata)
            if self._rptraits:
                sumabs = np.nansum(np.abs(out_extra['rdata']))
                _log.debug(f"sum(abs(rdata)): {sumabs}")
            if self._vptraits:
                sumabs = np.nansum(np.abs(out_extra['vdata']))
                _log.debug(f"sum(abs(vdata)): {sumabs}")
            if self._dptraits:
                sumabs = np.nansum(np.abs(out_extra['ddata']))
                _log.debug(f"sum(abs(ddata)): {sumabs}")

    @abc.abstractmethod
    def _impl_prepare(self, driver, dtype):
        pass

    @abc.abstractmethod
    def _impl_evaluate(
            self, driver, params,
            image, scube, rcube, wcube, rdata, vdata, ddata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass
