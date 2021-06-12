
import abc

import numpy as np

from gbkfit.params.descs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, miscutils


def _make_param_descs(key, nnodes, nw):
    return {key: ParamVectorDesc(key, nnodes) if nw else ParamScalarDesc(key)}


def _trait_param_info(traits, prefix, nrnodes):
    pdescs_list = []
    for i, trait in enumerate(traits):
        pdescs = trait.params_sm() + trait.params_rnw(nrnodes)
        pdescs_list.append({pdesc.name(): pdesc for pdesc in pdescs})
    pdescs, mappings = miscutils.merge_dicts_and_make_mappings(
        pdescs_list, prefix, True, False)
    return pdescs, mappings


def _prepare_node_array(driver, nodes_lst, nodes_arr, dtype):
    nodes_arr[:] = driver.mem_alloc_s(len(nodes_lst), dtype)
    nodes_arr[0][:] = nodes_lst
    driver.mem_copy_h2d(nodes_arr[0], nodes_arr[1])


def _prepare_trait_arrays(
        driver,
        traits, nnodes, nsubnodes,
        ary_uids,
        ary_ccounts, ary_pcounts,
        ary_cvalues, ary_pvalues,
        dtype):
    if not traits:
        return
    uids = []
    ccounts = []
    pcounts = []
    cvalues = []
    for trait in traits:
        consts = trait.consts()
        params_sm = trait.params_sm()
        params_nw = trait.params_rnw(nnodes)
        pcounts_sm = sum(p.size() for p in params_sm)
        pcounts_nw = len(params_nw) * nsubnodes
        uids.append(trait.uid())
        cvalues += consts
        ccounts += [len(consts)]
        pcounts += [pcounts_sm + pcounts_nw]
    ary_uids[:] = driver.mem_alloc_s(len(uids), np.int32)
    ary_ccounts[:] = driver.mem_alloc_s(len(ccounts), np.int32)
    ary_pcounts[:] = driver.mem_alloc_s(len(pcounts), np.int32)
    ary_cvalues[:] = driver.mem_alloc_s(len(cvalues), dtype)
    ary_pvalues[:] = driver.mem_alloc_s(sum(pcounts), dtype)
    ary_uids[0][:] = uids
    ary_ccounts[0][:] = ccounts
    ary_pcounts[0][:] = pcounts
    ary_cvalues[0][:] = cvalues
    driver.mem_copy_h2d(ary_uids[0], ary_uids[1])
    driver.mem_copy_h2d(ary_ccounts[0], ary_ccounts[1])
    driver.mem_copy_h2d(ary_pcounts[0], ary_pcounts[1])
    driver.mem_copy_h2d(ary_cvalues[0], ary_cvalues[1])


def _prepare_common_params_array(
        driver, params, arr, descs, nodes, subnodes, interp, nodewise):
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
    if not descs:
        return
    start = 0
    for trait, mapping in zip(traits, mappings):
        for name in trait.params_sm():
            new_name = mapping[name.name()]
            stop = start + descs[new_name].size()
            arr[0][start:stop] = params[new_name]
            start = stop
        for name in trait.params_rnw(len(nodes)):
            new_name = mapping[name.name()]
            stop = start + len(subnodes)
            params[new_name] = interp(nodes, params[new_name])(subnodes)
            arr[0][start:stop] = params[new_name]
            start = stop
    driver.mem_copy_h2d(arr[0], arr[1])


class Disk(abc.ABC):

    def __init__(
            self,
            loose, tilted, rnodes, rnstep, interp,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            zptraits,
            sptraits,
            wptraits):

        rnodes = tuple(rnodes)
        nrnodes = len(rnodes)

        self._loose = loose
        self._tilted = tilted
        self._rnodes = rnodes
        self._nrnodes = nrnodes
        self._rnstep = rnstep

        # Inner radial sub nodes
        # The end node will not be exact, but it will be good enough
        # The smaller the rnstep the better
        subrnodes = np.arange(rnodes[0] + rnstep / 2, rnodes[-1], rnstep)
        # Outer radial sub nodes
        subrnodes = np.insert(subrnodes, 0, rnodes[0])
        subrnodes = np.append(subrnodes, subrnodes[-1] + rnstep / 2)
        subrnodes = subrnodes.tolist()

        self._subrnodes = tuple(subrnodes)
        self._nsubrnodes = len(subrnodes)
        self._interp = interp

        self._rptraits = tuple(rptraits)
        self._rhtraits = tuple(rhtraits)
        self._vptraits = tuple(vptraits)
        self._vhtraits = tuple(vhtraits)
        self._dptraits = tuple(dptraits)
        self._dhtraits = tuple(dhtraits)
        self._zptraits = tuple(zptraits)
        self._sptraits = tuple(sptraits)
        self._wptraits = tuple(wptraits)

        self._vsys_pdescs = _make_param_descs('vsys', nrnodes, loose) \
            if self._vptraits else {}
        self._xpos_pdescs = _make_param_descs('xpos', nrnodes, loose)
        self._ypos_pdescs = _make_param_descs('ypos', nrnodes, loose)
        self._posa_pdescs = _make_param_descs('posa', nrnodes, tilted)
        self._incl_pdescs = _make_param_descs('incl', nrnodes, tilted)

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
         self._zpt_pnames) = _trait_param_info(zptraits, 'zpt_', nrnodes)
        (self._spt_pdescs,
         self._spt_pnames) = _trait_param_info(sptraits, 'spt', nrnodes)
        (self._wpt_pdescs,
         self._wpt_pnames) = _trait_param_info(wptraits, 'wpt', nrnodes)

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

        self._s_subrnodes = [None, None]
        self._s_vsys_pvalues = [None, None]
        self._s_xpos_pvalues = [None, None]
        self._s_ypos_pvalues = [None, None]
        self._s_posa_pvalues = [None, None]
        self._s_incl_pvalues = [None, None]

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

        self._disk = None
        self._dtype = None
        self._driver = None

    def loose(self):
        return self._loose

    def tilted(self):
        return self._tilted

    def rnodes(self):
        return self._rnodes

    def rnstep(self):
        return self._rnstep

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

        self._backend = driver
        self._dtype = dtype

        lcount = self._nsubrnodes if self._loose else 1
        tcount = self._nsubrnodes if self._tilted else 1

        self._s_vsys_pvalues = driver.mem_alloc_s(lcount, dtype) \
            if self._vptraits else [None, None]
        self._s_xpos_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._s_ypos_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._s_posa_pvalues = driver.mem_alloc_s(tcount, dtype)
        self._s_incl_pvalues = driver.mem_alloc_s(tcount, dtype)

        _prepare_node_array(driver, self._subrnodes, self._s_subrnodes, dtype)

        _prepare_trait_arrays(
            driver,
            self._rptraits, self._nrnodes, self._nsubrnodes,
            self._s_rpt_uids,
            self._s_rpt_ccounts, self._s_rpt_pcounts,
            self._s_rpt_cvalues, self._s_rpt_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._rhtraits, self._nrnodes, self._nsubrnodes,
            self._s_rht_uids,
            self._s_rht_ccounts, self._s_rht_pcounts,
            self._s_rht_cvalues, self._s_rht_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._vptraits, self._nrnodes, self._nsubrnodes,
            self._s_vpt_uids,
            self._s_vpt_ccounts, self._s_vpt_pcounts,
            self._s_vpt_cvalues, self._s_vpt_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._vhtraits, self._nrnodes, self._nsubrnodes,
            self._s_vht_uids,
            self._s_vht_ccounts, self._s_vht_pcounts,
            self._s_vht_cvalues, self._s_vht_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._dptraits, self._nrnodes, self._nsubrnodes,
            self._s_dpt_uids,
            self._s_dpt_ccounts, self._s_dpt_pcounts,
            self._s_dpt_cvalues, self._s_dpt_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._dhtraits, self._nrnodes, self._nsubrnodes,
            self._s_dht_uids,
            self._s_dht_ccounts, self._s_dht_pcounts,
            self._s_dht_cvalues, self._s_dht_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._zptraits, self._nrnodes, self._nsubrnodes,
            self._s_zpt_uids,
            self._s_zpt_ccounts, self._s_zpt_pcounts,
            self._s_zpt_cvalues, self._s_zpt_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._sptraits, self._nrnodes, self._nsubrnodes,
            self._s_spt_uids,
            self._s_spt_ccounts, self._s_spt_pcounts,
            self._s_spt_cvalues, self._s_spt_pvalues,
            dtype)
        _prepare_trait_arrays(
            driver,
            self._wptraits, self._nrnodes, self._nsubrnodes,
            self._s_wpt_uids,
            self._s_wpt_ccounts, self._s_wpt_pcounts,
            self._s_wpt_cvalues, self._s_wpt_pvalues,
            dtype)

        self._impl_prepare(driver, dtype)

    def evaluate(
            self, driver, params, image, scube, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        if self._driver is not driver or self._dtype is not dtype:
            self._prepare(driver, dtype)

        def prepare_common_params(arr, descs, nodewise):
            _prepare_common_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, nodewise)

        def prepare_traits_params(arr, descs, mappings, traits):
            _prepare_traits_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, mappings, traits)

        prepare_common_params(
            self._s_vsys_pvalues, self._vsys_pdescs, self._loose)
        prepare_common_params(
            self._s_xpos_pvalues, self._xpos_pdescs, self._loose)
        prepare_common_params(
            self._s_ypos_pvalues, self._ypos_pdescs, self._loose)
        prepare_common_params(
            self._s_posa_pvalues, self._posa_pdescs, self._tilted)
        prepare_common_params(
            self._s_incl_pvalues, self._incl_pdescs, self._tilted)

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

        self._impl_evaluate(
            driver, params, image, scube, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)

    @abc.abstractmethod
    def _impl_prepare(self, driver, dtype):
        pass

    @abc.abstractmethod
    def _impl_evaluate(
            self, driver, params, image, scube, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass
