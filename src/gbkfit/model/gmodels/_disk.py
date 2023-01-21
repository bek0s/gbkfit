
import abc
import logging
import typing

import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, miscutils, numutils
from . import traits


_log = logging.getLogger(__name__)


def _make_param_descs(key, nnodes, nw):
    return {key: ParamVectorDesc(key, nnodes) if nw else ParamScalarDesc(key)}


def _trait_param_info(traits_, prefix, nrnodes):
    params_list = []
    for i, trait in enumerate(traits_):
        params_sm = trait.params_sm()
        params_nw = trait.params_rnw(nrnodes)
        params_sm = [(pdesc, None, False) for pdesc in params_sm]
        params_nw = [(pdesc, nwmode, True) for pdesc, nwmode in params_nw]
        params = params_sm + params_nw
        params_list.append({tuple_[0].name(): tuple_ for tuple_ in params})
    params, mappings = miscutils.merge_dicts_and_make_mappings(
            params_list, prefix, True, False)
    pdescs = {name: tuple_[0] for name, tuple_ in params.items()}
    nwmodes = {name: tuple_[1] for name, tuple_ in params.items()}
    isnodewise = {name: tuple_[2] for name, tuple_ in params.items()}
    # The users of this function make the following assumptions about
    # the pdescs, nwmodes, and isnodewise dictionaries:
    # - their keys are in the same order
    # - their keys are in the same order with the associated traits
    # - for each trait, the nodewise come after the smooth parameters
    return pdescs, nwmodes, isnodewise, mappings


def _prepare_trait_arrays(
        driver,
        traits_, nnodes, nsubnodes,
        arr_uids,
        arr_cvalues, arr_ccounts,
        arr_pvalues, arr_pcounts,
        dtype):
    # Ignore unused traits
    if not traits_:
        return
    # Prepare trait data calculate their sizes
    uids = []
    cvalues = []
    ccounts = []
    pcounts = []
    for trait in traits_:
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


def _prepare_common_params_array(
        driver, params, arr, descs, nodes, subnodes, interp, isnw):
    if not descs:
        return
    start = 0
    for name, desc in descs.items():
        # Interpolate nodewise parameters
        # This will replace the per-node parameter values
        # with the interpolated per-subnode parameters values.
        if isnw:
            params[name] = interp(nodes, params[name])(subnodes)
            stop = start + len(subnodes)
        else:
            stop = start + desc.size()
        # Copy parameter values into the host memory buffer
        arr[0][start:stop] = params[name]
        start = stop
    # Transfer data from host to device
    driver.mem_copy_h2d(arr[0], arr[1])


def _prepare_traits_params_array(
        driver, params, arr, descs, nodes, subnodes, interp, isnw):
    if not descs:
        return
    start = 0
    for name, desc in descs.items():
        if isnw[name]:
            # Interpolate nodewise parameters
            # This will replace the per-node parameter values
            # with the interpolated per-subnode parameters values.
            params[name] = interp(nodes, params[name])(subnodes)
            stop = start + len(subnodes)
        else:
            stop = start + desc.size()
        # Copy parameter values into the host memory buffer
        arr[0][start:stop] = params[name]
        start = stop
    # Transfer data from host to device
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

        # Select the right prefix for the density traits
        rpt_prefix, rht_prefix = ('bpt', 'bht') \
            if isinstance(rptraits[0], traits.BPTrait) else ('opt', 'oht')

        # Make descs for trait disk parameters
        (self._rpt_pdescs,
         self._rpt_nwmodes,
         self._rpt_isnw,
         self._rpt_pnames) = _trait_param_info(rptraits, rpt_prefix, nrnodes)
        (self._rht_pdescs,
         self._rht_nwmodes,
         self._rht_isnw,
         self._rht_pnames) = _trait_param_info(rhtraits, rht_prefix, nrnodes)
        (self._vpt_pdescs,
         self._vpt_nwmodes,
         self._vpt_isnw,
         self._vpt_pnames) = _trait_param_info(vptraits, 'vpt', nrnodes)
        (self._vht_pdescs,
         self._vht_nwmodes,
         self._vht_isnw,
         self._vht_pnames) = _trait_param_info(vhtraits, 'vht', nrnodes)
        (self._dpt_pdescs,
         self._dpt_nwmodes,
         self._dpt_isnw,
         self._dpt_pnames) = _trait_param_info(dptraits, 'dpt', nrnodes)
        (self._dht_pdescs,
         self._dht_nwmodes,
         self._dht_isnw,
         self._dht_pnames) = _trait_param_info(dhtraits, 'dht', nrnodes)
        (self._zpt_pdescs,
         self._zpt_nwmodes,
         self._zpt_isnw,
         self._zpt_pnames) = _trait_param_info(zptraits, 'zpt', nrnodes)
        (self._spt_pdescs,
         self._spt_nwmodes,
         self._spt_isnw,
         self._spt_pnames) = _trait_param_info(sptraits, 'spt', nrnodes)
        (self._wpt_pdescs,
         self._wpt_nwmodes,
         self._wpt_isnw,
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

    def xpos_nwmode(self):
        return self._xpos_nwmode

    def ypos_nwmode(self):
        return self._ypos_nwmode

    def posa_nwmode(self):
        return self._posa_nwmode

    def incl_nwmode(self):
        return self._incl_nwmode

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
            self, driver, params,
            odata,
            image, scube, wdata, rdata, ordata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        if self._driver is not driver or self._dtype is not dtype:
            self._prepare(driver, dtype)

        #
        # Apply nodewise mode transform to parameters
        # TODO: revise the use of nested functions
        #

        def nwmode_transform_for_common_params(pdescs, nwmode):
            if nwmode is None:
                return
            for pdesc in pdescs:
                nwmode.transform(params[pdesc])

        def nwmode_transform_for_trait_params(pdescs, nwmodes):
            for pdesc in pdescs:
                if nwmodes[pdesc] is None:
                    continue
                nwmodes[pdesc].transform(params[pdesc])

        nwmode_transform_for_common_params(self._vsys_pdescs, self._vsys_nwmode)
        nwmode_transform_for_common_params(self._xpos_pdescs, self._xpos_nwmode)
        nwmode_transform_for_common_params(self._ypos_pdescs, self._ypos_nwmode)
        nwmode_transform_for_common_params(self._posa_pdescs, self._posa_nwmode)
        nwmode_transform_for_common_params(self._incl_pdescs, self._incl_nwmode)

        nwmode_transform_for_trait_params(self._rpt_pdescs, self._rpt_nwmodes)
        nwmode_transform_for_trait_params(self._rht_pdescs, self._rht_nwmodes)
        nwmode_transform_for_trait_params(self._vpt_pdescs, self._vpt_nwmodes)
        nwmode_transform_for_trait_params(self._vht_pdescs, self._vht_nwmodes)
        nwmode_transform_for_trait_params(self._dpt_pdescs, self._dpt_nwmodes)
        nwmode_transform_for_trait_params(self._dht_pdescs, self._dht_nwmodes)
        nwmode_transform_for_trait_params(self._zpt_pdescs, self._zpt_nwmodes)
        nwmode_transform_for_trait_params(self._spt_pdescs, self._spt_nwmodes)
        nwmode_transform_for_trait_params(self._wpt_pdescs, self._wpt_nwmodes)

        #
        # Prepare parameters
        # TODO: revise the use of nested functions
        #

        def prepare_common_params(arr, descs, isnw):
            _prepare_common_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, isnw)

        def prepare_traits_params(arr, descs, isnw):
            _prepare_traits_params_array(
                driver, params, arr, descs,
                self._rnodes, self._subrnodes, self._interp, isnw)

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
            self._s_rpt_pvalues, self._rpt_pdescs, self._rpt_isnw)
        prepare_traits_params(
            self._s_rht_pvalues, self._rht_pdescs, self._rht_isnw)
        prepare_traits_params(
            self._s_vpt_pvalues, self._vpt_pdescs, self._vpt_isnw)
        prepare_traits_params(
            self._s_vht_pvalues, self._vht_pdescs, self._vht_isnw)
        prepare_traits_params(
            self._s_dpt_pvalues, self._dpt_pdescs, self._dpt_isnw)
        prepare_traits_params(
            self._s_dht_pvalues, self._dht_pdescs, self._dht_isnw)
        prepare_traits_params(
            self._s_zpt_pvalues, self._zpt_pdescs, self._zpt_isnw)
        prepare_traits_params(
            self._s_spt_pvalues, self._spt_pdescs, self._spt_isnw)
        prepare_traits_params(
            self._s_wpt_pvalues, self._wpt_pdescs, self._wpt_isnw)

        wdata_cmp = None
        rdata_cmp = None
        vdata_cmp = None
        ddata_cmp = None
        ordata_cmp = None

        if out_extra is not None:
            shape = spat_size[::-1]
            if self._rptraits:
                rdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(rdata_cmp, 0)
            if self._vptraits:
                vdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdata_cmp, np.nan)
            if self._dptraits:
                ddata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ddata_cmp, np.nan)
            if self.wptraits():
                wdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(wdata_cmp, 1)
            if odata is not None:
                ordata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ordata_cmp, 0)

        self._impl_evaluate(
            driver, params,
            odata,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)

        if out_extra is not None:
            if self._rptraits:
                out_extra['rdata'] = driver.mem_copy_d2h(rdata_cmp)
            if self._vptraits:
                out_extra['vdata'] = driver.mem_copy_d2h(vdata_cmp)
            if self._dptraits:
                out_extra['ddata'] = driver.mem_copy_d2h(ddata_cmp)
            if self.wptraits():
                out_extra['wdata'] = driver.mem_copy_d2h(wdata_cmp)
            if odata is not None:
                out_extra['obdata'] = driver.mem_copy_d2h(ordata_cmp)
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
            odata,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        pass
