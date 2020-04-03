
import abc

from gbkfit.math import interpolation
from gbkfit.utils import iterutils
from . import _common


_INTERPOLATIONS = {
    'linear': interpolation.InterpolatorLinear,
    'akima': interpolation.InterpolatorAkima,
    'pchip': interpolation.InterpolatorPCHIP
}


class Disk(abc.ABC):

    def __init__(
            self,
            loose, tilted, rnodes,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits):

        interp = 'linear'
        subrsep = 1.0

        if rnodes is None or len(rnodes) < 2:
            raise RuntimeError(
                "At least two radial nodes must be provided")
        if not iterutils.is_ascending(rnodes):
            raise RuntimeError(
                "Radial nodes must be ascending")
        if not iterutils.all_positive(rnodes):
            raise RuntimeError(
                "Radial nodes must be positive")
        if not iterutils.all_unique(rnodes):
            raise RuntimeError(
                "Radial nodes must be unique")
        if subrsep > (rnodes[-1] - rnodes[0]):
            raise RuntimeError(
                "The distance between sub radial nodes must be less or equal "
                "than the distance between the first and last radial nodes")
        if interp not in _INTERPOLATIONS:
            raise RuntimeError(
                "Interpolation type must be one of the following: "
                f"{list(_INTERPOLATIONS.keys())}")

        rnodes = tuple(rnodes)
        nrnodes = len(rnodes)

        self._loose = bool(loose)
        self._tilted = bool(tilted)
        self._rnodes = rnodes
        self._nrnodes = nrnodes

        # Calculate radial subnodes
        subrsep2 = subrsep / 2
        subrnodes = []
        rcur = rnodes[0] + subrsep2
        while rcur + subrsep2 <= rnodes[-1]:
            subrnodes.append(rcur)
            rcur += subrsep
        subrnodes.insert(0, subrnodes[0] - subrsep2)
        subrnodes.append(subrnodes[-1] + subrsep2)

        self._subrsep = subrsep
        self._subrnodes = subrnodes
        self._nsubrnodes = len(subrnodes)
        self._interp_cls = _INTERPOLATIONS[interp]

        self._rptraits = tuple(rptraits)
        self._rhtraits = tuple(rhtraits)
        self._vptraits = tuple(vptraits)
        self._vhtraits = tuple(vhtraits)
        self._dptraits = tuple(dptraits)
        self._dhtraits = tuple(dhtraits)
        self._wptraits = tuple(wptraits)
        self._sptraits = tuple(sptraits)

        self._vsys_pdescs = _common.make_param_descs('vsys', nrnodes, loose) \
            if self._vptraits else {}
        self._xpos_pdescs = _common.make_param_descs('xpos', nrnodes, loose)
        self._ypos_pdescs = _common.make_param_descs('ypos', nrnodes, loose)
        self._posa_pdescs = _common.make_param_descs('posa', nrnodes, tilted)
        self._incl_pdescs = _common.make_param_descs('incl', nrnodes, tilted)

        (self._rpt_pdescs,
         self._rpt_pnames) = _common.trait_param_info(rptraits, 'rpt', nrnodes)
        (self._rht_pdescs,
         self._rht_pnames) = _common.trait_param_info(rhtraits, 'rht', nrnodes)
        (self._vpt_pdescs,
         self._vpt_pnames) = _common.trait_param_info(vptraits, 'vpt', nrnodes)
        (self._vht_pdescs,
         self._vht_pnames) = _common.trait_param_info(vhtraits, 'vht', nrnodes)
        (self._dpt_pdescs,
         self._dpt_pnames) = _common.trait_param_info(dptraits, 'dpt', nrnodes)
        (self._dht_pdescs,
         self._dht_pnames) = _common.trait_param_info(dhtraits, 'dht', nrnodes)
        (self._wpt_pdescs,
         self._wpt_pnames) = _common.trait_param_info(wptraits, 'wpt', nrnodes)
        (self._spt_pdescs,
         self._spt_pnames) = _common.trait_param_info(sptraits, 'spt', nrnodes)

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
            **self._wpt_pdescs,
            **self._spt_pdescs}

        self._m_subrnodes = [None, None]
        self._m_vsys_pvalues = [None, None]
        self._m_xpos_pvalues = [None, None]
        self._m_ypos_pvalues = [None, None]
        self._m_posa_pvalues = [None, None]
        self._m_incl_pvalues = [None, None]

        (self._m_rpt_uids,
         self._m_rpt_cvalues,
         self._m_rpt_pvalues,
         self._m_rpt_ccounts,
         self._m_rpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_rht_uids,
         self._m_rht_cvalues,
         self._m_rht_pvalues,
         self._m_rht_ccounts,
         self._m_rht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_vpt_uids,
         self._m_vpt_cvalues,
         self._m_vpt_pvalues,
         self._m_vpt_ccounts,
         self._m_vpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_vht_uids,
         self._m_vht_cvalues,
         self._m_vht_pvalues,
         self._m_vht_ccounts,
         self._m_vht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_dpt_uids,
         self._m_dpt_cvalues,
         self._m_dpt_pvalues,
         self._m_dpt_ccounts,
         self._m_dpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_dht_uids,
         self._m_dht_cvalues,
         self._m_dht_pvalues,
         self._m_dht_ccounts,
         self._m_dht_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_wpt_uids,
         self._m_wpt_cvalues,
         self._m_wpt_pvalues,
         self._m_wpt_ccounts,
         self._m_wpt_pcounts) = iterutils.make_tuple((5,), [None, None], True)
        (self._m_spt_uids,
         self._m_spt_cvalues,
         self._m_spt_pvalues,
         self._m_spt_ccounts,
         self._m_spt_pcounts) = iterutils.make_tuple((5,), [None, None], True)

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

    def wptraits(self):
        return self._wptraits

    def sptraits(self):
        return self._sptraits

    def params(self):
        return self._pdescs

    def _prepare(self, driver, dtype):

        self._backend = driver
        self._dtype = dtype

        lcount = self._nsubrnodes if self._loose else 1
        tcount = self._nsubrnodes if self._tilted else 1

        if self._vptraits:
            self._m_vsys_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._m_vsys_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._m_xpos_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._m_ypos_pvalues = driver.mem_alloc_s(lcount, dtype)
        self._m_posa_pvalues = driver.mem_alloc_s(tcount, dtype)
        self._m_incl_pvalues = driver.mem_alloc_s(tcount, dtype)

        _common.prepare_rnode_array(
            driver, dtype, self._m_subrnodes, self._subrnodes)
        # TODO self._nrnodes, self._nsubrnodes,
        if self._rptraits:
            _common.prepare_trait_arrays(
                self._rptraits, self._nrnodes, self._nsubrnodes,
                self._m_rpt_uids,
                self._m_rpt_ccounts, self._m_rpt_pcounts,
                self._m_rpt_cvalues, self._m_rpt_pvalues,
                dtype, driver)
        if self._rhtraits:
            _common.prepare_trait_arrays(
                self._rhtraits, self._nrnodes, self._nsubrnodes,
                self._m_rht_uids,
                self._m_rht_ccounts, self._m_rht_pcounts,
                self._m_rht_cvalues, self._m_rht_pvalues,
                dtype, driver)
        if self._vptraits:
            _common.prepare_trait_arrays(
                self._vptraits, self._nrnodes, self._nsubrnodes,
                self._m_vpt_uids,
                self._m_vpt_ccounts, self._m_vpt_pcounts,
                self._m_vpt_cvalues, self._m_vpt_pvalues,
                dtype, driver)
        if self._vhtraits:
            _common.prepare_trait_arrays(
                self._vhtraits, self._nrnodes, self._nsubrnodes,
                self._m_vht_uids,
                self._m_vht_ccounts, self._m_vht_pcounts,
                self._m_vht_cvalues, self._m_vht_pvalues,
                dtype, driver)
        if self._dptraits:
            _common.prepare_trait_arrays(
                self._dptraits, self._nrnodes, self._nsubrnodes,
                self._m_dpt_uids,
                self._m_dpt_ccounts, self._m_dpt_pcounts,
                self._m_dpt_cvalues, self._m_dpt_pvalues,
                dtype, driver)
        if self._dhtraits:
            _common.prepare_trait_arrays(
                self._dhtraits, self._nrnodes, self._nsubrnodes,
                self._m_dht_uids,
                self._m_dht_ccounts, self._m_dht_pcounts,
                self._m_dht_cvalues, self._m_dht_pvalues,
                dtype, driver)
        if self._wptraits:
            _common.prepare_trait_arrays(
                self._wptraits, self._nrnodes, self._nsubrnodes,
                self._m_wpt_uids,
                self._m_wpt_ccounts, self._m_wpt_pcounts,
                self._m_wpt_cvalues, self._m_wpt_pvalues,
                dtype, driver)
        if self._sptraits:
            _common.prepare_trait_arrays(
                self._sptraits, self._nrnodes, self._nsubrnodes,
                self._m_spt_uids,
                self._m_spt_ccounts, self._m_spt_pcounts,
                self._m_spt_cvalues, self._m_spt_pvalues,
                dtype, driver)

        self._impl_prepare(driver, dtype)

    def evaluate(
            self, driver, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):

        # Perform preparations if needed
        if self._driver is not driver or self._dtype is not dtype:
            self._prepare(driver, dtype)

        def prepare_common_params(ary, descs, nodewise):
            _common.prepare_common_params_array(
                driver, params, ary, descs,
                self._rnodes, self._subrnodes, self._interp_cls, nodewise)

        def prepare_traits_params(ary, descs, mappings, traits):
            _common.prepare_traits_params_array(
                driver, params, ary, descs,
                self._rnodes, self._subrnodes, self._interp_cls, mappings, traits)

        prepare_common_params(self._m_vsys_pvalues, self._vsys_pdescs, self._loose)
        prepare_common_params(self._m_xpos_pvalues, self._xpos_pdescs, self._loose)
        prepare_common_params(self._m_ypos_pvalues, self._ypos_pdescs, self._loose)
        prepare_common_params(self._m_posa_pvalues, self._posa_pdescs, self._tilted)
        prepare_common_params(self._m_incl_pvalues, self._incl_pdescs, self._tilted)

        if self._rptraits:
            prepare_traits_params(
                self._m_rpt_pvalues, self._rpt_pdescs, self._rpt_pnames,
                self._rptraits)
        if self._rhtraits:
            prepare_traits_params(
                self._m_rht_pvalues, self._rht_pdescs, self._rht_pnames,
                self._rhtraits)
        if self._vptraits:
            prepare_traits_params(
                self._m_vpt_pvalues, self._vpt_pdescs, self._vpt_pnames,
                self._vptraits)
        if self._vhtraits:
            prepare_traits_params(
                self._m_vht_pvalues, self._vht_pdescs, self._vht_pnames,
                self._vhtraits)
        if self._dptraits:
            prepare_traits_params(
                self._m_dpt_pvalues, self._dpt_pdescs, self._dpt_pnames,
                self._dptraits)
        if self._dhtraits:
            prepare_traits_params(
                self._m_dht_pvalues, self._dht_pdescs, self._dht_pnames,
                self._dhtraits)
        if self._wptraits:
            prepare_traits_params(
                self._m_wpt_pvalues, self._wpt_pdescs, self._wpt_pnames,
                self._wptraits)
        if self._sptraits:
            prepare_traits_params(
                self._m_spt_pvalues, self._spt_pdescs, self._spt_pnames,
                self._sptraits)

        self._impl_evaluate(
            driver, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)

    @abc.abstractmethod
    def _impl_prepare(self, driver, dtype):
        pass

    @abc.abstractmethod
    def _impl_evaluate(
            self, backend, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass
