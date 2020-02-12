
import abc

from gbkfit.utils import iterutils
from . import _common


class Disk(abc.ABC):

    def __init__(
            self,
            loose, tilted, rnodes,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits):

        if rnodes is None or len(rnodes) < 2:
            raise RuntimeError(
                "at least two radial nodes must be provided ")
        if rnodes is not None and not iterutils.is_ascending(rnodes):
            raise RuntimeError(
                "radial nodes must be ascending")
        if rnodes is not None and not iterutils.all_positive(rnodes):
            raise RuntimeError(
                "radial nodes must be positive")
        if rnodes is not None and not iterutils.all_unique(rnodes):
            raise RuntimeError(
                "radial nodes must be unique")

        rnodes = tuple(rnodes)
        nrnodes = len(rnodes)

        self._loose = bool(loose)
        self._tilted = bool(tilted)
        self._rnodes = rnodes
        self._nrnodes = nrnodes

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

        (self._rpt_uids,
         self._rpt_cvalues,
         self._rpt_ccounts,
         self._rpt_pcounts,
         self._rpt_pdescs,
         self._rpt_ponames,
         self._rpt_pnnames) = _common.trait_info(rptraits, 'rpt', nrnodes)
        (self._rht_uids,
         self._rht_cvalues,
         self._rht_ccounts,
         self._rht_pcounts,
         self._rht_pdescs,
         self._rht_ponames,
         self._rht_pnnames) = _common.trait_info(rhtraits, 'rht')
        (self._vpt_uids,
         self._vpt_cvalues,
         self._vpt_ccounts,
         self._vpt_pcounts,
         self._vpt_pdescs,
         self._vpt_ponames,
         self._vpt_pnnames) = _common.trait_info(vptraits, 'vpt', nrnodes)
        (self._vht_uids,
         self._vht_cvalues,
         self._vht_ccounts,
         self._vht_pcounts,
         self._vht_pdescs,
         self._vht_ponames,
         self._vht_pnnames) = _common.trait_info(vhtraits, 'vht')
        (self._dpt_uids,
         self._dpt_cvalues,
         self._dpt_ccounts,
         self._dpt_pcounts,
         self._dpt_pdescs,
         self._dpt_ponames,
         self._dpt_pnnames) = _common.trait_info(dptraits, 'dpt', nrnodes)
        (self._dht_uids,
         self._dht_cvalues,
         self._dht_ccounts,
         self._dht_pcounts,
         self._dht_pdescs,
         self._dht_ponames,
         self._dht_pnnames) = _common.trait_info(dhtraits, 'dht')
        (self._wpt_uids,
         self._wpt_cvalues,
         self._wpt_ccounts,
         self._wpt_pcounts,
         self._wpt_pdescs,
         self._wpt_ponames,
         self._wpt_pnnames) = _common.trait_info(wptraits, 'wpt', nrnodes)
        (self._spt_uids,
         self._spt_cvalues,
         self._spt_ccounts,
         self._spt_pcounts,
         self._spt_pdescs,
         self._spt_ponames,
         self._spt_pnnames) = _common.trait_info(sptraits, 'spt', nrnodes)

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

        self._m_rnodes = [None, None]
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
        self._backend = None

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

    def _prepare(self, backend, dtype):

        self._backend = backend
        self._dtype = dtype

        lcount = self._nrnodes if self._loose else 1
        tcount = self._nrnodes if self._tilted else 1

        if self._vptraits:
            self._m_vsys_pvalues = backend.mem_alloc(lcount, dtype)
        self._m_vsys_pvalues = backend.mem_alloc(lcount, dtype)
        self._m_xpos_pvalues = backend.mem_alloc(lcount, dtype)
        self._m_ypos_pvalues = backend.mem_alloc(lcount, dtype)
        self._m_posa_pvalues = backend.mem_alloc(tcount, dtype)
        self._m_incl_pvalues = backend.mem_alloc(tcount, dtype)

        _common.prepare_rnode_array(
            backend, dtype, self._m_rnodes, self._rnodes)

        if self._rptraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_rpt_uids, self._rpt_uids,
                self._m_rpt_ccounts, self._rpt_ccounts,
                self._m_rpt_pcounts, self._rpt_pcounts,
                self._m_rpt_cvalues, self._rpt_cvalues,
                self._m_rpt_pvalues)
        if self._rhtraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_rht_uids, self._rht_uids,
                self._m_rht_ccounts, self._rht_ccounts,
                self._m_rht_pcounts, self._rht_pcounts,
                self._m_rht_cvalues, self._rht_cvalues,
                self._m_rht_pvalues)
        if self._vptraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_vpt_uids, self._vpt_uids,
                self._m_vpt_ccounts, self._vpt_ccounts,
                self._m_vpt_pcounts, self._vpt_pcounts,
                self._m_vpt_cvalues, self._vpt_cvalues,
                self._m_vpt_pvalues)
        if self._vhtraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_vht_uids, self._vht_uids,
                self._m_vht_ccounts, self._vht_ccounts,
                self._m_vht_pcounts, self._vht_pcounts,
                self._m_vht_cvalues, self._vht_cvalues,
                self._m_vht_pvalues)
        if self._dptraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_dpt_uids, self._dpt_uids,
                self._m_dpt_ccounts, self._dpt_ccounts,
                self._m_dpt_pcounts, self._dpt_pcounts,
                self._m_dpt_cvalues, self._dpt_cvalues,
                self._m_dpt_pvalues)
        if self._dhtraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_dht_uids, self._dht_uids,
                self._m_dht_ccounts, self._dht_ccounts,
                self._m_dht_pcounts, self._dht_pcounts,
                self._m_dht_cvalues, self._dht_cvalues,
                self._m_dht_pvalues)
        if self._wptraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_wpt_uids, self._wpt_uids,
                self._m_wpt_ccounts, self._wpt_ccounts,
                self._m_wpt_pcounts, self._wpt_pcounts,
                self._m_wpt_cvalues, self._wpt_cvalues,
                self._m_wpt_pvalues)
        if self._sptraits:
            _common.prepare_trait_arrays(
                backend, dtype,
                self._m_spt_uids, self._spt_uids,
                self._m_spt_ccounts, self._spt_ccounts,
                self._m_spt_pcounts, self._spt_pcounts,
                self._m_spt_cvalues, self._spt_cvalues,
                self._m_spt_pvalues)

    def evaluate(
            self, backend, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):

        # Perform preparations if needed
        if self._backend is not backend or self._dtype is not dtype:
            self._prepare(backend, dtype)

        def prepare_param_array(array, descs):
            _common.prepare_param_array(backend, params, array, descs)

        prepare_param_array(self._m_vsys_pvalues, self._vsys_pdescs)
        prepare_param_array(self._m_xpos_pvalues, self._xpos_pdescs)
        prepare_param_array(self._m_ypos_pvalues, self._ypos_pdescs)
        prepare_param_array(self._m_posa_pvalues, self._posa_pdescs)
        prepare_param_array(self._m_incl_pvalues, self._incl_pdescs)
        if self._rpt_pdescs:
            prepare_param_array(self._m_rpt_pvalues, self._rpt_pdescs)
        if self._rht_pdescs:
            prepare_param_array(self._m_rht_pvalues, self._rht_pdescs)
        if self._vpt_pdescs:
            prepare_param_array(self._m_vpt_pvalues, self._vpt_pdescs)
        if self._vht_pdescs:
            prepare_param_array(self._m_vht_pvalues, self._vht_pdescs)
        if self._dpt_pdescs:
            prepare_param_array(self._m_dpt_pvalues, self._dpt_pdescs)
        if self._dht_pdescs:
            prepare_param_array(self._m_dht_pvalues, self._dht_pdescs)
        if self._wpt_pdescs:
            prepare_param_array(self._m_wpt_pvalues, self._wpt_pdescs)
        if self._spt_pdescs:
            prepare_param_array(self._m_spt_pvalues, self._spt_pdescs)

        self._evaluate_impl(
            backend, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)

    @abc.abstractmethod
    def _evaluate_impl(
            self, backend, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass
