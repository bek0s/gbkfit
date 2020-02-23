
import copy
import logging

import asteval
import numpy as np

import gbkfit.params
from gbkfit.broker.brokers import BrokerNone
from gbkfit.driver.drivers import DriverHost
from gbkfit.utils import iterutils

log = logging.getLogger(__name__)


class Model:

    def __init__(self, dmodels, gmodels, drivers, brokers):

        if not drivers:
            drivers = DriverHost()
        if not brokers:
            brokers = BrokerNone()

        dmodels = iterutils.listify(dmodels)
        gmodels = iterutils.listify(gmodels)
        drivers = iterutils.listify(drivers)
        brokers = iterutils.listify(brokers)

        ndmodels = len(dmodels)
        ngmodels = len(gmodels)
        ndrivers = len(drivers)
        nbrokers = len(brokers)

        if ndmodels != ngmodels:
            raise RuntimeError(
                f"The number of gmodels ({ngmodels}) "
                f"must be equal to the number of dmodels ({ndmodels}).")

        if ((nbrokers != ndmodels and nbrokers != 1)
                or (ndrivers != ndmodels and ndrivers != 1)):
            raise RuntimeError(
                f"The number of brokers ({nbrokers}) and drivers ({ndrivers}) "
                f"must be equal to the number of dmodels ({ndmodels}) or 1.")

        if len(drivers) == 1 and ndmodels > 1:
            drivers = iterutils.make_list((ndmodels,), drivers[0], True)

        if len(brokers) == 1 and ndmodels > 1:
            brokers = iterutils.make_list((ndmodels,), brokers[0], True)

        self._nmodels = ndrivers
        self._brokers = tuple(brokers)
        self._drivers = tuple(drivers)
        self._dmodels = tuple(dmodels)
        self._gmodels = tuple(gmodels)

        self._mappings = []
        self._param_descs = {}
        self._param_exprs = {}
        self._param_enames_all = []
        self._param_enames_free = []
        self._param_enames_fixed = []

        self._reset_param_mappings()
        self._reset_param_names()
        self._reset_interpreter()

    def drivers(self):
        return self._drivers

    def dmodels(self):
        return self._dmodels

    def gmodels(self):
        return self._gmodels

    def get_param_descs(self):
        return dict(self._param_descs)

    def get_param_exprs(self):
        return dict(self._param_exprs)

    def set_param_exprs(self, exprs):
        self._reset_param_mappings()
        self._reset_param_names()
        self._reset_interpreter()
        self._param_exprs, enames = self._prepare_param_exprs(exprs)
        for ename in enames:
            self._param_enames_free.remove(ename)
            self._param_enames_fixed.append(ename)

    def get_param_names(self, free=True, fixed=False):
        return self._param_enames_free * free + self._param_enames_fixed * fixed

    def evaluate(
            self, params, explode_params, check_params=True,
            out_dextra=None, out_gextra=None, out_params=None,
            out_eparams=None, out_eparams_free=None, out_eparams_fixed=None):

        def is_scalar(x):
            success = True
            try:
                float(x)
            except TypeError:
                success = False
            return success

        def is_vector(x):
            return isinstance(x, (tuple, list, np.ndarray)) \
                   and iterutils.is_sequence_of_type(x, (int, float, np.number))

        # Explode parameters if requested.
        # Otherwise, assume they are already exploded.
        eparams = self._prepare_param_values(params) \
            if explode_params else dict(params)

        # Check for undefined and already-fixed exploded parameters
        if check_params:
            self._check_param_enames(eparams.keys())

        # Set exploded parameter values
        for ename in self._param_enames_free:
            self._expreval(f'{ename}={eparams[ename]}')

        # Apply expressions
        for lhs_expr, rhs_expr in self._param_exprs.items():
            lhs_value = self._expreval(lhs_expr)
            rhs_value = self._expreval(rhs_expr)
            full_expr = f'{lhs_expr}={rhs_expr}'
            if self._expreval.error:
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"exception thrown while evaluating rhs; "
                    f"exception type: {self._expreval.error[0].exc.__name__}; "
                    f"exception message: {self._expreval.error[0].msg}")
            if not (is_scalar(rhs_value) or is_vector(rhs_value)):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"the rhs value after evaluation is invalid: '{rhs_value}'")
            if (is_scalar(lhs_value)
                    and is_vector(rhs_value)
                    and 1 < len(rhs_value)):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"cannot assign sequence of size {len(rhs_value)} "
                    f"to scalar")
            if (is_vector(lhs_value)
                    and is_vector(rhs_value)
                    and 1 < len(rhs_value) != len(lhs_value)):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"cannot assign sequence of size {len(rhs_value)} "
                    f"to sequence of size {len(lhs_value)}")
            self._expreval(f'{lhs_expr}={rhs_value}')

        # Build the final parameter dict from the symbol table
        params_all = {k: self._expreval.symtable[k] for k in self._param_descs}

        # Request model evaluation
        request_dextra = out_dextra is not None
        request_gextra = out_gextra is not None
        for i in range(self._nmodels):
            broker = self._brokers[i]
            driver = self._drivers[i]
            dmodel = self._dmodels[i]
            gmodel = self._gmodels[i]
            mapping = self._mappings[i]
            iparams = {p: params_all[mapping[p]] for p in gmodel.params()}
            broker.evaluate(
                driver, dmodel, gmodel, iparams,
                request_dextra, request_gextra)

        # Retrieve results as soon as they are ready
        outputs = []
        for i in range(self._nmodels):
            broker = self._brokers[i]
            ioutput, iout_dextra, iout_gextra = broker.output()
            if out_dextra is not None:
                out_dextra.append(iout_dextra)
            if out_gextra is not None:
                out_gextra.append(iout_gextra)
            outputs.append(ioutput)

        # Retrieve parameter values
        if out_params is not None:
            out_params.clear()
            out_params.update(copy.deepcopy(params_all))
        if out_eparams is not None:
            out_eparams.clear()
            out_eparams.update({
                p: self._expreval(p) for p in self._param_enames_all})
        if out_eparams_free is not None:
            out_eparams_free.clear()
            out_eparams_free.update({
                p: self._expreval(p) for p in self._param_enames_free})
        if out_eparams_fixed is not None:
            out_eparams_fixed.clear()
            out_eparams_fixed.update({
                p: self._expreval(p) for p in self._param_enames_fixed})

        return outputs

    def _reset_param_mappings(self):
        self._mappings.clear()
        self._param_descs.clear()
        for i, gmodel in enumerate(self._gmodels):
            self._mappings.append({})
            for old_name, desc in gmodel.params().items():
                new_name = f'model{i}_{old_name}' if i else old_name
                self._mappings[i][old_name] = new_name
                self._param_descs[new_name] = desc

    def _reset_param_names(self):
        self._param_exprs.clear()
        self._param_enames_all.clear()
        self._param_enames_free.clear()
        self._param_enames_fixed.clear()
        for name, desc in self._param_descs.items():
            for i in range(desc.size()):
                ename = name if desc.is_scalar() \
                    else gbkfit.params.make_param_symbol(name, i)
                self._param_enames_all.append(ename)
                self._param_enames_free.append(ename)

    def _reset_interpreter(self):

        class StdWriter:
            @staticmethod
            def write(msg):
                log.warning(msg)

        class ErrWriter:
            @staticmethod
            def write(msg):
                log.error(msg)
        """
        self._expreval = asteval.Interpreter(
            minimal=True, use_numpy=False,
            writer=StdWriter, err_writer=ErrWriter)
        """
        self._expreval = asteval.Interpreter(
            minimal=True, use_numpy=False)

        for name, desc in self._param_descs.items():
            self._expreval.symtable[name] = np.nan if desc.is_scalar() \
                else np.full(desc.size(), np.nan)

    def _prepare_param_exprs(self, params):

        # Parse and validate expressions
        keys, values, key_names, key_indices_list = \
            gbkfit.params.parse_param_exprs(params, self._param_descs)

        # Make sure all whole vector params are followed by [:]
        for i, (key, key_name) in enumerate(zip(keys, key_names)):
            is_vector_desc = self._param_descs[key_name].is_vector()
            is_vector_name = gbkfit.params._is_param_symbol_vector(key)
            if is_vector_desc and not is_vector_name:
                keys[i] += '[:]'

        # Exploded parameter names
        enames = gbkfit.params.explode_param_symbols(
            key_names, key_indices_list)

        return dict(zip(keys, values)), enames

    def _prepare_param_values(self, params):

        # Parse and validate values
        keys, values, key_names, key_indices_list = \
            gbkfit.params.parse_param_values(params, self._param_descs)

        # Exploded parameter names
        enames = gbkfit.params.explode_param_symbols(
            key_names, key_indices_list)

        return dict(zip(enames, iterutils.flatten(values)))

    def _check_param_enames(self, param_enames):

        # Check for missing parameters
        missing = set(self._param_enames_free).difference(set(param_enames))
        if missing:
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{missing}")

        # Check if any of the supplied parameters is fixed.
        isfixed = set(self._param_enames_fixed).intersection(set(param_enames))
        if isfixed:
            raise RuntimeError(
                f"the following supplied parameters are fixed: "
                f"{isfixed}")

    #def __deepcopy__(self, memodict={}):
