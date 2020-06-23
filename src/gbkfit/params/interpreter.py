
import copy
import logging
import numbers

import asteval
import numpy as np

from . import utils


log = logging.getLogger(__name__)


class _StdWriter:
    @staticmethod
    def write(msg):
        log.warning(msg)


class _ErrWriter:
    @staticmethod
    def write(msg):
        log.error(msg)


class ParamInterpreter:

    def __init__(self, descs, exprs=None, dtype=np.float32):
        descs = copy.deepcopy(descs)
        exprs = copy.deepcopy(exprs) if exprs else {}
        self._descs = descs
        self._exprs = exprs
        self._dtype = dtype
        # Setup interpreter
        self._interpr = asteval.Interpreter(
            minimal=True, use_numpy=False,
            writer=_StdWriter, err_writer=_ErrWriter)
        for name, desc in descs.items():
            self._interpr.symtable[name] = np.nan if desc.is_scalar() \
                else np.full(desc.size(), np.nan, dtype)
        # Setup exploded parameter names
        self._eparams_all = []
        self._eparams_free = []
        self._eparams_fixed = []
        for name, desc in descs.items():
            for i in range(desc.size()):
                eparam = name if desc.is_scalar() \
                    else utils.make_param_symbol(name, i)
                self._eparams_all.append(eparam)
                self._eparams_free.append(eparam)
        # Parse and validate expressions
        (self._expr_keys,
         self._expr_values,
         self._expr_names,
         self._expr_indices,
         self._expr_asts) = utils.parse_param_exprs(exprs, descs)
        # Adjust exploded parameter names based on the expressions
        for ename in utils.explode_params(self._expr_names, self._expr_indices):
            self._eparams_free.remove(ename)
            self._eparams_fixed.append(ename)

    def get_param_descs(self):
        return self._descs

    def get_param_exprs(self):
        return self._exprs

    def get_param_names(self, free=True, fixed=False):
        return self._eparams_free * free + self._eparams_fixed * fixed

    def evaluate(
            self, eparams, check=True,
            out_eparams=None, out_eparams_free=None, out_eparams_fixed=None):

        # Ensure all supplied exploded parameters are valid
        if check:
            self._check_eparams(eparams.keys())

        # Set exploded parameter values
        for eparam in self._eparams_free:
            self._interpr(f'{eparam}={eparams[eparam]}')

        # Apply expressions
        for lhs_str, rhs_str, lhs_name, lhs_indices, rhs_ast in zip(
                self._expr_keys, self._expr_values,
                self._expr_names, self._expr_indices,
                self._expr_asts):

            full_expr = f'{lhs_str} = {rhs_str}'

            # Execute the expression and check for errors
            rhs_value = self._interpr.run(rhs_ast)
            if self._interpr.error:
                exception_type = self._interpr.error[0].exc.__name__
                exception_message = self._interpr.error[0].msg
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"exception thrown while evaluating rhs; "
                    f"exception type: {exception_type}; "
                    f"exception message: {exception_message}")

            # Create a bunch of convenience variables
            lhs_is_scalar = isinstance(lhs_indices, (type(None), int))
            lhs_is_vector = isinstance(lhs_indices, (tuple, list, np.ndarray))
            rhs_is_scalar = isinstance(rhs_value, numbers.Number)
            rhs_is_vector = isinstance(rhs_value, (tuple, list, np.ndarray))

            # Make sure expression lhs and rhs are compatible
            if not (rhs_is_scalar or rhs_is_vector):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"the rhs value after evaluation is invalid: '{rhs_value}'")
            if (lhs_is_scalar
                    and rhs_is_vector
                    and 1 < len(rhs_value)):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"cannot assign sequence of size {len(rhs_value)} "
                    f"to scalar")
            if (lhs_is_vector
                    and rhs_is_vector
                    and 1 < len(rhs_value) != len(lhs_indices)):
                raise RuntimeError(
                    f"problem with expression '{full_expr}'; "
                    f"cannot assign sequence of size {len(rhs_value)} "
                    f"to sequence of size {len(lhs_indices)}")

            # lhs and rhs are valid and compatible with each other
            if self._descs[lhs_name].is_scalar():
                self._interpr.symtable[lhs_name] = rhs_value
            else:
                self._interpr.symtable[lhs_name][lhs_indices] = rhs_value

        # Retrieve exploded (free or fixed) parameter dicts
        if out_eparams is not None:
            out_eparams.clear()
            out_eparams.update({
                p: self._interpr(p) for p in self._eparams_all})
        if out_eparams_free is not None:
            out_eparams_free.clear()
            out_eparams_free.update({
                p: self._interpr(p) for p in self._eparams_free})
        if out_eparams_fixed is not None:
            out_eparams_fixed.clear()
            out_eparams_fixed.update({
                p: self._interpr(p) for p in self._eparams_fixed})

        # Build the final parameter dict
        return {k: self._interpr.symtable[k] for k in self._descs}

    def _check_eparams(self, eparams):
        missing = set(self._eparams_free).difference(eparams)
        if missing:
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{missing}")
        fixed = set(self._eparams_fixed).intersection(eparams)
        if fixed:
            raise RuntimeError(
                f"the following parameters are supposed to be fixed: "
                f"{fixed}")
        unknown = set(eparams).difference(self._eparams_all)
        if unknown:
            raise RuntimeError(
                f"the following parameters are not recognised: "
                f"{unknown}")
