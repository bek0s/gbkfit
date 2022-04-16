
import collections.abc
import copy

import numpy as np

from gbkfit.fitting.prior import *
from gbkfit.utils import iterutils, miscutils


class PriorDict(collections.abc.Mapping):

    def __init__(self, descs, priors_, conversions=None):
        self._descs = descs
        self._dict = dict(priors_)
        self._conversions_obj = conversions
        self._conversions_src = miscutils.get_source(conversions)
        self._enames = None

    def evaluate(self, eparams, check=True):

        if check and (missing := set(enames_all).difference(eparams)):
            raise RuntimeError()

        if check and (unknown := set(eparams).difference(enames_all)):
            raise RuntimeError()

        if self._conversions_obj is None:
            return

        try:
            result = copy.deepcopy()
            conversions_obj(self, priors)
        except Exception as e:
            raise RuntimeError()

        for k, v in result.items():

            def is_prior(x): return isinstance(x, Prior)

            if any([not is_prior(x) for x in iterutils.tuplify(v)]):
                raise RuntimeError()

            if not isinstance(v, Prior):
                pass

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return self._dict.__len__()

    def __repr__(self):
        return self._dict.__repr__()

    def __str__(self):
        return self._dict.__str__()

    def sample(self, size=None):
        pass

    def prob(self, keys, vals):
        return np.product(
            [self[key].prob(val) for key, val in zip(keys, vals)])

    def log_prob(self, keys, vals):
        return np.product(
            [self[key].log_prob(val) for key, val in zip(keys, vals)])

    def rescale(self, keys, vals):
        return [self[key].rescale(val) for key, val in zip(keys, vals)]
