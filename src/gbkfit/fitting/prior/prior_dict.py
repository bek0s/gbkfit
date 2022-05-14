
import collections.abc
import copy

import numpy as np


class PriorDict(collections.abc.Mapping):

    def __init__(self, priors, descs):
        self._priors = copy.deepcopy(priors)
        self._descs = copy.deepcopy(descs)

    def __getitem__(self, key):
        return self._priors.__getitem__(key)

    def __iter__(self):
        return self._priors.__iter__()

    def __len__(self):
        return self._priors.__len__()

    def __repr__(self):
        return self._priors.__repr__()

    def __str__(self):
        return self._priors.__str__()

    def sample(self, keys, size=None):
        return [self[k].sample(size) for k in keys]

    def rescale(self, sample):
        return [self[k].rescale(v) for k, v in sample.items()]

    def prob(self, sample):
        return np.product([self[k].prob(v) for k, v in sample.items()])

    def log_prob(self, sample):
        return np.product([self[k].log_prob(v) for k, v in sample.items()])
