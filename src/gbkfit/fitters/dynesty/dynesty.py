
import abc
import logging

import dynesty
import numpy as np

import gbkfit.fitter


log = logging.getLogger(__name__)


class FitterDynesty(gbkfit.fitter.Fitter):

    def __init__(self):
        super().__init__()

    def _impl_fit(self, data, model, params):

        return None
