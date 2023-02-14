
from gbkfit.utils import parseutils

from .prior_dict import *
from .priors import *


prior_parser = parseutils.TypedParser(Prior, [
    PriorUniform,
    PriorGauss,
    PriorGaussTrunc
])
