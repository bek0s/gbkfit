
from gbkfit.utils import parseutils

from .prior_dict import *
from .priors import *


prior_parser = parseutils.TypedParser(Prior)

prior_parser.register(PriorUniform)
prior_parser.register(PriorGauss)
prior_parser.register(PriorGaussTrunc)
