"""
The parameters of models: their descriptions, the modes of vector
parameters, the solves of solved elements, the priors of free elements
(and on several parameters), and the space of the parameters of a
configuration, which fixes, ties, frees or solves each of them and
evaluates them.
"""

from .modes import *
from .pdescs import *
from .priors import *
from .solves import *
from .space import *
from .joint_priors import *
