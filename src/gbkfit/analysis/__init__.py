"""
Analysis tools: measurements of maps that need no model of the galaxy
(e.g. of the data, the models or the residuals of a fit), on regular maps
or on bins: the asymmetry about the axes of a galaxy, its global
kinematic position angle, kinemetry, its pattern speed, and its
position-velocity diagrams with the rotation traced on them; the
structure function of maps (e.g. of residuals); the information that a
fit gains on its parameters; and the predictive accuracy of fits, to
compare models (PSIS-LOO and WAIC).
"""

from .asymmetry import *
from .information import *
from .kinematic_pa import *
from .kinemetry import *
from .pattern_speed import *
from .predictive import *
from .pv import *
from .structure import *
