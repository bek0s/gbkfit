
try:
    import pygmo
except ImportError:
    raise ImportError(
        "to use the pygmo fitter please install pygmo")

from .pygmo import FitterPygmo
