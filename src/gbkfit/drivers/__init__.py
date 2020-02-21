
try:
    from .cuda.driver import BackendCUDA
except ImportError:
    pass

from .openmp.driver import BackendOpenMP