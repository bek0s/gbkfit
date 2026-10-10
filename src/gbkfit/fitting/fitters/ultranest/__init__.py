try:
    import ultranest
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "the ultranest fitter needs the optional package ultranest: "
        "pip install 'gbkfit[ultranest]'") from e

from .core import *
