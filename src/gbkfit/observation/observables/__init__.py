from gbkfit.utils import parseutils
from .aspec import *
from .core import *
from .image import *
from .lslit import *
from .mmaps import *
from .scube import *


observable_parser = parseutils.TypedParser(Observable, [
    ASpec,
    Image,
    LSlit,
    MMaps,
    SCube])
