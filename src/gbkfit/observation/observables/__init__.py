from gbkfit.utils import parseutils
from .core import *
from .image import *
from .lslit import *
from .mmaps import *
from .scube import *


observable_parser = parseutils.TypedParser(Observable, [
    Image,
    LSlit,
    MMaps,
    SCube])
