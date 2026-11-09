"""
Parsing of configurations.

A configuration describes an object with a dict of options: the
parameters of a constructor of its class, __init__ or a named one (e.g.
from_file; see Serializable.config_constructors). A configuration with
the option 'type' describes an object of one of several classes (e.g. a
component), which is chosen by its type. The errors in a configuration
have the path to the part with the error (see ConfigError).

The families of configurations are those that their parsers name (see
Family and TypedParser); their catalogue, which imports the whole
library, is gbkfit.catalogue. The schema of configurations is written by
the module schema (see gbkfit.schema).
"""

from .errors import *
from .families import *
from .options import *
from .parsers import *
from .paths import *
from .serializable import *
from .types import *
from .version import *
