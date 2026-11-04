"""
Parsing of configurations.

A configuration describes an object with a dict of options: the
parameters of a constructor of its class, __init__ or a named one (e.g.
from_file; see Serializable.config_constructors). A configuration with
the option 'type' describes an object of one of several classes (e.g. a
component), which is chosen by its type. The errors in a configuration
have the path to the part with the error (see ConfigError).

The catalogue of the families and the schema of configurations are in
the modules catalogue and schema, which import the whole library.
"""

from .errors import *
from .options import *
from .parsers import *
from .paths import *
from .serializable import *
from .types import *
