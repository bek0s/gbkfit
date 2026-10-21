"""
The tasks of GBKFIT: evaluate a configuration (eval), fit it (fit, and
resume or bootstrap a fit), plot a run of a fit (plot), and prepare data
(prep). Each is a function that writes its outputs to a directory.
Programs run them in-process, or in a process of their own with
`python -m gbkfit.tasks` (see __main__).
"""

from ._console import *
