"""
Workers that evaluate an objective at the same time, each on a copy of
it (see Workers): e.g. the batches of points of a fit (see Fitter.fit),
and its sub-fits (e.g. the refits of a bootstrap; see run_subfits), in
threads of this process (e.g. one per GPU), in processes of this
machine, or on the machines of a cluster (MPI, Ray, Dask; their modules
are optional, see requires).

Run as a module, it is the launcher of programs with workers of MPI (see
WorkersMPI): mpiexec -n N python -m gbkfit.parallel (file | -m module)
[arguments].
"""

from .dask import *
from .mpi import *
from .pool import *
from .processes import *
from .ray import *
from .threads import *


# The kinds of workers, those that need optional packages too: see their
# requires
workers_parser.register([
    WorkersDask, WorkersMPI, WorkersProcesses, WorkersRay, WorkersThreads])
