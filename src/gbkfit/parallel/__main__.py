"""
The launcher of programs with workers of MPI (see WorkersMPI):
mpiexec -n N python -m gbkfit.parallel (file | -m module) [arguments].
"""

import sys

from .mpi import main

main(sys.argv[1:])
