======
GBKFIT
======

|cicd| |docs|

GBKFIT is software for modelling the surface brightness and kinematics of
galaxies, and for fitting the models to observations. It is a Python
library (with C++ and CUDA kernels), used from Python scripts, from the
command line (``gbkfit-cli``), or in the browser with the Studio
(``gbkfit-studio``): a web interface to set up models, preview them live,
fit them and see their results. The Studio runs on the GBKFIT server
(``gbkfit-server``), which keeps projects and runs fits over HTTP. An MCP
server, for AI assistants, is planned.

Its models are forward models: they predict the observations, through the
instrument, and are compared with the data. They range from kinematic
models (disks, clumpy disks, outflow cones, point sources, or the particles
of a simulation) to mass models, whose rotation follows from the mass in
Newtonian gravity, MOND or f(R) gravity, and dynamical models of spheroids
(Jeans and orbit-based models).

GBKFIT fits images, spectral cubes and moment maps, long-slit and slitless
spectra, spectra or moments of regions (e.g. fibres or Voronoi bins),
profiles (e.g. rotation curves) and interferometric visibilities. It uses
optimisers and samplers such as least squares, CMA-ES, MCMC and nested
sampling, on multi-core CPUs and NVIDIA GPUs, and across the machines of a
cluster (with MPI, Ray or Dask).

GBKFIT is in its Alpha phase: expect incomplete features and bugs.

Installation
============

GBKFIT is on the Test instance of PyPI for now:

.. code-block:: console

   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple gbkfit

See the `installation guide <https://gbkfit.readthedocs.io/en/latest/install.html>`_
for GPU support, optional packages and building from source.

Documentation
=============

The documentation is at `gbkfit.readthedocs.io <https://gbkfit.readthedocs.io>`_.

Citing
======

If you use GBKFIT in your research, please cite Bekiaris et al. (2027, in
prep.).

License
=======

GBKFIT is licensed under the BSD 3-Clause License (see ``LICENSE``).

Legacy version
==============

The old version of GBKFIT is at
`gbkfit-legacy <https://github.com/bek0s/gbkfit-legacy>`_.

.. |cicd| image:: https://github.com/bek0s/gbkfit/actions/workflows/cicd.yml/badge.svg
   :target: https://github.com/bek0s/gbkfit/actions/workflows/cicd.yml
   :alt: CI/CD status

.. |docs| image:: https://readthedocs.org/projects/gbkfit/badge/?version=latest
   :target: https://gbkfit.readthedocs.io/en/latest/
   :alt: Documentation status
