Basic Concepts
==============

The modelling strategy of GBKFIT can be described using a series of concepts
which are explained in the next sections.

Data models
-----------
- Image
- Spectral cube
- Moment maps
- Long-slit spectra

Galaxies
--------
A galaxy is a set of components, with optional mass and distance.

Components
^^^^^^^^^^
- The smooth disk
- The Monte Carlo disk
- The point

Each component emits light (in a band, seen by images, or in lines, seen
by spectral cubes and the data made from them), removes light (dust), or
both. The density of dust is its optical depth at V; an extinction law
(ccm, calzetti or a table) scales it at the rest wavelength of each line,
and without one the dust is grey.

Traits
^^^^^^

Lines of sight
--------------
- Flat: the galaxies are evaluated on the plane of the sky (thin disks
  and points)
- Grid: the galaxies are evaluated on a grid with a depth axis (thick
  disks, thin disks and points at their depth, and dust, which dims the
  light behind it)

Fitters
-------
Least-Squares
^^^^^^^^^^^^^
Swarm Intelligence
^^^^^^^^^^^^^^^^^^
Markov Chain Monte Carlo
^^^^^^^^^^^^^^^^^^^^^^^^
Nested Sampling
^^^^^^^^^^^^^^^

Drivers
-------
Host
^^^^
CUDA
^^^^
